# %% Load libraries ------------------------------------------------------------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import geoopt
from tqdm import tqdm
from scipy import sparse

# %% Loading data -------------------------------------------------------------
paper_table = pd.read_csv("../../data/preprocessed/paper_table.csv")
paper_concept_table = pd.read_csv("../../data/preprocessed/paper_concept_table.csv")
concept_table = pd.read_csv("../../data/preprocessed/concept_table.csv")
#data = np.load("../../data/derived/embeddings/embeddings-2025-0220.npz")
data = np.load("../../data/derived/embeddings/embeddings.npz")
paper_author_table = pd.read_csv("../../data/preprocessed/author_paper_table.csv")
concept_table = pd.read_csv("../../data/preprocessed/concept_table.csv")
paper_ids = data["paper_ids"]
embeddings_query = data["embeddings_query"].astype(np.float32)
embeddings_key = data["embeddings_key"].astype(np.float32)


# %% Preparation --------------------------------------------------------------
def create_author_paper_matrix(paper_table, paper_author_table):
    """Create author-paper matrix and compute paper weights based on author counts.

    Args:
        paper_table: DataFrame containing paper information
        paper_author_table: DataFrame containing paper-author relationships

    Returns:
        Tuple containing:
        - author2paper: Sparse matrix mapping authors to papers
        - weight_paper: Array of paper weights based on author counts
    """
    years = paper_table["year"].values
    n_papers = len(paper_table)
    n_authors = int(paper_author_table["author_id"].max() + 1)

    # Create author-paper matrix
    print("Creating author-paper matrix...")
    author2paper = sparse.csr_matrix(
        (
            np.ones(len(paper_author_table)),
            (
                paper_author_table["author_id"].values,
                paper_author_table["paper_id"].values,
            ),
        ),
        shape=(n_authors, n_papers),
    )

    n_authors_per_paper = np.array(author2paper.sum(axis=0)).flatten()
    weight_paper = n_authors_per_paper / n_authors_per_paper.sum()

    return author2paper, weight_paper


# Compute density using Lorentz distances
def poincare_to_lorentz(xy):
    """Convert (x, y) in Poincaré disk to (z_0, z_1, z_2) in Lorentz model."""
    x, y = xy[..., 0], xy[..., 1]
    denom = 1 - x**2 - y**2
    z0 = (1 + x**2 + y**2) / denom
    z1 = (2 * x) / denom
    z2 = (2 * y) / denom
    return torch.stack([z0, z1, z2], dim=-1)


def lorentz_to_poincare(xyz):
    """Convert (z_0, z_1, z_2) in Lorentz model to (x, y) in Poincaré disk."""
    z0, z1, z2 = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    denom = z0 + 1  # Avoid division by zero
    x = z1 / denom
    y = z2 / denom
    return torch.stack([x, y], dim=-1)


def compute_density_grid(
    embeddings_query,
    embeddings_key,
    weight_papers,
    grid_size=40,
    r=1.2,
    sigma=1.0,
    batch_size=1000,
):
    """Compute density values on a grid in the Poincaré disk.

    Args:
        embeddings: Numpy array of embeddings
        weight_papers: Weights for each embedding point
        grid_size: Number of grid points in each dimension
        r: Radius of the grid in the Poincaré disk
        sigma: Bandwidth parameter for density estimation

    Returns:
        Numpy array of density values on the grid
    """
    # Create grid of points in Poincaré disk
    focal_points_on_poincare_disk = torch.tensor(
        np.array(
            [
                (x, y)
                for x, y in np.mgrid[-r : r : grid_size * 1j, -r : r : grid_size * 1j]
                .reshape(2, -1)
                .T
            ]
        ),
        dtype=torch.float64,
    )

    # Convert points to Lorentz model
    focal_points_on_lorentz_manifold = poincare_to_lorentz(
        focal_points_on_poincare_disk
    )

    # Compute densities
    n = len(focal_points_on_lorentz_manifold)
    densities = torch.zeros(n, 2, dtype=torch.float64)

    # Compute in batches to avoid memory issues
    manifold = geoopt.Lorentz()
    all_points_query = embeddings_query.unsqueeze(0)  # Shape: [1, n, dim]
    all_points_key = embeddings_key.unsqueeze(0)  # Shape: [1, n, dim]
    for i in tqdm(range(0, n, batch_size)):
        batch_end = min(i + batch_size, n)
        # Get batch of points and all points
        batch_points = focal_points_on_lorentz_manifold[i:batch_end, :].unsqueeze(
            1
        )  # Shape: [batch, 1, dim]

        # Compute hyperbolic distance using arcosh
        distances_query = manifold.dist2(batch_points, all_points_query).squeeze()
        distances_key = manifold.dist2(batch_points, all_points_key).squeeze()

        # Replace NaN values with large distances that will result in near-zero kernel values
        distances_query = torch.where(
            torch.isnan(distances_query), torch.tensor(1e10), distances_query
        )
        distances_key = torch.where(
            torch.isnan(distances_key), torch.tensor(1e10), distances_key
        )

        # Compute kernel values using the hyperbolic distances
        kernel_values_query = torch.exp(-distances_query / sigma)
        kernel_values_key = torch.exp(-distances_key / sigma)

        # Weight the kernel values and sum for density
        weighted_kernel_query = kernel_values_query * weight_papers
        weighted_kernel_key = kernel_values_key * weight_papers

        densities[i:batch_end, 0] = torch.sum(weighted_kernel_query, dim=1)
        densities[i:batch_end, 1] = torch.sum(weighted_kernel_key, dim=1)

    return focal_points_on_poincare_disk, densities


def get_paper_concepts(
    paper_concept_table, concept_table, paper_table, num_top_concepts=8
):
    """Get primary concepts for each paper and identify most frequent concepts.

    Args:
        paper_concept_table: DataFrame with paper-concept scores
        concept_table: DataFrame with concept metadata
        paper_table: DataFrame with paper metadata
        num_top_concepts: Number of top concepts to consider

    Returns:
        tuple: (concept_categories, papers_with_concepts)
            - concept_categories: Index of top concept names plus "Other"
            - papers_with_concepts: DataFrame of papers with their primary concepts
    """
    # Add concept names to paper-concept scores
    paper_concepts_with_names = paper_concept_table.merge(
        concept_table[["id", "display_name"]],
        left_on="concept_id",
        right_on="id",
        how="left",
    )

    # Get highest scoring concept for each paper
    paper_primary_concepts = (
        paper_concepts_with_names.sort_values("score", ascending=False)
        .groupby("paper_id")
        .first()
        .reset_index()
    )

    # Get most frequent primary concepts
    concept_frequencies = paper_primary_concepts["display_name"].value_counts()
    most_frequent_concepts = concept_frequencies.nlargest(num_top_concepts)

    # Create final concept categories including "Other"
    concept_categories = pd.Index(list(most_frequent_concepts.index) + ["Other"])

    # Add primary concept to each paper's metadata
    papers_with_concepts = paper_table.merge(
        paper_primary_concepts, on="paper_id", how="left"
    )

    return concept_categories, papers_with_concepts

grid_size = 40

embeddings_query_tensor = torch.tensor(embeddings_query, dtype=torch.float64)
embeddings_key_tensor = torch.tensor(embeddings_key, dtype=torch.float64)

xy_query = lorentz_to_poincare(embeddings_query_tensor)
xy_key = lorentz_to_poincare(embeddings_key_tensor)

author2paper, weight_papers = create_author_paper_matrix(
    paper_table, paper_author_table
)

focal_points_on_poincare_disk, point_densities = compute_density_grid(
    embeddings_query_tensor, embeddings_key_tensor, weight_papers, grid_size=grid_size
)
top_concepts, paper_table_extended = get_paper_concepts(
    paper_concept_table, concept_table, paper_table
)

# %% Plotting ------------------------------------------------------------------


# Create figure and axis
def plot_canvas(ax):

    # Add circle boundary of Poincaré disk
    circle = plt.Circle((0, 0), 1, fill=False, color="white", linestyle="--", alpha=0.5)
    ax.add_artist(circle)

    # Add radius circles
    # Convert equidistant points in Lorentz model to Poincare disk radii
    # In Lorentz model, points at equal distances lie on "spheres" centered at origin
    # We convert these to radii in Poincare disk using the mapping formula
    lorentz_distances = np.linspace(0, 3, 10)  # Equal distances in Lorentz space
    radii = np.tanh(lorentz_distances / 2)  # Convert to Poincare disk radii

    for r in radii:
        circle = plt.Circle(
            (0, 0),
            r,
            fill=False,
            color="gray",
            alpha=1.0,
            linestyle="--",
            linewidth=0.5,
        )
        ax.add_artist(circle)

    # Add angle lines
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)  # 12 evenly spaced angles
    max_radius = max(radii)
    for angle in angles:
        # Draw line from origin
        x = [0, max_radius * np.cos(angle)]
        y = [0, max_radius * np.sin(angle)]
        ax.plot(x, y, color="gray", alpha=0.2, linestyle=":", linewidth=0.5)
        # Add angle label (in degrees)
        degrees = int(np.degrees(angle))

    # Add x and y axis lines that stop at circle boundary
    ax.plot([-1, 1], [0, 0], color="gray", alpha=0.9, linestyle="-", linewidth=1.0)
    ax.plot([0, 0], [-1, 1], color="gray", alpha=0.9, linestyle="-", linewidth=1.0)
    return ax


def plot_density(ax, xy, point_densities, **contourf_kwargs):
    # Create mask for points outside unit circle
    grid_size = int(np.sqrt(len(xy)))
    x = xy[:, 0].reshape(grid_size, grid_size)
    y = xy[:, 1].reshape(grid_size, grid_size)
    mask = x**2 + y**2 > 1.0

    # Convert point_densities to numpy array if it's a tensor
    if torch.is_tensor(point_densities):
        point_densities = point_densities.numpy()

    # Reshape point_densities
    density_grid = point_densities.reshape(grid_size, grid_size)

    # Mask the density values
    masked_densities = np.ma.masked_array(density_grid, mask=mask)

    # Create the contour plot
    contour = ax.contourf(
        x, y, masked_densities, levels=100, antialiased=True, **contourf_kwargs
    )

    # Add dark overlay
    circle = plt.Circle((0, 0), 1, color="black", alpha=0.3)
    ax.add_artist(circle)

    return contour


def plot_arrows(ax, xy, point_densities, **contourf_kwargs):
    # Create mask for points outside unit circle
    grid_size = int(np.sqrt(len(xy)))
    x = xy[:, 0].reshape(grid_size, grid_size)
    y = xy[:, 1].reshape(grid_size, grid_size)
    mask = x**2 + y**2 > 1.0

    # Convert point_densities to numpy array if it's a tensor
    if torch.is_tensor(point_densities):
        point_densities = point_densities.numpy()

    # Reshape point_densities
    density_grid = point_densities.reshape(grid_size, grid_size)

    # Calculate gradients

    # Apply Gaussian smoothing before computing gradients
    from scipy.ndimage import gaussian_filter
    dy, dx = np.gradient(density_grid, edge_order=2)  # Use 8th order accurate gradient for higher precision

    # Create a coarser grid for the arrows
    arrow_spacing = grid_size // 30
    x_grid = x[::arrow_spacing, ::arrow_spacing]
    y_grid = y[::arrow_spacing, ::arrow_spacing]
    dx_grid = dx[::arrow_spacing, ::arrow_spacing]
    dy_grid = dy[::arrow_spacing, ::arrow_spacing]

    # Create mask for points inside unit circle
    mask_arrows = (x_grid**2 + y_grid**2 <= 1.0)

    # Project dx/dy vectors to Lorentz manifold and calculate magnitude
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    dx_flat = dx_grid.flatten()
    dy_flat = dy_grid.flatten()

    # Convert base points to Lorentz coordinates using poincare_to_lorentz
    points = torch.stack([x_flat, y_flat], dim=-1)
    lorentz_points = poincare_to_lorentz(points)  # (z0, z1, z2)

    # Convert points + dx/dy to Lorentz coordinates
    points_displaced = torch.stack([x_flat + dx_flat, y_flat + dy_flat], dim=-1)
    lorentz_points_displaced = poincare_to_lorentz(points_displaced)

    # Calculate vector in Lorentz space
    lorentz_vector = lorentz_points_displaced - lorentz_points

    # Calculate Minkowski inner product for magnitude
    magnitude = torch.sqrt(torch.abs(-lorentz_vector[:,0]**2 + lorentz_vector[:,1]**2 + lorentz_vector[:,2]**2))
    magnitude = magnitude.reshape(dx_grid.shape)

    # Create threshold mask
    threshold = np.percentile(magnitude[mask_arrows].flatten(), 25)
    magnitude_mask = magnitude > threshold

    # Combine masks
    final_mask = mask_arrows & magnitude_mask

    # Scale and normalize arrows
    arrow_scale = 0.03
    with np.errstate(divide='ignore', invalid='ignore'):
        dx_norm = np.where(magnitude > 0, dx_grid / magnitude * arrow_scale, 0)
        dy_norm = np.where(magnitude > 0, dy_grid / magnitude * arrow_scale, 0)

    # Plot arrows
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            if final_mask[i, j]:
                ax.arrow(
                    x_grid[i, j],
                    y_grid[i, j],
                    dx_norm[i, j],
                    dy_norm[i, j],
                    head_width=0.01,
                    head_length=0.01,
                    fc='white',
                    ec='white',
                    alpha=0.2
                )

    return ax


def plot_papers(ax, xy, top_concepts, paper_table_extended):
    # Define an improved color palette with a cosmic feel
    palette = {
        concept: color
        for concept, color in zip(top_concepts, [c for c in sns.color_palette("bright") if not (c[1] > c[0] and c[1] > c[2])])
    }
    edge_color_palette = {
        concept: color for concept, color in zip(top_concepts, sns.color_palette())
    }
    palette["Other"] = "#dfdfdf"

    top_concepts = pd.Index(list(top_concepts) + ["Other"])

    # Create color column - grey for papers not in top 5 concepts
    colors = paper_table_extended["display_name"].apply(
        lambda x: x if x in top_concepts else "Other"
    )

    # Create DataFrame with coordinates and colors
    plot_df = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "color": colors,
            "title": paper_table_extended["title"],
        }
    ).sort_values("color", ascending=False)

    # Plot points for each non-Other class
    sns.scatterplot(
        data=plot_df.groupby("color").sample(10000).query("color != 'Other'"),
        x="x",
        y="y",
        hue="color",
        palette=palette,
        # alpha=0.3,
        s=0.3,
        ax=ax,
        edgecolor=None,
        linewidth=0.1,
    )

    # Add contour lines around each class
    for concept in top_concepts:
        if concept == "Other":
            continue

        # Get points for this concept
        concept_points = plot_df[plot_df["color"] == concept]
        print(len(concept_points), concept)
        concept_points = concept_points.sample(10000)

        if len(concept_points) < 10:
            continue

        # Calculate point density using KDE
        from scipy.stats import gaussian_kde

        points = concept_points[["x", "y"]].values
        kde = gaussian_kde(points.T, bw_method=0.1)
        density = kde(points.T)

        # Filter to dense regions
        dense_points = points[density > np.percentile(density, 50)]

        if len(dense_points) < 10:
            continue

        # Create smooth hull
        from scipy.spatial import ConvexHull
        from scipy.interpolate import splprep, splev

        hull = ConvexHull(dense_points)
        hull_points = dense_points[hull.vertices]
        hull_points = np.append(hull_points, [hull_points[0]], axis=0)

        # Fit spline and generate smooth points
        tck, _ = splprep([hull_points[:, 0], hull_points[:, 1]], s=0.01, per=True)
        smooth_points = np.array(splev(np.linspace(0, 1, 2000), tck)).T

        # Add label at rightmost point
        # Find rightmost point along the boundary
        # Randomly choose which edge to place the label
        edge_choice = np.random.choice(['right', 'top', 'bottom', 'left'])

        if edge_choice == 'right':
            idx = np.argmax(smooth_points[:, 0])
            label_x = smooth_points[idx, 0] + 0.05
            label_y = smooth_points[idx, 1]
            valign = "center"
        elif edge_choice == 'top':
            idx = np.argmax(smooth_points[:, 1])
            label_x = smooth_points[idx, 0]
            label_y = smooth_points[idx, 1] + 0.05
            valign = "bottom"
        elif edge_choice == 'bottom':
            idx = np.argmin(smooth_points[:, 1])
            label_x = smooth_points[idx, 0]
            label_y = smooth_points[idx, 1] - 0.05
            valign = "top"
        else: # left
            idx = np.argmin(smooth_points[:, 0])
            label_x = smooth_points[idx, 0] - 0.05
            label_y = smooth_points[idx, 1]
            valign = "center"

        ax.text(
            label_x,
            label_y,
            concept,
            color="white",
            alpha=1.0,
            verticalalignment=valign,
            #path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground="black")]
        )


#    ax.legend(
#        bbox_to_anchor=(1.05, 1),
#        loc="upper left",
#        fontsize=12,  # Increase font size
#        labelspacing=1.2,  # Increase spacing between labels
#        markerscale=50,  # Scale up legend marker size
#        handletextpad=1.5,  # Space between marker and text
#        borderpad=1,  # Padding inside the legend box
#    )
    ax.legend().remove()
    ax.axis("off")

##  Potential difference in density --------------------------------------------
#sns.set_style("white")
#plt.style.use("dark_background")
#sns.set_context("notebook", font_scale=1.2)
#fig, ax = plt.subplots(figsize=(10, 10))
#
#potential_diff = point_densities[:, 0] / torch.sum(
#    point_densities[:, 0]
#) - point_densities[:, 1] / torch.sum(
#    point_densities[:, 1]
#)  # in-potential - out-potential
#
#plot_density(
#    ax=ax,
#    xy=focal_points_on_poincare_disk,
#    point_densities=potential_diff,
#    cmap=sns.color_palette("icefire", as_cmap=True),
#    vmin=-max(abs(torch.min(potential_diff)), abs(torch.max(potential_diff))),
#    vmax=max(abs(torch.min(potential_diff)), abs(torch.max(potential_diff))),
#)
#
#plot_canvas(ax)
#plot_arrows(ax, focal_points_on_poincare_disk, potential_diff)
## Potential difference in density without arrows--------------------------------------------
#sns.set_style("white")
#plt.style.use("dark_background")
#sns.set_context("notebook", font_scale=1.2)
#fig, ax = plt.subplots(figsize=(10, 10))
#
#potential_diff = point_densities[:, 0] / torch.sum(
#    point_densities[:, 0]
#) - point_densities[:, 1] / torch.sum(
#    point_densities[:, 1]
#)  # in-potential - out-potential
#
#plot_density(
#    ax=ax,
#    xy=focal_points_on_poincare_disk,
#    point_densities=potential_diff,
#    cmap=sns.color_palette("icefire", as_cmap=True),
#    vmin=-max(abs(torch.min(potential_diff)), abs(torch.max(potential_diff))),
#    vmax=max(abs(torch.min(potential_diff)), abs(torch.max(potential_diff))),
#)
#
#plot_canvas(ax)
#
## %% Scatter plot of papers
#sns.set_style("white")
#plt.style.use("dark_background")
#sns.set_context("notebook", font_scale=1.2)
#fig, ax = plt.subplots(figsize=(10, 10))
#
#plot_canvas(ax)
#plot_papers(ax, xy_query, top_concepts, paper_table_extended)
#
# % Scatter plot of papers with density

sns.set_style("white")
plt.style.use("dark_background")
sns.set_context("notebook", font_scale=1.2)
fig, ax = plt.subplots(figsize=(15, 15))

potential_diff = point_densities[:, 0] / torch.sum(
    point_densities[:, 0]
) - point_densities[:, 1] / torch.sum(
    point_densities[:, 1]
)  # in-potential - out-potential

plot_density(
    ax=ax,
    xy=focal_points_on_poincare_disk,
    point_densities=potential_diff,
    cmap=sns.color_palette("icefire", as_cmap=True),
    vmin=-max(abs(torch.min(potential_diff)), abs(torch.max(potential_diff))),
    vmax=max(abs(torch.min(potential_diff)), abs(torch.max(potential_diff))),
)
plot_canvas(ax)
circle = plt.Circle((0, 0), 1, color="black", alpha=0.3)
ax.add_artist(circle)
plot_arrows(ax, focal_points_on_poincare_disk, potential_diff)
plot_papers(ax, xy_query, top_concepts, paper_table_extended)
#fig.savefig("lorenz-embedding-potential-diff.png", dpi=500)
# %%


# %%
xy = focal_points_on_poincare_disk
z = point_densities[:, 0] - point_densities[:, 1]
grid_size = int(np.sqrt(len(xy)))
print(grid_size)
x = xy[:, 0].reshape(grid_size, grid_size)
y = xy[:, 1].reshape(grid_size, grid_size)
mask = x**2 + y**2 > 1.0

# Convert point_densities to numpy array if it's a tensor
if torch.is_tensor(z):
    z = z.numpy()

# Reshape point_densities
density_grid = z.reshape(grid_size, grid_size)

# Calculate gradients

# Apply Gaussian smoothing before computing gradients
from scipy.ndimage import gaussian_filter
dy, dx = np.gradient(density_grid, edge_order=2)  # Use 8th order accurate gradient for higher precision

z_smoothed = dy.reshape(-1)
x_smoothed = x.reshape(-1)
y_smoothed = y.reshape(-1)

sns.scatterplot(x=x_smoothed, y=y_smoothed, hue=z_smoothed, linewidth=0.0, s=5, palette="coolwarm")
dx, dy
# %%
density_grid