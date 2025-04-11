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

root_dir = "/data/projects/bu-art-2025/data-bu"
paper_table = pd.read_csv(f"{root_dir}/preprocessed/paper_table.csv")
paper_concept_table = pd.read_csv(f"{root_dir}/preprocessed/paper_concept_table.csv")
concept_table = pd.read_csv(f"{root_dir}/preprocessed/concept_table.csv")
#data = np.load(f"{root_dir}/derived/embeddings/embedding-checkpoint.npz")
data = np.load(f"{root_dir}/derived/embeddings/embeddings.npz")

#data = np.load("../../data/derived/embeddings/embeddings.npz")
paper_author_table = pd.read_csv(f"{root_dir}/preprocessed/author_paper_table.csv")
concept_table = pd.read_csv(f"{root_dir}/preprocessed/concept_table.csv")
paper_ids = data["paper_ids"]
embeddings_query = data["embeddings_query"].astype(np.float32)
embeddings_key = data["embeddings_key"].astype(np.float32)

embeddings_query = torch.tensor(embeddings_query)
embeddings_key = torch.tensor(embeddings_key)

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
    n_samples=100000,  # Number of random samples to use for density estimation
):
    """Compute density values on a grid in the Poincaré disk using random sampling.

    Args:
        embeddings_query: Query embeddings tensor
        embeddings_key: Key embeddings tensor
        weight_papers: Weights for each embedding point
        grid_size: Number of grid points in each dimension
        r: Radius of the grid in the Poincaré disk
        sigma: Bandwidth parameter for density estimation
        n_samples: Number of random samples to use for density estimation

    Returns:
        Tuple containing:
        - focal_points_on_poincare_disk: Grid points in Poincaré disk
        - densities: Density values for each grid point
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
        dtype=torch.float32,
    )

    # Convert points to Lorentz model
    focal_points_on_lorentz_manifold = poincare_to_lorentz(
        focal_points_on_poincare_disk
    )

    # Initialize densities
    n_grid = len(focal_points_on_lorentz_manifold)
    densities = torch.zeros(n_grid, 2, dtype=torch.float32)

    # Convert embeddings to float32 if they aren't already
    embeddings_query = embeddings_query.to(dtype=torch.float32)
    embeddings_key = embeddings_key.to(dtype=torch.float32)
    weight_papers = torch.tensor(weight_papers, dtype=torch.float32)

    # Randomly sample points for density estimation
    n_total = len(embeddings_query)
    if n_samples < n_total:
        indices = torch.randperm(n_total)[:n_samples]
        embeddings_query_sample = embeddings_query[indices]
        embeddings_key_sample = embeddings_key[indices]
        weight_papers_sample = weight_papers[indices]
        # Adjust weights to account for sampling
        weight_papers_sample = weight_papers_sample * (n_total / n_samples)
    else:
        embeddings_query_sample = embeddings_query
        embeddings_key_sample = embeddings_key
        weight_papers_sample = weight_papers
        n_samples = n_total

    # Compute in batches to avoid memory issues
    manifold = geoopt.Lorentz()
    batch_size = 100  # Process grid points in small batches

    # Process grid points in batches
    for i in tqdm(range(0, n_grid, batch_size), desc="Processing grid points"):
        batch_end = min(i + batch_size, n_grid)
        batch_points = focal_points_on_lorentz_manifold[i:batch_end, :].unsqueeze(1)

        # Compute distances for sampled points
        distances_query = manifold.dist2(batch_points, embeddings_query_sample.unsqueeze(0)).squeeze()
        distances_key = manifold.dist2(batch_points, embeddings_key_sample.unsqueeze(0)).squeeze()

        # Replace NaN values
        distances_query = torch.where(
            torch.isnan(distances_query), torch.tensor(1e10, dtype=torch.float32), distances_query
        )
        distances_key = torch.where(
            torch.isnan(distances_key), torch.tensor(1e10, dtype=torch.float32), distances_key
        )

        # Compute kernel values
        kernel_values_query = torch.exp(-distances_query / sigma)
        kernel_values_key = torch.exp(-distances_key / sigma)

        # Weight the kernel values and sum for density
        densities[i:batch_end, 0] = torch.sum(kernel_values_query * weight_papers_sample, dim=1)
        densities[i:batch_end, 1] = torch.sum(kernel_values_key * weight_papers_sample, dim=1)

        # Clear unnecessary tensors
        del distances_query, distances_key, kernel_values_query, kernel_values_key
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return focal_points_on_poincare_disk, densities

# %% Preparation --------------------------------------------------------------
author2paper, weight_paper = create_author_paper_matrix(paper_table, paper_author_table)

# %% Compute density --------------------------------------------------------------

focal_points_on_poincare_disk, densities = compute_density_grid(embeddings_query, embeddings_key, weight_paper)

# %% Count the number of authors who travel between different focal points on the Poincaré disk

def get_paper_sequences(
    author2paper: sparse.csr_matrix,
    paper_years: np.ndarray,
    n_authors_sample: int = 100000
):
    """Get chronologically ordered paper sequences for sampled authors.

    Args:
        author2paper: Sparse matrix mapping authors to papers
        paper_years: Array of paper years
        n_authors_sample: Number of authors to sample

    Returns:
        Tuple containing:
        - sequences: List of paper sequences
        - weights: List of weights for each sequence
    """
    sequences = []
    weights = []

    # Sample authors who have published more than one paper
    author_paper_counts = np.diff(author2paper.indptr)
    eligible_authors = np.where(author_paper_counts > 1)[0]

    if len(eligible_authors) > n_authors_sample:
        sampled_authors = np.random.choice(eligible_authors, size=n_authors_sample, replace=False)
    else:
        sampled_authors = eligible_authors

    # Process sampled authors
    for author_id in tqdm(sampled_authors, desc="Processing authors"):
        # Get papers for this author
        start_idx = author2paper.indptr[author_id]
        end_idx = author2paper.indptr[author_id + 1]
        papers = author2paper.indices[start_idx:end_idx]

        # Create year-paper dictionary
        year_papers = {}
        for pid in papers:
            year = paper_years[pid]
            if year not in year_papers:
                year_papers[year] = []
            year_papers[year].append(pid)

        # Create sequence with one random paper per year
        years = sorted(year_papers.keys())
        if len(years) > 1:  # Only include if there are papers from multiple years
            sequence = [np.random.choice(year_papers[year]) for year in years]
            sequences.append(sequence)
            # Weight proportional to number of eligible authors
            weights.append(len(eligible_authors) / len(sampled_authors))

    return sequences, weights

sequences, weights = get_paper_sequences(author2paper, paper_table["year"].values)

# %% Classify the papers into different focal points

# Compute the Minkowski inner dot product
focal_points_on_lorentz_manifold = poincare_to_lorentz(focal_points_on_poincare_disk)

# Convert to same dtype (float32) before operations
focal_points_on_lorentz_manifold = focal_points_on_lorentz_manifold.to(torch.float32)
Dsrc = embeddings_query[:, 1:] @ focal_points_on_lorentz_manifold[:, 1:].T
Dsrc = Dsrc - embeddings_query[:, 0].reshape((-1,1)) * focal_points_on_lorentz_manifold[:, 0].reshape((1,-1))
Dsrc = torch.abs(Dsrc)

Dtrg = embeddings_key[:, 1:] @ focal_points_on_lorentz_manifold[:, 1:].T
Dtrg = Dtrg - embeddings_key[:, 0].reshape((-1,1)) * focal_points_on_lorentz_manifold[:, 0].reshape((1,-1))
Dtrg = torch.abs(Dtrg)
closest_focal_points_src = torch.argmin(Dsrc, dim = 1)
closest_focal_points_trg = torch.argmin(Dtrg, dim = 1)
# %% Count the flux between different focal points

pair_list = []
for i in range(len(sequences)):
    for j in range(len(sequences[i]) - 1):
        src = sequences[i][j]
        trg = sequences[i][j+1]
        src = closest_focal_points_src[src]
        trg = closest_focal_points_trg[trg]
        pair_list.append((src, trg))

# %%
manifold = geoopt.Lorentz()
flux = pd.DataFrame(torch.tensor(pair_list), columns = ["src", "trg"])
uniq_pairs = flux.groupby(["src", "trg"]).size().reset_index(name="flux")
uniq_pairs["distance"] = uniq_pairs.apply(lambda row: manifold.dist(focal_points_on_lorentz_manifold[row["src"]], focal_points_on_lorentz_manifold[row["trg"]]).item(), axis=1)
# %%
Win = np.bincount(closest_focal_points_src, weights = torch.tensor(weight_paper))
Wout = np.bincount(closest_focal_points_trg, weights = torch.tensor(weight_paper))
uniq_pairs["flux_exp"] = Win[uniq_pairs["src"]] * Wout[uniq_pairs["trg"]] * np.exp(-uniq_pairs["distance"])
uniq_pairs = uniq_pairs.dropna()
# %%
flux = pd.merge(flux, uniq_pairs, on = ["src", "trg"], how = "left")
flux = flux.dropna()
# %%
# %%
#ax = sns.scatterplot(data = flux, x = "flux_exp", y = "flux", hue = "flux")
#ax.set_xscale("log")
#ax.set_yscale("log")

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot 1: Distance vs Flux
hb1 = ax[0].hexbin(
    flux["distance"],
    np.log(flux["flux"]+1),
    gridsize=30,
    bins='log',  # Use logarithmic binning for the colors
    cmap='YlOrRd'  # Yellow-Orange-Red colormap
)
plt.colorbar(hb1, ax=ax[0], label='Count (log)')
ax[0].set_xlabel('Distance between focal points')
ax[0].set_ylabel('Flux (number of transitions)')
ax[0].set_title('Distance vs Flux between Focal Points')
# Plot 2: Expected vs Observed Flux - Only showing flows above median
# Filter data to only show flows above median
median_flux = flux["flux"].min()
high_flux = flux[flux["flux"] > median_flux]

eps = 1e-10
hb2 = ax[1].hexbin(
    np.log(high_flux["flux_exp"]+eps),
    np.log(high_flux["flux"]+eps),
    gridsize=50,
    bins='log',  # Use logarithmic binning for the colors
    cmap='YlOrRd'  # Yellow-Orange-Red colormap
)
plt.colorbar(hb2, ax=ax[1], label='Count (log)')
ax[1].set_xlabel('Flux Expected (log)')
ax[1].set_ylabel('Flux Observed (log)')
ax[1].set_title('Expected vs Observed Flux (Above Median)')

import scipy
# Add regression line to the second plot
x = np.log(high_flux["flux_exp"]+eps)
y = np.log(high_flux["flux"]+eps)
mask = ~np.isnan(x) & ~np.isnan(y)
x_clean = x[mask]
y_clean = y[mask]

# Calculate regression line
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_clean, y_clean)
r_squared = r_value

# Plot regression line
x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
y_line = slope * x_line + intercept
ax[1].plot(x_line, y_line, 'b-', linewidth=2)

# Add R² text to the plot
ax[1].text(0.05, 0.95, f'R² = {r_squared:.3f}',
           transform=ax[1].transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax[1].set_xlim(-20, -3)
ax[1].set_ylim(0, 13)
sns.despine(fig)
# Adjust layout and save
fig.tight_layout()
#fig.savefig("flux-between-focal-points.png", dpi=300, bbox_inches='tight')
#plt.close(fig)  # Close the figure to free memory

# %%
