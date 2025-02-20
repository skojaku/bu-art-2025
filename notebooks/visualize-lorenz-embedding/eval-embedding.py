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
data = np.load("../../data/derived/embeddings/embeddings-2025-0220.npz")
#data = np.load("../../data/derived/embeddings/embeddings.npz")
paper_author_table = pd.read_csv("../../data/preprocessed/author_paper_table.csv")
concept_table = pd.read_csv("../../data/preprocessed/concept_table.csv")
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

# %% Preparation --------------------------------------------------------------
author2paper, weight_paper = create_author_paper_matrix(paper_table, paper_author_table)

# %% Compute density --------------------------------------------------------------

focal_points_on_poincare_disk, densities = compute_density_grid(embeddings_query, embeddings_key, weight_paper)

# %% Count the number of authors who travel between different focal points on the Poincaré disk

def get_paper_sequences(
    author2paper: sparse.csr_matrix, paper_years: np.ndarray, n_epochs: int
):
    """Get chronologically ordered paper sequences for each author."""
    sequences = []
    weights = []
    for epoch in tqdm(range(n_epochs)):
        for author_id in range(author2paper.shape[0]):
            # Get papers for this author
            papers = author2paper.indices[
                author2paper.indptr[author_id] : author2paper.indptr[author_id + 1]
            ]

            if len(papers) > 1:
                # Sort papers by year
                paper_data = [(pid, paper_years[pid]) for pid in papers]

                # Group papers by year
                year_groups = {}
                for pid, year in paper_data:
                    if year not in year_groups:
                        year_groups[year] = []
                    year_groups[year].append(pid)

                # Create sequence with one random paper per year
                sequence = []
                for year in sorted(year_groups.keys()):
                    sequence.append(np.random.choice(year_groups[year]))

                sequences.append(sequence)
                weights.append(1.0)

    return sequences, weights

sequences, weights = get_paper_sequences(author2paper, paper_table["year"].values, 10)

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
# Create hexbin plot of distance vs flux
plt.hexbin(flux["distance"], np.log(flux["flux"]+1),
           gridsize=30,
           bins='log',  # Use logarithmic binning for the colors
           cmap='YlOrRd')  # Yellow-Orange-Red colormap
plt.colorbar(label='Count (log)')
plt.xlabel('Distance between focal points')
plt.ylabel('Flux (number of transitions)')
#plt.yscale('log')  # Keep log scale on y-axis
plt.title('Hexbin plot of Distance vs Flux between Focal Points')


# %%

plt.hexbin(np.log(flux["flux_exp"]+1), np.log(flux["flux"]+1),
           gridsize=50,
           bins='log',  # Use logarithmic binning for the colors
           cmap='YlOrRd')  # Yellow-Orange-Red colormap
plt.colorbar(label='Count (log)')
plt.xlabel('Flux Expected')
plt.ylabel('Flux Observed')
#plt.yscale('log')  # Keep log scale on y-axis
plt.title('Hexbin plot of Flux Expected vs Flux Observed')


# %%
