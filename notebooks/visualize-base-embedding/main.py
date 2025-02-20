# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# %% Loading data -------------------------------------------------------------
paper_table = pd.read_csv("../../data/preprocessed/paper_table.csv")
paper_concept_table = pd.read_csv("../../data/preprocessed/paper_concept_table.csv")
concept_table = pd.read_csv("../../data/preprocessed/concept_table.csv")
data = np.load("../../data/derived/embeddings/base_paper_embeddings.npz")
concept_table = pd.read_csv("../../data/preprocessed/concept_table.csv")
#paper_ids = data["paper_ids"]
embeddings = data["embeddings"].astype(np.float32)
instruction = data["instruction"]

# %% Identify the most relevant concepts for each paper --------------------
top_concepts = (
    paper_concept_table
    .sort_values(['paper_id', 'score'], ascending=[True, False])
    .groupby('paper_id')
    .first()
    .reset_index()
)

# Merge with paper table to get paper metadata
paper_table_extended = paper_table.merge(top_concepts, on="paper_id", how="left")
paper_table_extended = pd.merge(paper_table_extended, concept_table[["id", "display_name", "level"]], left_on = "concept_id", right_on = "id", how="left")

# Filter for English titles only
paper_table_extended["is_english_title"] = paper_table_extended["title"].apply(lambda x: str(x).isascii())

# %% Umapping -----------------------------------------------------------------
# Take a random sample of embeddings for visualization
from cuml.manifold import UMAP
import cupy as cp

sample_size = 500000
sample_indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
# Fitting
model = UMAP(n_components=2, random_state=42, metric="cosine", n_neighbors=30, min_dist=0.01).fit(embeddings[sample_indices])

# %% Transforming
# Transform embeddings in batches to avoid memory issues
batch_size = 50000
n_batches = len(embeddings) // batch_size + (1 if len(embeddings) % batch_size != 0 else 0)
xy = []

import gc
from tqdm import tqdm
for i in tqdm(range(n_batches)):
    gc.collect()

    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(embeddings))
    batch = embeddings[start_idx:end_idx]
    xy_batch = model.transform(batch)
    xy.append(cp.asnumpy(xy_batch))

xy = np.vstack(xy)
xy = cp.asnumpy(xy)

# Save xy
np.savez("xy.npz", xy=xy)

# %% Visualize -----------------------------------------------------------------
xy = np.load("xy.npz")["xy"]
# Rescale xy to have zero mean and unit variance
xy = (xy - np.mean(xy, axis=0)) / np.std(xy, axis=0)

# Remove outliers using IQR method
Q1 = np.percentile(xy, 25, axis=0)
Q3 = np.percentile(xy, 75, axis=0)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Create mask for points within bounds
mask = np.all((xy >= lower_bound) & (xy <= upper_bound), axis=1)

# Filter embeddings and paper table
xy_cropped = xy[mask]
paper_table_extended_cropped = paper_table_extended[mask].reset_index(drop=True)



# Get 5 most common concepts
# Get 5 most common concepts, excluding "Download"
topk = 10
top_concepts = (paper_table_extended_cropped['display_name']
                 .value_counts()
                 .nlargest(topk)
                 .index)

# Create color column - grey for papers not in top 5 concepts
colors = paper_table_extended_cropped['display_name'].apply(lambda x: x if x in top_concepts else 'Other')

# Create DataFrame with coordinates and colors
plot_df = pd.DataFrame({
    'UMAP 1': xy_cropped[:,0],
    'UMAP 2': xy_cropped[:,1],
    'color': colors,
    'is_english_title': paper_table_extended_cropped["is_english_title"],
    "title": paper_table_extended_cropped["title"]
}).sort_values('color', ascending=False)



# %%
sns.set_style("white")
plt.style.use("dark_background")

# Improved visualization with a "starry night" aesthetic

# Define an improved color palette with a cosmic feel
palette = {concept: color for concept, color in zip(top_concepts, sns.color_palette("muted"))}
edge_color_palette = {concept: color for concept, color in zip(top_concepts, sns.color_palette())}

sns.set_context("notebook", font_scale=1.2)
fig, ax = plt.subplots(figsize=(10, 10))

# Add density plot first
sns.kdeplot(data=plot_df.sample(1000),
            x='UMAP 1', y='UMAP 2',
            fill=True,
            thresh=0,
            cmap="mako",
            bw_adjust = 0.5,
            alpha=0.8,
            levels=100,  # Increase number of levels for smoother transitions
            linewidth=0,
            ax=ax)

# Add scatter plots on top
#ax = sns.scatterplot(data=plot_df.query("color == 'Other' and is_english_title"),
#                x='UMAP 1', y='UMAP 2',
#                color='#dfdfdf',
#                s=0.01, label='Other', ax=ax, edgecolor=None, alpha=1)

# Foreground clusters with a glowing effect
for category, color in palette.items():
    sns.scatterplot(data=plot_df.query(f"color == '{category}'"),
                    x='UMAP 1', y='UMAP 2',
                    color=color,
                    alpha=1.0,
                    s=0.05,
                    ax=ax,
                    edgecolor=None,
                    label=category,
                    linewidth=0.1)

ax.set(title='UMAP of Paper Embeddings')
ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=12,       # Increase font size
    labelspacing=1.2,  # Increase spacing between labels
    markerscale=50,     # Scale up legend marker size
    handletextpad=1.5, # Space between marker and text
    borderpad=1        # Padding inside the legend box
)
ax.axis("off")
# %%
