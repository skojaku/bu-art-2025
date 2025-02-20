# %%
from tqdm import tqdm
import torch
import polars
import numpy as np
import pytorch_lightning as pl
from scipy import sparse
from torch.utils.data import Dataset, DataLoader, random_split
from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import sys

# Add the parent directory to Python path so we can import the workflow module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "snakemake" in sys.modules:
    from workflow.model import LorentzModel, Config
else:
    from model import LorentzModel, Config


# Load data
print("Loading data...")
if "snakemake" in sys.modules:
    paper_table = polars.read_csv(snakemake.input["paper_table_file"])
    base_embeddings = np.load(snakemake.input["base_embedding_file"])["embeddings"]
    model_file = snakemake.input["model_file"]
    output_file = snakemake.output["output_file"]
else:
    paper_table = polars.read_csv("../data/preprocessed/paper_table.csv")
    base_embeddings = np.load("../data/derived/embeddings/base_paper_embeddings.npz")[
        "embeddings"
    ]
    # model_file = "../data/derived/embeddings/model.pt"
    model_file = "../checkpoints/lorentz-epoch=58-val_loss=-4.87.ckpt"
    output_file = "../data/derived/embeddings/embedding-checkpoint.npz"

# Initialize model
print("Loading model...")
config = Config()
model = LorentzModel(base_embeddings.shape[1], config, model_file)

# Load trained weights with a map_location to handle potential device mismatches
checkpoint = torch.load(model_file, map_location="cpu")
if "state_dict" in checkpoint:  # Handle Lightning checkpoint format
    model.load_state_dict(checkpoint["state_dict"])
else:  # Handle direct state dict format
    model.load_state_dict(checkpoint)
model.eval()

# Generate embeddings
print("Generating embeddings...")
with torch.no_grad():
    embeddings_query = model.forward_query(torch.tensor(base_embeddings)).numpy()
    embeddings_key = model.forward_key(torch.tensor(base_embeddings)).numpy()

# %% Save embeddings
print("Saving embeddings...")
np.savez(
    output_file,
    embeddings_query=embeddings_query,
    embeddings_key=embeddings_key,
    paper_ids=paper_table["paper_id"].to_numpy(),
    instruction="Lorentz embedding",
)

# %%
