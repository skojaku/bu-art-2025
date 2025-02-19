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
    base_embeddings = np.load("../data/derived/embeddings/base_paper_embeddings.npz")["embeddings"]
    model_file = "../data/derived/embeddings/model.pt"
    output_file = "../data/derived/embeddings/embeddings.npz"

# Initialize model
print("Loading model...")
config = Config()
model = LorentzModel(base_embeddings.shape[1], config, model_file)

# Load trained weights
model.load_state_dict(torch.load(model_file))
model.eval()

# Generate embeddings
print("Generating embeddings...")
with torch.no_grad():
    embeddings = model(torch.tensor(base_embeddings)).numpy()

# %% Save embeddings
print("Saving embeddings...")
np.savez(
    output_file,
    embeddings=embeddings,
    paper_ids=paper_table["paper_id"].to_numpy(),
    instruction="Lorentz embedding"
)

# %%
