# %%
"""
Lorentz embedding implementation using PyTorch Lightning.
"""
from tqdm import tqdm
import torch
import geoopt
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
    from odel import LorentzModel, Config


class TrajectoryDataset(Dataset):
    """Dataset for paper trajectories."""

    def __init__(
        self,
        sequences: List[List[int]],
        weights: List[float],
        base_embeddings: np.ndarray,
    ):
        self.sequences = sequences
        self.weights = weights
        self.base_embeddings = base_embeddings

        # Prepare pairs
        self.src_indices = []
        self.trg_indices = []
        self.neg_indices = []
        self.pair_weights = []

        for seq, weight in zip(sequences, weights):
            if len(seq) < 2:
                continue

            # Create positive pairs from consecutive papers
            for i in range(len(seq) - 1):
                self.src_indices.append(seq[i])
                self.trg_indices.append(seq[i + 1])

                # Random negative sampling
                neg_idx = np.random.randint(0, len(sequences))
                neg_seq = sequences[neg_idx]
                self.neg_indices.append(np.random.choice(neg_seq))

                self.pair_weights.append(weight)

        # Convert to tensors
        self.src_indices = torch.tensor(self.src_indices)
        self.trg_indices = torch.tensor(self.trg_indices)
        self.neg_indices = torch.tensor(self.neg_indices)
        self.pair_weights = torch.tensor(self.pair_weights)
        self.base_embeddings = torch.tensor(self.base_embeddings)

    def __len__(self):
        return len(self.src_indices)

    def __getitem__(self, idx):
        return {
            "src": self.base_embeddings[self.src_indices[idx]],
            "trg": self.base_embeddings[self.trg_indices[idx]],
            "neg": self.base_embeddings[self.neg_indices[idx]],
            "weight": self.pair_weights[idx],
        }


def get_paper_sequences(
    author2paper: sparse.csr_matrix, paper_years: np.ndarray, n_epochs: int
) -> Tuple[List[List[int]], List[float]]:
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


# Configuration
config = Config()

# Load data
print("Loading data...")
if "snakemake" in sys.modules:
    paper_table = polars.read_csv(snakemake.input["paper_table_file"])
    paper_author_table = polars.read_csv(snakemake.input["paper_author_table_file"])
    base_embeddings = np.load(snakemake.input["base_embedding_file"])["embeddings"]
    model_file = snakemake.output["model_file"]
else:
    paper_table = polars.read_csv("../data/preprocessed/paper_table.csv")
    paper_author_table = polars.read_csv("../data/preprocessed/author_paper_table.csv")
    base_embeddings = np.load("../data/derived/embeddings/base_paper_embeddings.npz")[
        "embeddings"
    ]
    model_file = "checkpoints/lorentz-00-0.00.pt"

# Get years and prepare author-paper matrix
years = paper_table["year"].to_numpy()
n_papers = len(paper_table)
n_authors = int(paper_author_table["author_id"].max() + 1)

# Create author-paper matrix
print("Creating author-paper matrix...")
author2paper = sparse.csr_matrix(
    (
        np.ones(len(paper_author_table)),
        (
            paper_author_table["author_id"].to_numpy(),
            paper_author_table["paper_id"].to_numpy(),
        ),
    ),
    shape=(n_authors, n_papers),
)

# Get sequences
print("Generating sequences...")
n_epochs = 10
sequences, weights = get_paper_sequences(author2paper, years, n_epochs=n_epochs)

# %%
# Create dataset and split into train/val
print("Creating dataset...")
full_dataset = TrajectoryDataset(sequences, weights, base_embeddings)
train_size = int((1 - config.validation_split) * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
)

val_loader = DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
)

# Initialize model and trainer
print("Setting up model and trainer...")
model = LorentzModel(base_embeddings.shape[1], config, model_file)

# Load checkpoint if it exists
if os.path.exists(model_file):
    print(f"Loading checkpoint from {model_file}")
    model.load_state_dict(torch.load(model_file))

# %%
trainer = pl.Trainer(
    max_epochs=config.epochs,
    accelerator="auto",  # Automatically detect GPU/CPU
    devices=1,
    enable_progress_bar=True,
    logger=pl.loggers.TensorBoardLogger("lightning_logs/"),
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath="checkpoints/",
            filename="lorentz-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
        ),
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        # Add this new callback
        # pl.callbacks.RichProgressBar()
    ],
)

# Train model
print("Starting training...")
trainer.fit(model, train_loader, val_loader)

print(f"Training logs saved in {trainer.logger.log_dir}")
# %%
