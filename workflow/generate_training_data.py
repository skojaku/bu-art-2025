"""
Lorentz embedding implementation using PyTorch Lightning.
"""

import torch
import geoopt
import polars as pl
import numpy as np
import pytorch_lightning as pl
from scipy import sparse
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Tuple, List
import os


@dataclass
class Config:
    """Configuration parameters"""
    embedding_dim: int = 2  # Embedding dimension (not including time dimension)
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    min_text_len: int = 10
    n_test_samples: int = 5000
    end_train_year: int = 2023


class TrajectoryDataset(Dataset):
    """Dataset for paper trajectories."""

    def __init__(self, sequences: List[List[int]], weights: List[float], n_papers: int):
        self.sequences = sequences
        self.weights = weights
        self.n_papers = n_papers

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

    def __len__(self):
        return len(self.src_indices)

    def __getitem__(self, idx):
        return {
            'src': self.src_indices[idx],
            'trg': self.trg_indices[idx],
            'neg': self.neg_indices[idx],
            'weight': self.pair_weights[idx]
        }


class LorentzModel(pl.LightningModule):
    """Lorentz embedding model using PyTorch Lightning."""

    def __init__(self, num_papers: int, config: Config, embedding_file: str, model_file: str):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.num_papers = num_papers
        self.embedding_file = embedding_file
        self.model_file = model_file

        # Initialize manifold and embeddings
        self.manifold = geoopt.Lorentz()
        self.embeddings = geoopt.ManifoldParameter(
            torch.empty(num_papers, config.embedding_dim + 1),
            manifold=self.manifold
        )

        with torch.no_grad():
            # Initialize in tangent space
            self.embeddings.uniform_(-0.001, 0.001)
            # Project onto the manifold
            self.embeddings.proj_()

    def forward(self, x):
        return self.embeddings[x]

    def training_step(self, batch, batch_idx):
        # Get embeddings
        src_emb = self(batch['src'])
        trg_emb = self(batch['trg'])
        neg_emb = self(batch['neg'])

        # Compute distances
        pos_dist = self.manifold.dist(src_emb, trg_emb)
        neg_dist = self.manifold.dist(src_emb, neg_emb)

        # Compute loss
        margin = 1.0
        loss = torch.mean(batch['weight'] * torch.relu(margin + pos_dist - neg_dist))

        # Log loss
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return geoopt.optim.RiemannianAdam(
            self.parameters(),
            lr=self.config.learning_rate,
            stabilize=10
        )

    def on_train_end(self):
        # Save embeddings at the end of training
        embeddings = self.embeddings.detach().cpu().numpy()
        np.savez(
            self.embedding_file,
            embeddings=embeddings,
            config=np.array([
                self.config.embedding_dim,
                self.config.batch_size,
                self.config.epochs,
                self.config.learning_rate
            ])
        )

        self.save_model()

    def save_model(self):
        """Save the model state dict."""
        torch.save(self.state_dict(), self.hparams.model_file)


def get_paper_sequences(author2paper: sparse.csr_matrix, paper_years: np.ndarray) -> Tuple[List[List[int]], List[float]]:
    """Get chronologically ordered paper sequences for each author."""
    sequences = []
    weights = []

    for author_id in range(author2paper.shape[0]):
        # Get papers for this author
        papers = author2paper.indices[
            author2paper.indptr[author_id]:author2paper.indptr[author_id + 1]
        ]

        if len(papers) > 1:
            # Sort papers by year
            paper_data = [(pid, paper_years[pid]) for pid in papers]
            paper_data.sort(key=lambda x: x[1])  # Sort by year

            # Extract just the paper IDs in chronological order
            sequence = [x[0] for x in paper_data]
            sequences.append(sequence)
            weights.append(1.0)  # Equal weight for all sequences

    return sequences, weights


# Configuration
config = Config()

# Load data
print("Loading data...")
paper_table = pl.read_csv(snakemake.input["paper_table"])
paper_author_table = pl.read_csv(snakemake.input["paper_author_table"])
model_file = snakemake.output["model_file"]
embedding_file = snakemake.output["embedding_file"]

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
            paper_author_table["paper_id"].to_numpy()
        )
    ),
    shape=(n_authors, n_papers)
)

# Get sequences
print("Generating sequences...")
sequences, weights = get_paper_sequences(author2paper, years)

# Create dataset and dataloader
print("Creating dataset...")
dataset = TrajectoryDataset(sequences, weights, n_papers)
dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4
)

# Initialize model and trainer
print("Setting up model and trainer...")
model = LorentzModel(n_papers, config, embedding_file, model_file)

trainer = pl.Trainer(
    max_epochs=config.epochs,
    accelerator='auto',  # Automatically detect GPU/CPU
    devices=1,
    enable_progress_bar=True,
    logger=pl.loggers.TensorBoardLogger('lightning_logs/'),
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints/',
            filename='lorentz-{epoch:02d}-{train_loss:.2f}',
            save_top_k=3,
            monitor='train_loss'
        ),
        pl.callbacks.EarlyStopping(
            monitor='train_loss',
            patience=10,
            mode='min'
        )
    ]
)

# Train model
print("Starting training...")
trainer.fit(model, dataloader)

print(f"Done! Embeddings saved to {embedding_file}")
print(f"Training logs saved in {trainer.logger.log_dir}")