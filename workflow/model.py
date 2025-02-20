# %%
"""
Lorentz embedding implementation using PyTorch Lightning.
"""
from tqdm import tqdm
import torch
import geoopt
import pytorch_lightning as pl
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration parameters"""

    embedding_dim: int = 2  # Embedding dimension (not including time dimension)
    batch_size: int = 1024
    epochs: int = 60
    learning_rate: float = 1e-4
    validation_split: float = 0.05
    dropout_rate: float = 0.2  # 0 means no dropout
    margin: float = 1.0  # Hyperparameter for margin


class RiemannianLayerNorm(torch.nn.Module):
    def __init__(self, manifold, embedding_dim):
        super().__init__()
        self.manifold = manifold  # Lorentz manifold
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # Move to tangent space at (1,0,...,0)
        tangent_x = self.manifold.logmap0(x)

        # Apply Euclidean LayerNorm in tangent space
        tangent_x = torch.nn.functional.layer_norm(tangent_x, (self.embedding_dim,))

        # Move back to Lorentz hyperboloid
        return self.manifold.expmap0(tangent_x)


class LorentzModel(pl.LightningModule):
    """Lorentz embedding model using PyTorch Lightning."""

    def __init__(self, input_dim: int, config: Config, model_file: str):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.input_dim = input_dim
        self.model_file = model_file
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Initialize manifold
        self.manifold = geoopt.Lorentz()

        # Create a deeper network with non-linear activations
        self.projection_query = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(config.dropout_rate),
            torch.nn.Linear(input_dim // 2, input_dim // 4),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(config.dropout_rate),
            torch.nn.Linear(input_dim // 4, config.embedding_dim + 1),
        )

        self.projection_key = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(config.dropout_rate),
            torch.nn.Linear(input_dim // 2, input_dim // 4),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(config.dropout_rate),
            torch.nn.Linear(input_dim // 4, config.embedding_dim + 1),
        )

        # Initialize weights using Xavier initialization with small values
        for layer in self.projection_query:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(
                    layer.weight, gain=1.0
                    #layer.weight, gain=0.3
                )  # Reduced gain for smaller weights
                torch.nn.init.zeros_(layer.bias)

        for layer in self.projection_key:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(
                    layer.weight, gain=1.0
                )  # Reduced gain for smaller weights
                torch.nn.init.zeros_(layer.bias)

        # Add this line to track best loss
        self.best_loss = float("inf")

    def forward(self, x):
        return self.forward_query(x)

    def forward_query(self, x):
        projected = self.projection_query(x)
        return self.manifold.projx(projected)

    def forward_key(self, x):
        projected = self.projection_key(x)
        return self.manifold.projx(projected)

    def compute_loss(self, batch):
        # Get embeddings
        src_emb = self.forward_query(batch["src"])
        trg_emb = self.forward_key(batch["trg"])
        neg_emb = self.forward_key(batch["neg"])

        # Compute distances
        pos_dist = self.manifold.dist(src_emb, trg_emb)
        neg_dist = self.manifold.dist(src_emb, neg_emb)

        # Compute binary cross entropy loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            -pos_dist, torch.ones_like(pos_dist), weight=batch["weight"], reduction="sum"
        )
        loss += torch.nn.functional.binary_cross_entropy_with_logits(
            -neg_dist,
            torch.zeros_like(neg_dist),
            weight=batch["weight"],
            reduction="sum",
        )
        loss /= 2 * len(pos_dist)

        # Triplet loss
        # triplet_loss = torch.clamp(pos_dist - neg_dist + self.config.margin, min=0.0)
        # loss = torch.mean(batch['weight'] * triplet_loss)

        # Weight and combine the losses
        # loss = torch.mean(batch['weight'] * (pos_loss + neg_loss))

        # Compute loss
        # Compute triplet loss with margin
        # Contrastive loss formulation
        # loss_pos = pos_dist
        # loss_neg = torch.log(torch.exp(self.margin - neg_dist) + 1)
        # loss_neg = torch.nn.functional.relu(self.config.margin - neg_dist)
        # triplet_loss = torch.clamp(pos_dist - neg_dist + self.config.margin, min=0.0)
        # loss = torch.mean(batch['weight'] * triplet_loss)
        # loss = torch.mean(batch['weight'] * (loss_pos + loss_neg))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # Apply gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

        # Enhanced logging with sync_dist=True
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # Track best loss
        if loss < self.best_loss:
            self.best_loss = loss
            self.log("best_loss", self.best_loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = geoopt.optim.RiemannianAdam(
            self.parameters(),
            lr=self.config.learning_rate,  # Try reducing to 1e-4
            stabilize=10,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def on_train_end(self):
        self.save_model()

    def save_model(self):
        """Save the model state dict."""
        torch.save(self.state_dict(), self.hparams.model_file)


def test_embedding_initialization(model, base_embeddings):
    """Test if embeddings are initialized near the origin in Lorentz space."""
    # Get all embeddings
    with torch.no_grad():
        # Pass base embeddings through model
        test_embeddings = model(base_embeddings)

    # Calculate distance from origin in Lorentz space
    # Origin in Lorentz space is (1,0,0,...,0)
    origin = torch.zeros_like(test_embeddings)
    origin[:, 0] = 1.0

    distances = model.manifold.dist(test_embeddings, origin)

    mean_dist = distances.mean().item()
    max_dist = distances.max().item()

    print(f"Mean distance from origin: {mean_dist:.2f}")
    print(f"Max distance from origin: {max_dist:.2f}")

    return mean_dist, max_dist
