import pandas as pd
import numpy as np
from InstructorEmbedding import INSTRUCTOR
import torch
from snakemake.shell import shell

def generate_embeddings(input_file, output_file, instruction, batch_size=512, device=None):
    """
    Generate embeddings for paper titles using the Instructor model.

    Args:
        input_file (str): Path to the input CSV file (from snakemake)
        output_file (str): Path to save the output NPZ file (from snakemake)
        instruction (str): Instruction for the embedding model
        batch_size (int): Batch size for embedding generation
        device (str): Device to use for computation ('cuda' or 'cpu')
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load paper data
    print(f"Loading paper data from {input_file}")
    paper_table = pd.read_csv(input_file)

    # Initialize the model
    print(f"Initializing INSTRUCTOR model on {device}")
    encoder_model = INSTRUCTOR("hkunlp/instructor-large", device=device)

    # Prepare titles
    titles = paper_table["title"].values
    is_missing_title = pd.isna(titles)
    titles[is_missing_title] = "None"

    # Prepare input data
    input_data = [[instruction, title] for title in titles]

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = encoder_model.encode(
        input_data,
        batch_size=batch_size,
        show_progress_bar=True,
        device=device,
        normalize_embeddings=True
    )

    # Set embeddings for missing titles to 0
    embeddings[is_missing_title, :] = 0

    # Save embeddings
    print(f"Saving embeddings to {output_file}")
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        titles=titles,
        instruction=instruction
    )

    print("Done!")
    return embeddings

# Get input and output files from snakemake
input_file = snakemake.input["input_file"]
output_file = snakemake.output["output_file"]

# Get parameters from snakemake config if available, otherwise use defaults
instruction = snakemake.params["instruction"]
batch_size = snakemake.params["batch_size"]
device = snakemake.params["device"]

# Generate embeddings
generate_embeddings(
    input_file=input_file,
    output_file=output_file,
    instruction=instruction,
    batch_size=batch_size,
    device=device
)