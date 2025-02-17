import pandas as pd
import numpy as np
from InstructorEmbedding.instructor import INSTRUCTOR
import torch
from snakemake.shell import shell

def generate_embeddings(input_file, output_file, instruction, batch_size=512, device=None):
    """
    Generate embeddings for a subset of paper titles using the Instructor model.

    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output NPZ file
        instruction (str): Instruction for the embedding model
        batch_size (int): Batch size for embedding generation
        device (str): Device to use for computation (e.g., 'cuda:0', 'cuda:1', etc.)
    """
    # Extract GPU ID from device string
    gpu_id = int(device.split(':')[1]) if 'cuda' in device else None

    # Load paper data
    print(f"Loading paper data from {input_file}")
    paper_table = pd.read_csv(input_file)

    # Filter papers for this GPU based on paper_id modulus
    if gpu_id is not None:
        n_gpus = len(snakemake.config.get('GPUS', [0, 1, 2, 3]))
        paper_table = paper_table[paper_table.index % n_gpus == gpu_id].copy()
        print(f"Processing {len(paper_table)} papers on GPU {gpu_id}")

    # Initialize the model
    print(f"Initializing INSTRUCTOR model on {device}")
    model = INSTRUCTOR('hkunlp/instructor-large', device=device)

    # Prepare titles
    titles = paper_table["title"].values
    paper_ids = paper_table.index.values
    is_missing_title = pd.isna(titles)
    titles[is_missing_title] = "None"

    # Prepare input data
    input_data = [[instruction, title] for title in titles]

    # Generate embeddings
    print(f"Generating embeddings on {device}...")
    embeddings = model.encode(
        input_data,
        batch_size=batch_size,
        show_progress_bar=True,
        device=device,
        normalize_embeddings=True
    )

    # Set embeddings for missing titles to 0
    embeddings[is_missing_title, :] = 0

    # Save embeddings with paper IDs
    print(f"Saving embeddings to {output_file}")
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        paper_ids=paper_ids,
        instruction=instruction
    )

    print("Done!")
    return embeddings

# Get input and output files from snakemake
input_file = snakemake.input["input_file"]
output_file = snakemake.output["output_file"]

# Get parameters from snakemake config
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