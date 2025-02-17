import numpy as np

def combine_partial_embeddings(input_files, output_file):
    """
    Combine partial embeddings from multiple GPUs into a single file.

    Args:
        input_files (list): List of paths to partial embedding files
        output_file (str): Path to save the combined embeddings
    """
    # Lists to store all data
    all_embeddings = []
    all_paper_ids = []
    instruction = None

    # Load and combine all partial embeddings
    for file in input_files:
        data = np.load(file)
        all_embeddings.append(data['embeddings'])
        all_paper_ids.append(data['paper_ids'])

        # Get instruction (should be the same for all files)
        if instruction is None:
            instruction = str(data['instruction'])

    # Combine arrays
    paper_ids = np.concatenate(all_paper_ids)
    embeddings = np.concatenate(all_embeddings)

    # Sort by paper_ids
    sort_idx = np.argsort(paper_ids)
    paper_ids = paper_ids[sort_idx]
    embeddings = embeddings[sort_idx]

    # Save combined data
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        paper_ids=paper_ids,
        instruction=instruction
    )

# Get input and output files from snakemake
input_files = snakemake.input.partial_embeddings
output_file = snakemake.output.output_file

# Combine embeddings
combine_partial_embeddings(input_files, output_file)