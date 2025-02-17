from os.path import join as j

configfile: "workflow/config.yaml"

# Import utilities
include: "workflow/workflow_utils.smk"

DATA_DIR = config["data_dir"]

# GPU Configuration
GPUS = list(range(4))  # [0, 1, 2, 3]

# Input files ------------------------------------------------------------------
PREPROCESSED_DATA_DIR = j(DATA_DIR, "preprocessed")
PAPER_TABLE_FILE = j(PREPROCESSED_DATA_DIR, "paper_table.csv")
AUTHOR_TABLE_FILE = j(PREPROCESSED_DATA_DIR, "author_table.csv")
AUTHOR_PAPER_TABLE_FILE = j(PREPROCESSED_DATA_DIR, "author_paper_table.csv")
PAPER_CONCEPT_TABLE_FILE = j(PREPROCESSED_DATA_DIR, "paper_concept_table.csv")

# Intermediate files ----------------------------------------------------------
INTERMEDIATE_EMB_DIR = j(DATA_DIR, "derived", "embeddings", "intermediate")
BASE_EMB_FILE = j(DATA_DIR, "derived", "embeddings", "base_paper_embeddings.npz")

# Output files -----------------------------------------------------------------
PAPER_DIR = config["paper_dir"]
PAPER_SRC, SUPP_SRC = [j(PAPER_DIR, f) for f in ("main.tex", "supp.tex")]
PAPER, SUPP = [j(PAPER_DIR, f) for f in ("main.pdf", "supp.pdf")]

rule all:
    input:
        BASE_EMB_FILE

rule generate_partial_embeddings:
    input:
        input_file = PAPER_TABLE_FILE
    output:
        output_file = temp(j(INTERMEDIATE_EMB_DIR, "partial_embeddings_{gpu_id}.npz"))
    params:
        instruction = "Represent the Science title.",
        batch_size = 512,
        device = lambda wildcards: f"cuda:{wildcards.gpu_id}"
    script:
        "workflow/instructor-embedding.py"

rule combine_embeddings:
    input:
        partial_embeddings = expand(j(INTERMEDIATE_EMB_DIR, "partial_embeddings_{gpu_id}.npz"), gpu_id=GPUS)
    output:
        output_file = protected(BASE_EMB_FILE)
    script:
        "workflow/combine-embeddings.py"
