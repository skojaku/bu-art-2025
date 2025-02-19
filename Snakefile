from os.path import join as j

configfile: "workflow/config.yaml"

# Import utilities
include: "workflow/workflow_utils.smk"

DATA_DIR = config["data_dir"]

# GPU Configuration
GPUS = list(range(4))  # [0, 1, 2, 3]

# Input files ------------------------------------------------------------------
PREPROCESSED_DATA_DIR = j(DATA_DIR, "preprocessed")
RAW_DATA_DIR = j(DATA_DIR, "raw")

RAW_PAPER_TABLE_FILE = j(RAW_DATA_DIR, "paper_table.csv")
RAW_AUTHOR_TABLE_FILE = j(RAW_DATA_DIR, "author_table.csv")
RAW_AUTHOR_PAPER_TABLE_FILE = j(RAW_DATA_DIR, "author_paper_table.csv")
RAW_PAPER_CONCEPT_TABLE_FILE = j(RAW_DATA_DIR, "paper_concept_table.csv")
RAW_CONCEPT_TABLE_FILE = j(RAW_DATA_DIR, "concept_table.csv")
RAW_CONCEPT_TABLE_FILE = j(RAW_DATA_DIR, "concept_table.csv")

PAPER_TABLE_FILE = j(PREPROCESSED_DATA_DIR, "paper_table.csv")
AUTHOR_TABLE_FILE = j(PREPROCESSED_DATA_DIR, "author_table.csv")
AUTHOR_PAPER_TABLE_FILE = j(PREPROCESSED_DATA_DIR, "author_paper_table.csv")
PAPER_CONCEPT_TABLE_FILE = j(PREPROCESSED_DATA_DIR, "paper_concept_table.csv")
CONCEPT_TABLE_FILE = j(PREPROCESSED_DATA_DIR, "concept_table.csv")

# Intermediate files ----------------------------------------------------------
INTERMEDIATE_EMB_DIR = j(DATA_DIR, "derived", "embeddings", "intermediate")
BASE_EMB_FILE = j(DATA_DIR, "derived", "embeddings", "base_paper_embeddings.npz")

EMB_MODEL_FILE = j(DATA_DIR, "derived", "embeddings", "model.pt")
EMB_EMBEDDING_FILE = j(DATA_DIR, "derived", "embeddings", "embeddings.npz")

# Output files -----------------------------------------------------------------
PAPER_DIR = config["paper_dir"]
PAPER_SRC, SUPP_SRC = [j(PAPER_DIR, f) for f in ("main.tex", "supp.tex")]
PAPER, SUPP = [j(PAPER_DIR, f) for f in ("main.pdf", "supp.pdf")]

rule all:
    input:
        EMB_EMBEDDING_FILE
        #BASE_EMB_FILE

rule preprocess_data:
    input:
        raw_paper_table_file = RAW_PAPER_TABLE_FILE,
        raw_author_table_file = RAW_AUTHOR_TABLE_FILE,
        raw_author_paper_table_file = RAW_AUTHOR_PAPER_TABLE_FILE,
        raw_paper_concept_table_file = RAW_PAPER_CONCEPT_TABLE_FILE,
        raw_concept_table_file = RAW_CONCEPT_TABLE_FILE
    output:
        preprocessed_paper_table_file = PAPER_TABLE_FILE,
        preprocessed_author_table_file = AUTHOR_TABLE_FILE,
        preprocessed_author_paper_table_file = AUTHOR_PAPER_TABLE_FILE,
        preprocessed_paper_concept_table_file = PAPER_CONCEPT_TABLE_FILE,
        preprocessed_concept_table_file = CONCEPT_TABLE_FILE
    script:
        "workflow/preprocess_data.py"

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

rule train_model:
    input:
        paper_table_file = PAPER_TABLE_FILE,
        paper_author_table_file = AUTHOR_PAPER_TABLE_FILE,
        base_embedding_file = BASE_EMB_FILE
    output:
        model_file = EMB_MODEL_FILE
    script:
        "workflow/train_model.py"