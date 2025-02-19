import pandas as pd
import numpy as np
import os

raw_paper_table_file = snakemake.input["raw_paper_table_file"]
raw_paper_concept_table_file = snakemake.input["raw_paper_concept_table_file"]
raw_concept_table_file = snakemake.input["raw_concept_table_file"]
raw_author_paper_table_file = snakemake.input["raw_author_paper_table_file"]
raw_author_table_file = snakemake.input["raw_author_table_file"]
preprocessed_paper_table_file = snakemake.output["preprocessed_paper_table_file"]
preprocessed_paper_concept_table_file = snakemake.output["preprocessed_paper_concept_table_file"]
preprocessed_concept_table_file = snakemake.output["preprocessed_concept_table_file"]
preprocessed_author_paper_table_file = snakemake.output["preprocessed_author_paper_table_file"]
preprocessed_author_table_file = snakemake.output["preprocessed_author_table_file"]


# Load raw data
paper_table = pd.read_csv(raw_paper_table_file)
paper_concept_table = pd.read_csv(raw_paper_concept_table_file)
concept_table = pd.read_csv(raw_concept_table_file)
author_paper_table = pd.read_csv(raw_author_paper_table_file, low_memory=False)
author_table = pd.read_csv(raw_author_table_file)
# Remove papers with no title
paper_table = paper_table[~pd.isna(paper_table["title"])]

# Handle duplicate titles
title_counts = paper_table['title'].value_counts()

# Get titles that appear more than once
duplicate_titles = title_counts[title_counts > 1].index

# Split papers into those with unique titles and those with duplicates
unique_title_papers = paper_table[~paper_table['title'].isin(duplicate_titles)]
duplicate_title_papers = paper_table[paper_table['title'].isin(duplicate_titles)]

# For duplicate titles, keep the first occurrence
filtered_duplicates = duplicate_title_papers.groupby('title').first().reset_index()

# Create final paper table by combining unique titles and filtered duplicates
unique_paper_table = pd.concat([
    unique_title_papers,
    filtered_duplicates
]).sort_values('paper_id').reset_index(drop=True)

# Verify no duplicates remain
assert len(unique_paper_table['title'].unique()) == len(unique_paper_table), "Duplicate titles found in unique_paper_table"


# Create mapping from old to new paper IDs
paper_id_mapping = dict(zip(unique_paper_table['paper_id'], range(len(unique_paper_table))))

# Update paper IDs in unique_paper_table
unique_paper_table['paper_id'] = unique_paper_table['paper_id'].map(paper_id_mapping)

# Remove papers from concept table that are no longer in paper table and update paper IDs
paper_concept_table = paper_concept_table[
    paper_concept_table['paper_id'].isin(paper_id_mapping.keys())
]
paper_concept_table['paper_id'] = paper_concept_table['paper_id'].map(paper_id_mapping)

# Remove papers from author table that are no longer in paper table and update paper IDs
author_paper_table = author_paper_table[
    author_paper_table['paper_id'].isin(paper_id_mapping.keys())
]
author_paper_table['paper_id'] = author_paper_table['paper_id'].map(paper_id_mapping)

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(preprocessed_paper_table_file), exist_ok=True)

# Save cleaned data
unique_paper_table.to_csv(preprocessed_paper_table_file, index=False)
author_table.to_csv(preprocessed_author_table_file, index=False)
paper_concept_table.to_csv(preprocessed_paper_concept_table_file, index=False)
concept_table.to_csv(preprocessed_concept_table_file, index=False)
author_paper_table.to_csv(preprocessed_author_paper_table_file, index=False)

print(f"Original number of papers: {len(paper_table)}")
print(f"Number of papers after removing duplicates: {len(unique_paper_table)}")
print(f"Number of paper-concept relationships: {len(paper_concept_table)}")
print(f"Number of author-paper relationships: {len(author_paper_table)}")
