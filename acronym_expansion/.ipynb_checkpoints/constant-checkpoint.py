import os

DATA_DIR = "../data"

### MIMIC-3 ICD-9 datasets (in JSON format) preprocessed by following KEPT GitHub 
ICD_DIR = os.path.join(DATA_DIR, "mimic3_icd")

### Saved acronym expanded sections here
GEN_DIR = os.path.join(DATA_DIR, "generated_sections")

### Saved merged train files here (original+expanded)
MERGE_DIR = os.path.join(DATA_DIR, "merged_datasets")

### Title synonyms from NeurIPS 2023 paper: "Towards Semi-Structured Automatic ICD Coding via Tree-based Contrastive Learning"
TITLE_PATH = os.path.join(DATA_DIR, "title_synonyms.json")