# NLP Project — Problem Statement 4

## BERT Word Embeddings for Cross-Project Story Point Estimation with Temporal Ordering

### Problem Statement

> Evaluate BERT-based word embeddings for predicting story points of later issues in a target project using source-project training data and chronological ordering.

| Dimension | Configuration |
|---|---|
| **Text Representation** | Word embeddings (BERT) |
| **Project Setting** | Cross-project (39 source→target pairs) |
| **Data Splitting** | Temporal ordering (chronological) |
| **Evaluation Metrics** | MAE, MdAE |

### Dataset

**TAWOS** (Technical and Agile Work-item Open Source) — A dataset of software issues from Jira repositories with story point annotations.

- Source: https://doi.org/10.5522/04/21308124
- ~500K issues across 44 projects
- Key fields: Title, Description, Story Points, Creation Date, Project

### Project Structure

```
NLP project/
├── story_point_estimation.ipynb   # Main notebook (all experiments)
├── config.yaml                     # Hyperparameters, seeds, model names
├── requirements.txt                # Python dependencies
├── TAWOS.sql                       # Raw dataset (MySQL dump)
├── src/
│   └── extract_data.py             # SQL → CSV extraction script
├── data/
│   ├── raw/                        # Raw data files
│   └── processed/                  # Cleaned CSVs, ready for modeling
│       ├── issues.csv
│       └── projects.csv
└── results/
    ├── embeddings/                  # Cached BERT/SBERT embeddings
    ├── tables/                      # Result CSVs
    └── plots/                       # EDA and comparison figures
```

### Setup & Reproduction

#### Option A: Local Workstation (macOS / Linux)

1. **Clone this repository**
   ```bash
   git clone https://github.com/rithvik-999/nlp-story-point-estimation.git
   cd nlp-story-point-estimation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pre-executed notebook**
   ```bash
   jupyter notebook code.ipynb
   # or story_point_estimation.ipynb
   ```
   - The notebook is 100% pre-executed with all cell outputs (tables, metrics, plots) visible for instant examination and viva presentation.
   - The complete final report is available in [report.tex](report.tex).

---

#### Option B: Kaggle & Google Colab Cloud Reproduction

When executing in cloud environments (e.g. Kaggle Notebooks with GPU accelerator), prepend the following bulletproof cell at the top of your notebook or execute `python kaggle_bootstrap.py`.

This completely eliminates the three common Kaggle runtime failure modes:
1. `FileNotFoundError: config.yaml`: Automatically discovers and synchronizes `config.yaml` across `/kaggle/input/` and `/kaggle/working/`.
2. `AttributeError: '_CudaDeviceProperties' has no attribute 'total_mem'`: Uses the official PyTorch API `total_memory` with safe fallback via `getattr`.
3. `FileExistsError: [Errno 17] File exists: 'TAWOS.sql'`: Idempotent symlink management that removes stale links before recreating.

```python
# ==============================================================================
# BULLETPROOF KAGGLE & CLOUD BOOTSTRAP CELL (COPY & PASTE TO FIRST CELL ON KAGGLE)
# ==============================================================================
import os, sys, shutil

# Fix 1: Prevent FileNotFoundError for config.yaml across /kaggle directories
if not os.path.exists('config.yaml'):
    for root, _, files in os.walk('/kaggle'):
        if 'config.yaml' in files:
            shutil.copyfile(os.path.join(root, 'config.yaml'), 'config.yaml')
            print(f"✓ Synchronized config.yaml from {root}")
            break

# Fix 2: Prevent AttributeError: '_CudaDeviceProperties' has no attribute 'total_mem'
try:
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024 ** 3)
        print(f"✓ GPU Active: {props.name} | VRAM: {vram:.2f} GB | PyTorch: {torch.__version__}")
    else:
        print("✓ CPU Mode Active")
except Exception as e:
    print(f"ℹ GPU check: {e}")

# Fix 3: Prevent FileExistsError: [Errno 17] File exists: 'TAWOS.sql' (Idempotent link)
target_sql = 'TAWOS.sql'
if os.path.islink(target_sql):
    if not os.path.exists(os.path.realpath(target_sql)):
        os.unlink(target_sql)  # Remove stale link
elif not os.path.exists(target_sql):
    for root, _, files in os.walk('/kaggle/input'):
        if 'TAWOS.sql' in files:
            os.symlink(os.path.join(root, 'TAWOS.sql'), target_sql)
            print(f"✓ Idempotent symlink created for TAWOS.sql")
            break
print("✓ Cloud environment initialized successfully.")
# ==============================================================================
```

---

### Methodology

1. **Text Preprocessing:** Combine Title + Description, clean Jira markup
2. **BERT Embeddings:** Extract token-level embeddings from `bert-base-uncased`, aggregate via mean pooling → 768-dim vectors
3. **SBERT Embeddings:** Extract sentence embeddings from `all-MiniLM-L6-v2` → 384-dim vectors
4. **Combined Embeddings:** Concatenate BERT + SBERT → 1152-dim vectors
5. **Models:** Ridge Regression, Random Forest, XGBoost
6. **Evaluation:** 5-fold temporal CV on source project + held-out test on target project

### Results

See `results/tables/` for complete result tables. The comparison table reports MAE and MdAE for:
- BERT word embeddings
- SBERT sentence embeddings
- Combined BERT + SBERT embeddings

### Team Members

- [Add team member names here]

### References

- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," 2019
- Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," 2019
- TAWOS Dataset: https://doi.org/10.5522/04/21308124
