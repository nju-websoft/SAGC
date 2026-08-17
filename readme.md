# Diagnosing and Correcting Geometric Bias in Diversity-Driven Code Instruction Data Selection

This repository contains the official implementation, reproduction scripts, selection metadata, and supplementary appendix for the paper **"Diagnosing and Correcting Geometric Bias in Diversity-Driven Code Instruction Data Selection"**.

📄 The supplementary appendix is available at [`supplementary_appendix.pdf`](./supplementary_appendix.pdf).

If the PDF preview does not render correctly in the browser, please download the file and open it locally.

We propose **Semantically Anchored Geometric Coverage (SAGC)**, a representation-level correction for diversity-driven code instruction data selection. SAGC uses LLM-generated macro-intent labels as semantic anchors for dense micro-context representations. Rather than replacing standard diversity objectives, SAGC reshapes the representation geometry exposed to them, guiding subset selection toward functionally organized coverage while preserving fine-grained implementation context.

Our code supports the full experimental pipeline, including macro-intent label generation, embedding construction, diversity-driven subset selection, supervised fine-tuning, and LiveCodeBench-style evaluation.

## ⚠️ Data Availability

Due to the file size limitations of GitHub, the datasets used in this paper are hosted on the **Open Science Framework (OSF)**. 

To reproduce the experiments, please access the anonymized data via the following link:
> **(https://osf.io/65wk2/overview?view_only=3bbe70cec0e64c19b8dc318c65b20b40)**

Please follow the instructions in the [Data Preparation](#-data-preparation) section to place the downloaded files correctly.

## 📂 Project Structure

The project is organized into clear modules corresponding to our data processing pipeline:

```text
.
├── data/                       # Dataset storage
│   ├── raw/                    # Original dataset inputs
│   ├── processed/              # Cleaned data for processing
│   └── training_sets/          # Final selected data for SFT
├── envs/                       # Environment configuration files
├── output/                     # Model checkpoints and evaluation results
├── results/                    # Intermediate analysis (Embeddings, Clusters, Labels)
├── scripts/                    # Shell scripts for one-click execution
└── src/                        # Source code
    ├── labeling/               # Stage 1: Label generation & analysis
    ├── embedding/              # Stage 2: Embedding generation
    ├── selection/              # Stage 3: Sampling algorithms (SAGC core)
    ├── sft/                    # Stage 4: Supervised Fine-Tuning (XTuner)
    └── eval/                   # Stage 5: Evaluation (LiveCodeBench)
```

## 💾 Data Preparation

We provide the necessary raw and processed data on OSF to reproduce our experiments. 

### 1. Download and Placement
After downloading and unziping the files from the OSF link provided above, please place them in the following directory structure:

*   **Raw Data**: Move `origin_dataset.jsonl` to:
    ```bash
    data/raw/origin_dataset.jsonl
    ```
*   **Processed Data**: Move `cleaned_labeled_dataset.jsonl` to:
    ```bash
    data/processed/cleaned_labeled_dataset.jsonl
    ```

### 2. Dataset Descriptions

*   **`data/raw/origin_dataset.jsonl`**: The raw dataset containing problem descriptions, dialogs, and metadata. Required for **Stage 1**.
*   **`data/processed/cleaned_labeled_dataset.jsonl`**: The pre-processed dataset with semantic labels used for embedding. Required for **Stage 2**.

### 📝 Note on Intermediate Files
To optimize storage, the intermediate file `labeled_dataset.jsonl` (generated before cleaning) is **not included**. 
*   If you wish to reproduce the full pipeline from scratch, running **Stage 1** will automatically regenerate this file using the raw dataset.
*   If you wish to skip label generation, you can directly proceed to **Stage 2** using the provided `cleaned_labeled_dataset.jsonl`.

### 3. Large Artifacts
Intermediate artifacts such as **Embeddings (.npy)** and **SFT Checkpoints** are not included due to size constraints. They can be fully reproduced by running the provided scripts in sequence.

## 🛠️ Environment Setup

To ensure reproducibility, we provide a complete Conda environment configuration.

### 1. Prerequisites

* **Python**: 3.10+
* **CUDA**: 11.8 or 12.1 (Recommended for VLLM and XTuner)
* **VLLM**: For efficient label generation (Stage 1).
* **XTuner**: For efficient SFT training (Stage 4).

### 2. Installation

**Step 1: Create Base Environment**
Create the environment using the provided YAML file (this installs PyTorch and base dependencies).

    # Create the environment
    conda env create -f envs/environment_main.yml

    # Activate the environment
    conda activate SAGC

**Step 2: Install Flash Attention**
Manually install `flash-attn` to ensure it links correctly to the installed PyTorch.

    # Install with --no-build-isolation to avoid version conflicts
    pip install flash-attn==2.8.3 --no-build-isolation


## 🚀 Reproduction


### Stage 1: Label Generation & Analysis
We provide a one-click script to run the full labeling pipeline. This process includes:
1.  **Generation**: Inferring semantic labels using the base model.
2.  **Cleaning**: Normalizing labels (e.g., lemmatization).
3.  **Analysis**: Filtering low-frequency labels and building the vocabulary.

```bash
    # Run the Stage 1 pipeline
    bash scripts/run_stage1_labeling.sh
```

*Note: You may need to modify `MODEL_PATH` inside the script to point to your local model checkpoint.*

### Stage 2: Dual-View Embedding Generation

```bash
    # Run the Stage 2 pipeline
    bash scripts/run_stage2_embedding.sh
```

#### Configuration Notes
* **Model Selection**: The script uses **`Qwen/Qwen3-Embedding-8B`**. Ensure this model is accessible (or modify the `MODEL_PATH` in the script to point to your local checkpoint).
* **Embedding Code Solutions**: By default, we embed the problem description (`--mode "query"`). If you wish to use code solutions for the embedding space (e.g., for ablation studies), please edit `scripts/run_stage2_embedding.sh` and change the argument to `--mode "code"`.

### Stage 3: SAGC Selection

This is the core step of our framework. It performs diversity-driven selection in a semantically anchored representation space constructed from dense micro-context and macro-intent labels.

The script runs the **SAGC (K-Means)** configuration with **Label-Max** anchoring, corresponding to one of the strongest settings reported in the paper.

```bash
    # Run the selection algorithm
    bash scripts/run_stage3_selection.sh
```

#### Output
* **`data/training_sets/selected_kmeans_max_n20000.jsonl`**: The final subset selected for SFT training.

#### Advanced Configuration (Ablation Studies)
You can modify the variables in `scripts/run_stage3_selection.sh` to reproduce our ablation studies:
* **Feature Fusion**: Change `CONCAT_TYPE` to `'sum'` or `'tfidf'` to test different label aggregation strategies.
* **Sampling Method**: Change `METHOD` to `'max_sum'` or `'max_min'` to evaluate other sampling methods.

### Stage 4: Supervised Fine-Tuning (SFT)

In this stage, we fine-tune the base model using the data selected by SAGC.

We follow a strict output structure:
* **Intermediate Results**: Checkpoints and logs are saved in `output/models/{Exp_Name}/`.
* **Final Model**: The converted HuggingFace model is saved in `output/models/{Exp_Name}/final_hf/`.

```bash
    # Run SFT Training
    bash scripts/run_stage4_sft.sh
```

#### Output Location
* **`output/models/Qwen2.5-7B-SAGC-KMeans/final_hf/`**: The final fine-tuned model weights.


### Stage 5: Evaluation (LiveCodeBench)

We use a **customized version** of the LiveCodeBench framework (embedded in `src/eval/lcb_runner`).
This modified framework includes specific extensions to support **MCS** evaluation, ensuring alignment with our proposed metrics.

To avoid dependency conflicts with the training stage (Stage 4), we run this stage in a **standalone isolated environment**.

#### 1. Setup Evaluation Environment
We use the `pyproject.toml` provided in `src/eval/` to install the customized package and its dependencies.

```bash
    # 1. Create a fresh virtual environment (Python 3.11 recommended)
    uv venv envs/.venv_eval --python 3.11

    # 2. Activate the environment
    source envs/.venv_eval/bin/activate

    # 3. Install the local lcb_runner package and its dependencies
    #    This reads src/eval/pyproject.toml to install required libs (vLLM, etc.)
    #    and links your local code to the environment.
    uv pip install -e src/eval/
    
    # 4. Deactivate (The script will use the python binary directly)
    deactivate
```

#### 2. Run Evaluation Script
The script `scripts/run_stage5_eval.sh` is pre-configured to use this isolated environment.
```bash
    # Run Evaluation (Inference + Scoring)
    bash scripts/run_stage5_eval.sh
```

#### Output
* **`output/evaluation/{Exp_Name}/generations_scores.json`**: The final comprehensive metrics, including the **Standard MCS** score reported in our paper.


## 📄 License

This project is licensed under the **Apache 2.0 License**.

## 🙏 Acknowledgements

We express our gratitude to the following open-source projects that made this work possible:
* **[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)** for the robust evaluation framework.
* **[XTuner](https://github.com/InternLM/xtuner)** for the efficient SFT training infrastructure.
* **[Qwen](https://github.com/QwenLM/Qwen)** for the powerful base models and embedding models.
