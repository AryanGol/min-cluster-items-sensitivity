# Text Clustering & Topic Modeling (BERTopic + UMAP + HDBSCAN)

A fast, end-to-end pipeline for clustering research articles and modeling topics from a CSV of titles & abstracts. Optimized for Intel Arc GPUs (XPU), with CPU fallback.

> **Author:** Aryan Golzaryan — aryan.golzaryan@gmail.com

---

## ✨ What this project does

- Cleans your **Title** and **Abstract** text and exports “before/after” views  
- Computes **top words per document**, word/token **count diagnostics**, and **768-dim embeddings** (`all-mpnet-base-v2`)  
- Reduces embeddings with **UMAP (5D + 2D)** and clusters with **HDBSCAN**  
- Fits **BERTopic** using **precomputed embeddings** for reproducible topics  
- Generates **interactive Plotly** HTML visualizations (documents, bar chart, heatmap, hierarchy, datamaps)  
- (Optional) Refines representations with **KeyBERTInspired** and **auto-labels topics** with **FLAN-T5**  
- Runs **batch experiments** across multiple `min_cluster_size` values  
- Computes a full **evaluation suite** (coverage, size balance, silhouette, DB, CH, DBCV, persistence, ARI/VI, bootstrapped consensus/PAC, coherence NPMI/UMass)

---

## 🧾 Input

A CSV with at least these columns (case-insensitive):

- `title` — article title  
- `abstract` — article abstract

Configure these near the top of the script:

```python
FOLDER = r"C:\path\to\your\folder"
CSV_NAME = "pmc_metadata.csv"
