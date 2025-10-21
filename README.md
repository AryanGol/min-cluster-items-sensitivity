# Text Clustering & Topic Modeling (BERTopic + UMAP + HDBSCAN)

*A fast, end-to-end pipeline for clustering research articles and modeling topics from a CSV of titles & abstracts. Optimized for Intel Arc GPUs (XPU), with CPU fallback.*

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

FOLDER = r"C:\path\to\your\folder"
CSV_NAME = "pmc_metadata.csv"

All outputs are written to `FOLDER/results/`.

---

## 🚀 Quickstart

1. (Optional) Intel Arc GPU (XPU)
   - Install Intel oneAPI drivers/runtime.
   - Use a Python environment where **PyTorch has XPU support**.
   - The script checks `torch.xpu.is_available()` and prints a helpful error if not detected.

2. Create and activate an environment
   conda create -n texttopic python=3.10 -y
   conda activate texttopic

3. Install dependencies
   pip install torch torchvision torchaudio
   pip install numpy pandas matplotlib scikit-learn
   pip install sentence-transformers umap-learn hdbscan bertopic
   pip install transformers plotly
   pip install openpyxl xlsxwriter

4. Configure & run
   - Edit `FOLDER`, `CSV_NAME`, and optionally `AUTO_LABEL = True/False`.
   - Run the script (or execute cells in Jupyter/Spyder) in order.

---

## 📂 Example output tree

results/
├── cleaned.csv
├── top10_words_per_title_abstract.xlsx
├── titles_abstracts_before.csv
├── titles_abstracts_after.csv
├── word_counts_before_after.xlsx
├── token_counts_title_abstract_clean.xlsx
├── embeddings_title_abstract_clean_max512.npy
├── embeddings_title_abstract_clean_max512.xlsx
├── embeddings_heatmap_docs_x_768.png
├── umap_5d.csv
├── umap_5d_heatmap.png
├── umap_2d.csv
├── umap_2d_scatter.png
├── evaluation_summary_by_min_cluster_size.csv
├── pairwise_adjacent_ARI_VI.csv
├── consensus_PAC_per_cs.csv
├── mincs_005/
│ ├── topic_info.csv
│ ├── doc_assignments.csv
│ ├── titles_by_topic.txt
│ ├── documents.html
│ ├── barchart.html
│ ├── heatmap.html
│ ├── hierarchy.html
│ ├── datamap.html (optional)
│ ├── datamap_numeric.html (optional)
│ ├── diffs_after_keybert.csv
│ └── bertopic_model/
└── mincs_010/
└── ...

---

## 📜 Citation

If you use this project, please consider citing:

- Grootendorst, M. **BERTopic** — https://github.com/MaartenGr/BERTopic  
- Reimers, N., & Gurevych, I. **Sentence-BERT** — https://arxiv.org/abs/1908.10084  
- McInnes, L. et al. **UMAP** — https://arxiv.org/abs/1802.03426  
- Campello, R. J. G. B. et al. **HDBSCAN** — https://doi.org/10.1007/s10115-013-0673-8

---

## 📄 License

MIT License © 2025 Aryan Golzaryan

---

## 🙏 Acknowledgements

Thanks to the authors/maintainers of **PyTorch**, **SentenceTransformers**, **BERTopic**, **UMAP**, **HDBSCAN**, and **Transformers**.
