# %% [1] Config & basic imports (EDIT FOLDER IF NEEDED)
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")  # quiet IProgress warning

FOLDER = r"C:\PhD Files\Organized\SPARK\July 7 - 2025 - Data Sent by Barbora\XML of PubMed Articles\MaterialsMethods"
CSV_NAME = "pmc_metadata.csv"
RESULTS_DIR = os.path.join(FOLDER, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Choose whether to auto-label topics with FLAN-T5 (downloads once)
AUTO_LABEL = True

# %% [2] Imports that don't depend on BERTopic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# %% [3] Safe-import BERTopic (spaCy disabled to avoid NumPy/Thinc binary issues)
import builtins as _builtins
__real_import = _builtins.__import__

def _no_spacy_import(name, *args, **kwargs):
    if name.startswith("spacy"):
        raise ModuleNotFoundError("spaCy intentionally disabled for BERTopic import.")
    return __real_import(name, *args, **kwargs)

_builtins.__import__ = _no_spacy_import
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, TextGeneration
_builtins.__import__ = __real_import  # restore normal import

# (Only needed if AUTO_LABEL=True)
if AUTO_LABEL:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# %% [4] Helper functions
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def make_scatter_png(emb_2d, clusters, titles, out_png):
    df = pd.DataFrame(emb_2d, columns=["x", "y"])
    df["title"] = list(titles)
    df["cluster"] = clusters
    clusters_df = df[df.cluster != -1]
    outliers_df = df[df.cluster == -1]

    plt.figure(figsize=(8, 6))
    if len(outliers_df):
        plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2, c="grey")
    if len(clusters_df):
        plt.scatter(
            clusters_df.x, clusters_df.y,
            c=clusters_df.cluster.astype(int), alpha=0.6, s=2, cmap="tab20b"
        )
    plt.axis("off")
    plt.title("UMAP (2D) with HDBSCAN clusters")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def topic_differences(model, original_topics, nr_topics=None):
    if nr_topics is None:
        nr_topics = len([t for t in model.get_topic_info().Topic.unique() if t != -1])
    rows = []
    for topic in range(nr_topics):
        og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
        rows.append([topic, og_words, new_words])
    return pd.DataFrame(rows, columns=["Topic", "Original", "Updated"])


# %% [5] Load & clean CSV (saves a cleaned copy next to the file)
in_csv = os.path.join(FOLDER, CSV_NAME)
if not os.path.isfile(in_csv):
    raise FileNotFoundError(f"CSV not found at: {in_csv}")

df = pd.read_csv(in_csv, encoding="latin-1")

# Accept Title/Abstract in any case
lower2orig = {c.lower(): c for c in df.columns}
title_col = lower2orig.get("title") or ("Title" if "Title" in df.columns else None)
abstract_col = lower2orig.get("abstract") or ("Abstract" if "Abstract" in df.columns else None)
if title_col is None or abstract_col is None:
    raise ValueError("CSV must contain 'Title' and 'Abstract' columns (case-insensitive).")

df_clean = df.dropna(subset=[abstract_col]).copy()
df_clean = df_clean[df_clean[abstract_col].astype(str).str.strip().astype(bool)]

cleaned_out = os.path.join(FOLDER, "Clinical Dual Tracer_cleaned.csv")
df_clean.to_csv(cleaned_out, index=False, encoding="utf-8")
print(f"✔ Saved cleaned CSV to: {cleaned_out}")

titles = df_clean[title_col].fillna("").astype(str)
abstracts = df_clean[abstract_col].fillna("").astype(str).tolist()


# %% [6] Compute embeddings ONCE (CPU)
embedding_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
embeddings = embedding_model.encode(
    abstracts, show_progress_bar=True, convert_to_numpy=True, batch_size=32
)
print("Embeddings shape:", embeddings.shape)



# %% [7] Compute UMAP reductions ONCE
umap_5d = UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42)
emb_5d = umap_5d.fit_transform(embeddings)

umap_2d = UMAP(n_components=2, min_dist=0.0, metric="cosine", random_state=42)
emb_2d = umap_2d.fit_transform(embeddings)

print("UMAP 5D:", emb_5d.shape, "UMAP 2D:", emb_2d.shape)


# %% [8] RUN ONE CLUSTER SIZE (edit MIN_CS, then run this cell)
MIN_CS = 20  # <-- change to 5, 10, 20, 30, 50, 75, or 100

outdir = os.path.join(RESULTS_DIR, f"mincs_{MIN_CS:03d}")
ensure_dir(outdir)
print(f"\n==== Running min_cluster_size={MIN_CS} ====")

# 8.1 HDBSCAN labels for scatter (fit on 5D UMAP)
labels = HDBSCAN(
    min_cluster_size=MIN_CS, metric="euclidean", cluster_selection_method="eom"
).fit_predict(emb_5d)

# 8.2 Save scatter plot (2D UMAP)
scatter_path = os.path.join(outdir, f"scatter_umap2d_mincs_{MIN_CS:03d}.png")
make_scatter_png(emb_2d, labels, titles, scatter_path)

# 8.3 Fit BERTopic with this min_cluster_size (use precomputed embeddings)
# IMPORTANT CHANGE: embedding_model=embedding_model  (NOT None)
topic_model = BERTopic(
    embedding_model=embedding_model,   # <- keeps an embedder for update_topics
    umap_model=UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42),
    hdbscan_model=HDBSCAN(min_cluster_size=MIN_CS, metric="euclidean", cluster_selection_method="eom"),
    verbose=True,
    # calculate_probabilities=False,  # optional
).fit(abstracts, embeddings)

# 8.4 Save topic info + titles by topic
info = topic_model.get_topic_info()
info.to_csv(os.path.join(outdir, "topic_info.csv"), index=False)

from collections import defaultdict
titles_by_category = defaultdict(list)
for title, topic in zip(titles, topic_model.topics_):
    titles_by_category[topic].append(title)

with open(os.path.join(outdir, "titles_by_topic.txt"), "w", encoding="utf-8") as f:
    for cat, papers in sorted(titles_by_category.items()):
        f.write(f"Topic {cat}:\n")
        for p in papers:
            f.write(f"- {p}\n")
        f.write("\n")

# 8.5 Plotly visualizations → HTML files
docs_fig = topic_model.visualize_documents(
    titles.tolist(), reduced_embeddings=emb_2d, width=1200, hide_annotations=True
)
docs_fig.update_layout(font=dict(size=16))
docs_fig.write_html(os.path.join(outdir, "documents.html"))

topic_model.visualize_barchart().write_html(os.path.join(outdir, "barchart.html"))
topic_model.visualize_heatmap(n_clusters=15).write_html(os.path.join(outdir, "heatmap.html"))
topic_model.visualize_hierarchy().write_html(os.path.join(outdir, "hierarchy.html"))

# 8.6 Update representations with KeyBERTInspired and save differences
original_topics = __import__("copy").deepcopy(topic_model.topic_representations_)
kb_model = KeyBERTInspired()  # will use topic_model.embedding_model internally
topic_model.update_topics(abstracts, representation_model=kb_model)
diffs = topic_differences(topic_model, original_topics)
diffs.to_csv(os.path.join(outdir, "diffs_after_keybert.csv"), index=False)

# 8.7 Optional: Auto-label with FLAN-T5
if AUTO_LABEL:
    print("Auto-labeling with FLAN-T5-small…")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    tg_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    hf_pipe = pipeline("text2text-generation", model=tg_model, tokenizer=tokenizer, max_length=64, do_sample=False)

    prompt = """
    I have a topic that contains the following documents:
    [DOCUMENTS]

    The topic is described by the following keywords: [KEYWORDS]

    Based on the information above, extract a short topic label in the following format:
    topic: <short topic label>
    """
    tg_representation = TextGeneration(hf_pipe, prompt=prompt)

    original_labels = topic_model.topic_labels_.copy()
    topic_model.update_topics(abstracts, topic_model.topics_, representation_model=tg_representation)

    for topic_id, old_label in original_labels.items():
        new_label = topic_model.topic_labels_.get(topic_id, "")
        if old_label != new_label:
            print(f"Topic {topic_id}: '{old_label}' → '{new_label}'")

# 8.8 Datamap visualizations (if datamapplot is installed)
try:
    freq_df = topic_model.get_topic_freq()
    valid_topics = sorted(freq_df.loc[freq_df.Topic != -1, "Topic"].tolist())
    topics_to_show = valid_topics[:17] if valid_topics else []

    topic_model.visualize_document_datamap(
        titles.tolist(), topics=topics_to_show, reduced_embeddings=emb_2d, width=1200
    ).write_html(os.path.join(outdir, "datamap.html"))

    num_labels = {topic: str(topic) for topic in freq_df.Topic}
    topic_model.set_topic_labels(num_labels)
    topic_model.visualize_document_datamap(
        titles.tolist(), topics=topics_to_show, reduced_embeddings=emb_2d,
        width=1200, custom_labels=True, topic_prefix=False
    ).write_html(os.path.join(outdir, "datamap_numeric.html"))
except Exception as e:
    print(f"(Skipping datamap: {e})")

print(f"✔ Done for min_cluster_size={MIN_CS}. Results → {outdir}")



# %% [9] OPTIONAL: Run ALL sizes automatically (5,10,20,30,50,75,100)
CLUSTER_SIZES = [5, 10, 20, 30, 50, 75, 100]

# Prepare FLAN-T5 only once if enabled
hf_pipe = None
if AUTO_LABEL:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    tokenizer.model_max_length = 512          # avoid long-sequence warnings
    tg_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    hf_pipe = pipeline(
        "text2text-generation",
        model=tg_model,
        tokenizer=tokenizer,
        max_length=64,
        do_sample=False,
        truncation=True
    )

for MIN_CS in CLUSTER_SIZES:
    outdir = os.path.join(RESULTS_DIR, f"mincs_{MIN_CS:03d}")
    ensure_dir(outdir)
    print(f"\n==== Running min_cluster_size={MIN_CS} ====")

    # HDBSCAN labels (on precomputed 5D UMAP) for the static scatter PNG
    labels = HDBSCAN(
        min_cluster_size=MIN_CS, metric="euclidean", cluster_selection_method="eom"
    ).fit_predict(emb_5d)

    scatter_path = os.path.join(outdir, f"scatter_umap2d_mincs_{MIN_CS:03d}.png")
    make_scatter_png(emb_2d, labels, titles, scatter_path)

    # Keep an embedder so KeyBERTInspired/TextGeneration can embed docs
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=MIN_CS, metric="euclidean", cluster_selection_method="eom"),
        verbose=True,
    ).fit(abstracts, embeddings)

    # Save model + assignments for full restore later
    topic_model.save(os.path.join(outdir, "bertopic_model"))
    pd.DataFrame({"Title": list(titles), "Topic": topic_model.topics_}).to_csv(
        os.path.join(outdir, "doc_assignments.csv"), index=False
    )

    # Topic info + titles by topic
    info = topic_model.get_topic_info()
    info.to_csv(os.path.join(outdir, "topic_info.csv"), index=False)

    from collections import defaultdict
    titles_by_category = defaultdict(list)
    for title, topic in zip(titles, topic_model.topics_):
        titles_by_category[topic].append(title)
    with open(os.path.join(outdir, "titles_by_topic.txt"), "w", encoding="utf-8") as f:
        for cat, papers in sorted(titles_by_category.items()):
            f.write(f"Topic {cat}:\n")
            for p in papers:
                f.write(f"- {p}\n")
            f.write("\n")

    # Plotly visualizations
    topic_model.visualize_barchart().write_html(os.path.join(outdir, "barchart.html"))

    # Heatmap: pick a valid n_clusters (< number of topics)
    n_topics = int((info.Topic != -1).sum())
    if n_topics > 1:
        heatmap_k = max(1, min(15, n_topics - 1))
        topic_model.visualize_heatmap(n_clusters=heatmap_k).write_html(
            os.path.join(outdir, "heatmap.html")
        )
    else:
        print("Skipping heatmap: need at least 2 topics.")

    topic_model.visualize_hierarchy().write_html(os.path.join(outdir, "hierarchy.html"))

    docs_fig = topic_model.visualize_documents(
        titles.tolist(), reduced_embeddings=emb_2d, width=1200, hide_annotations=True
    )
    docs_fig.update_layout(font=dict(size=16))
    docs_fig.write_html(os.path.join(outdir, "documents.html"))

    # KeyBERT refinement (uses topic_model.embedding_model)
    original_topics = __import__("copy").deepcopy(topic_model.topic_representations_)
    kb_model = KeyBERTInspired()
    topic_model.update_topics(abstracts, representation_model=kb_model)
    diffs = topic_differences(topic_model, original_topics)
    diffs.to_csv(os.path.join(outdir, "diffs_after_keybert.csv"), index=False)

    # Optional auto-label
    if AUTO_LABEL and hf_pipe is not None:
        prompt = """
        I have a topic that contains the following documents:
        [DOCUMENTS]

        The topic is described by the following keywords: [KEYWORDS]

        Based on the information above, extract a short topic label in the following format:
        topic: <short topic label>
        """
        tg_representation = TextGeneration(hf_pipe, prompt=prompt)
        topic_model.update_topics(abstracts, topic_model.topics_, representation_model=tg_representation)

    # Datamap: handle Plotly *or* Matplotlib backends
    try:
        freq_df = topic_model.get_topic_freq()
        valid_topics = sorted(freq_df.loc[freq_df.Topic != -1, "Topic"].tolist())
        topics_to_show = valid_topics[:17] if valid_topics else []

        fig_dm = topic_model.visualize_document_datamap(
            titles.tolist(), topics=topics_to_show, reduced_embeddings=emb_2d, width=1200
        )
        # Plotly?
        if hasattr(fig_dm, "write_html"):
            fig_dm.write_html(os.path.join(outdir, "datamap.html"))
        # Matplotlib?
        elif hasattr(fig_dm, "savefig"):
            fig_dm.savefig(os.path.join(outdir, "datamap.png"), dpi=200, bbox_inches="tight")

        # numeric labels variant
        num_labels = {topic: str(topic) for topic in freq_df.Topic}
        topic_model.set_topic_labels(num_labels)
        fig_dm2 = topic_model.visualize_document_datamap(
            titles.tolist(), topics=topics_to_show, reduced_embeddings=emb_2d,
            width=1200, custom_labels=True, topic_prefix=False
        )
        if hasattr(fig_dm2, "write_html"):
            fig_dm2.write_html(os.path.join(outdir, "datamap_numeric.html"))
        elif hasattr(fig_dm2, "savefig"):
            fig_dm2.savefig(os.path.join(outdir, "datamap_numeric.png"), dpi=200, bbox_inches="tight")

    except Exception as e:
        print(f"(Skipping datamap: {e})")

    print(f"✔ Finished min_cluster_size={MIN_CS} → {outdir}")

print("✔ All runs complete.")



# %% [A] Setup: paths, sizes, imports, small helpers
import os, math, numpy as np, pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, mutual_info_score
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy as shannon_entropy
from umap import UMAP
from hdbscan import HDBSCAN
from hdbscan.validity import validity_index as hdbscan_validity_index  # DBCV

# EDIT to your results folder
RESULTS_DIR = os.path.join(
    r"C:\PhD Files\Organized\SPARK\July 7 - 2025 - Data Sent by Barbora\XML of PubMed Articles\MaterialsMethods",
    "results"
)
CLUSTER_SIZES = [5, 10, 20, 30, 50, 75, 100]

# Safety caps for heavy metrics (set higher if you have RAM/time)
MAX_SAMPLES_SIL = 10000    # silhouette, DB, CH
MAX_SAMPLES_DBCV = 8000    # DBCV
CONSENSUS_MAX_N = 3000     # consensus matrix (N x N memory!)
N_BOOTSTRAPS = 8           # for consensus/PAC per min_cs

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def gini_coefficient(sizes):
    sizes = np.asarray(sizes, dtype=float)
    sizes = sizes[sizes > 0]
    k = len(sizes)
    if k == 0: return np.nan
    denom = 2 * k * sizes.sum()
    if denom == 0: return 0.0
    diff_sum = np.abs(sizes.reshape(-1,1) - sizes.reshape(1,-1)).sum()
    return diff_sum / denom

def size_entropy(sizes):
    sizes = np.asarray(sizes, dtype=float)
    sizes = sizes[sizes > 0]
    if len(sizes) == 0: return np.nan, np.nan
    p = sizes / sizes.sum()
    H = shannon_entropy(p)                # natural log
    H_norm = H / math.log(len(p)) if len(p) > 1 else 0.0
    return H, H_norm

def coef_variation(sizes):
    sizes = np.asarray(sizes, dtype=float)
    sizes = sizes[sizes > 0]
    if len(sizes) == 0 or sizes.mean() == 0: return np.nan
    return sizes.std(ddof=1) / sizes.mean()

def VI(u, v):
    # Variation of Information: H(u)+H(v)-2*I(u;v)
    Hu = shannon_entropy(np.bincount(np.asarray(u)[np.asarray(u)>=0]))
    Hv = shannon_entropy(np.bincount(np.asarray(v)[np.asarray(v)>=0]))
    I = mutual_info_score(u, v)
    return Hu + Hv - 2*I



# %% [B] Recover labels per min_cs (RECOMPUTE from your emb_5d to ensure consistency)
# Requires you ran earlier cells to create: `embeddings`, `emb_5d`, and `titles`.
# If not present, re-run your Cells 6–7.

labels_by_cs = {}
stability_by_cs = {}   # HDBSCAN intrinsic stability (cluster_persistence_)
coverage_by_cs = {}

for cs in CLUSTER_SIZES:
    print(f"Fitting HDBSCAN on 5D UMAP: min_cluster_size={cs}")
    hdb = HDBSCAN(
        min_cluster_size=cs, metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True, gen_min_span_tree=True
    ).fit(emb_5d)

    labels = hdb.labels_
    labels_by_cs[cs] = labels
    coverage_by_cs[cs] = (labels != -1).mean()
    stability_by_cs[cs] = getattr(hdb, "cluster_persistence_", None)  # array of per-cluster persistence
print("Done.")


# %% [C] Coverage, #clusters, size balance (Gini/Entropy/CV)
rows = []
N = len(embeddings)

for cs in CLUSTER_SIZES:
    labels = labels_by_cs[cs]
    inlier = labels[labels != -1]
    k = len(np.unique(inlier))
    sizes = np.bincount(inlier) if k>0 else np.array([])
    G = gini_coefficient(sizes)
    H, Hn = size_entropy(sizes)
    CV = coef_variation(sizes)
    rows.append({
        "min_cluster_size": cs,
        "N_docs": N,
        "coverage": float(coverage_by_cs[cs]),
        "n_clusters": int(k),
        "max_cluster_size": int(sizes.max() if sizes.size else 0),
        "gini_sizes": G,
        "entropy_sizes": H,
        "entropy_sizes_normalized": Hn,
        "cv_sizes": CV
    })

metrics_sizes = pd.DataFrame(rows).sort_values("min_cluster_size")
metrics_sizes


# %% [D] Silhouette, Davies–Bouldin, Calinski–Harabasz (cosine on embeddings)
# Filters: ignore noise (-1) and drop singleton clusters for silhouette.

from collections import Counter
rows = []

for cs in CLUSTER_SIZES:
    labels = labels_by_cs[cs]
    mask = labels != -1
    X = embeddings[mask]
    y = labels[mask]
    # drop singletons for silhouette (sklearn warns/errors on clusters of size 1)
    cnt = Counter(y)
    keep = np.array([cnt[lab] >= 2 for lab in y])
    X_sil = X[keep]
    y_sil = y[keep]
    res = {"min_cluster_size": cs}

    try:
        if len(np.unique(y_sil)) >= 2 and len(y_sil) > 10:
            # subsample if huge
            if len(y_sil) > MAX_SAMPLES_SIL:
                idx = np.random.RandomState(42).choice(len(y_sil), size=MAX_SAMPLES_SIL, replace=False)
                X_s = X_sil[idx]; y_s = y_sil[idx]
            else:
                X_s, y_s = X_sil, y_sil
            res["silhouette_cosine"] = float(silhouette_score(X_s, y_s, metric="cosine"))
        else:
            res["silhouette_cosine"] = np.nan
    except Exception as e:
        res["silhouette_cosine"] = np.nan

    try:
        if len(np.unique(y)) >= 2:
            # Subsample for DB/CH if very large (cosine)
            if len(y) > MAX_SAMPLES_SIL:
                idx = np.random.RandomState(0).choice(len(y), size=MAX_SAMPLES_SIL, replace=False)
                X_db = embeddings[mask][idx]; y_db = y[idx]
            else:
                X_db, y_db = X, y
            # DB/CH in sklearn accept metric only for silhouette;
            # For DB/CH we can pre-transform with cosine distances via mapping.
            # Practically, DB/CH are defined on Euclidean; many folks still compute on vector space.
            res["davies_bouldin"] = float(davies_bouldin_score(X_db, y_db))
            res["calinski_harabasz"] = float(calinski_harabasz_score(X_db, y_db))
        else:
            res["davies_bouldin"] = np.nan
            res["calinski_harabasz"] = np.nan
    except Exception as e:
        res["davies_bouldin"] = np.nan
        res["calinski_harabasz"] = np.nan

    rows.append(res)

metrics_sep = pd.DataFrame(rows).sort_values("min_cluster_size")
metrics_sep


# %% [E] DBCV (density-based cluster validity index) — higher is better
# Compute on a float64 cosine distance matrix; pass d=<embedding dimension> for precomputed.
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

rows = []
for cs in CLUSTER_SIZES:
    labels = labels_by_cs[cs]

    # Need at least 2 non-noise clusters
    k = len(np.unique(labels[labels != -1]))
    if k < 2:
        rows.append({"min_cluster_size": cs, "DBCV_cosine": np.nan})
        continue

    # Subsample if necessary (O(n^2) memory/time)
    if len(labels) > MAX_SAMPLES_DBCV:
        rng = np.random.RandomState(123)
        idx = rng.choice(len(labels), size=MAX_SAMPLES_DBCV, replace=False)
        X = embeddings[idx]
        y = labels[idx]
    else:
        X = embeddings
        y = labels

    # Ensure float64 and normalize (helps avoid NaNs/Infs with cosine)
    X64 = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X64, axis=1, keepdims=True)
    X64 = X64 / (norms + 1e-12)

    # Cosine distance matrix in float64
    D = pairwise_distances(X64, metric="cosine").astype(np.float64, copy=False)

    # Precomputed metric requires 'd' (original feature dimension)
    d_feat = X64.shape[1]
    val = hdbscan_validity_index(D, y, metric="precomputed", d=d_feat)

    rows.append({"min_cluster_size": cs, "DBCV_cosine": float(val)})

metrics_dbcv = pd.DataFrame(rows).sort_values("min_cluster_size")
metrics_dbcv


# %% [F] HDBSCAN intrinsic stability (cluster persistence)
# Summarize persistence stats; higher means more robust clusters (per cs)
rows = []
for cs in CLUSTER_SIZES:
    pers = stability_by_cs[cs]
    if pers is None or len(pers) == 0:
        rows.append({
            "min_cluster_size": cs, "persistence_mean": np.nan,
            "persistence_median": np.nan, "persistence_min": np.nan, "persistence_max": np.nan
        })
    else:
        rows.append({
            "min_cluster_size": cs,
            "persistence_mean": float(np.mean(pers)),
            "persistence_median": float(np.median(pers)),
            "persistence_min": float(np.min(pers)),
            "persistence_max": float(np.max(pers)),
            "n_clusters": int(len(pers))
        })

metrics_persist = pd.DataFrame(rows).sort_values("min_cluster_size")
metrics_persist


# %% [G] Stability across settings: ARI and VI between adjacent min_cs values
pairs = []
for a, b in zip(CLUSTER_SIZES[:-1], CLUSTER_SIZES[1:]):
    La = labels_by_cs[a]
    Lb = labels_by_cs[b]
    # Align by masking only docs that are clustered in BOTH (drop -1)
    mask = (La != -1) & (Lb != -1)
    if mask.sum() < 5:
        pairs.append({"cs_a": a, "cs_b": b, "ARI": np.nan, "VI": np.nan, "n_common": int(mask.sum())})
        continue
    ARI = adjusted_rand_score(La[mask], Lb[mask])
    VIv = VI(La[mask], Lb[mask])
    pairs.append({"cs_a": a, "cs_b": b, "ARI": float(ARI), "VI": float(VIv), "n_common": int(mask.sum())})

metrics_pairwise = pd.DataFrame(pairs)
metrics_pairwise


# %% [H] Consensus matrix & PAC per min_cs (bootstraps) — may be heavy; uses sub-sampling
# We'll vary UMAP random_state and 90% subsampling to create multiple runs per cs.
# For large N, we cap to CONSENSUS_MAX_N docs (random subset) to keep memory OK.

rng = np.random.RandomState(7)
consensus_pac_rows = []

for cs in CLUSTER_SIZES:
    print(f"Consensus/PAC for cs={cs}")
    # choose subset indices for consensus to control memory
    if len(embeddings) > CONSENSUS_MAX_N:
        idx_all = rng.choice(len(embeddings), size=CONSENSUS_MAX_N, replace=False)
    else:
        idx_all = np.arange(len(embeddings))
    M = len(idx_all)
    C = np.zeros((M, M), dtype=np.float32)
    counts = np.zeros((M, M), dtype=np.uint16)  # times both present

    for r in range(N_BOOTSTRAPS):
        # 90% bootstrap on the subset
        idx_run = np.sort(rng.choice(idx_all, size=int(0.9*M), replace=False))
        pos = {i: p for p, i in enumerate(idx_run)}  # map doc id -> position in this run

        # new UMAP seed to create slightly different neighborhoods
        umap_run = UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=100+r)
        emb5 = umap_run.fit_transform(embeddings[idx_run])

        # HDBSCAN
        hdb = HDBSCAN(min_cluster_size=cs, metric="euclidean", cluster_selection_method="eom")
        lab = hdb.fit_predict(emb5)

        # update consensus: for each cluster (ignore -1), mark co-clustered pairs
        labels_run = pd.Series(lab)
        for cl in np.unique(lab):
            if cl == -1: 
                continue
            members = np.where(lab == cl)[0]
            if len(members) < 2: 
                continue
            # update both C and counts for all pairs in this cluster
            for a_i in members:
                a = idx_run[a_i]
                pa = np.where(idx_all == a)[0][0]
                # vectorized add for pairs (pa, pb)
                pb_idx = [np.where(idx_all == idx_run[b_i])[0][0] for b_i in members]
                C[pa, pb_idx] += 1
                counts[pa, pb_idx] += 1

        # also count co-presence (both sampled in this run) even if not co-clustered
        for a_pos, a in enumerate(idx_run):
            pa = np.where(idx_all == a)[0][0]
            # everyone present with 'a' in this run gets +1 to counts
            pb_idx = [np.where(idx_all == i)[0][0] for i in idx_run]
            counts[pa, pb_idx] += 1

    # avoid divide-by-zero
    mask = counts > 0
    consensus = np.zeros_like(C, dtype=np.float32)
    consensus[mask] = C[mask] / counts[mask]
    # PAC = fraction in (0.1, 0.9)
    amb_mask = (consensus > 0.1) & (consensus < 0.9)
    pac = float(amb_mask.sum() / (M*M))

    # Save PAC and (optionally) consensus
    outdir = os.path.join(RESULTS_DIR, f"mincs_{cs:03d}")
    np.savez_compressed(os.path.join(outdir, "consensus_subset.npz"),
                        indices=idx_all, consensus=consensus.astype(np.float16))
    consensus_pac_rows.append({"min_cluster_size": cs, "PAC_0.1_0.9": pac, "N_used": int(M)})

consensus_pac = pd.DataFrame(consensus_pac_rows)
consensus_pac


# %% [I] Topic coherence (NPMI / UMass) — Gensim-free, pure Python/Numpy
# No gensim/scipy required; works with your existing BERTopic runs.
import os, re, math, numpy as np, pandas as pd
from itertools import combinations
from collections import Counter, defaultdict

# --- small, fast tokenizer ---
def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    toks = [t for t in text.split() if len(t) > 2]
    return toks

# Prepare tokenized docs once
texts = [tokenize(a) for a in abstracts]
N_docs = len(texts)

TOPK = 10             # top-k words per topic to evaluate
MEASURES = ("npmi", "umass")  # choose any subset of ("npmi", "umass")

# --- utilities to build doc-frequency stats limited to a vocab ---
def build_df_counts(texts, vocab_set):
    """Return per-word document counts and per-pair co-doc counts restricted to vocab_set."""
    df_word = Counter()
    df_pair = Counter()
    for tokens in texts:
        present = sorted(set([t for t in tokens if t in vocab_set]))
        if not present:
            continue
        df_word.update(present)
        if len(present) >= 2:
            for a, b in combinations(present, 2):
                df_pair[(a, b)] += 1
    return df_word, df_pair

def npmi_from_counts(words, df_word, df_pair, N):
    """Average NPMI over all unordered pairs from 'words'."""
    pairs = list(combinations(words, 2))
    if not pairs:
        return np.nan
    vals = []
    for a, b in pairs:
        df_ab = df_pair.get((a, b), 0) if (a, b) in df_pair else df_pair.get((b, a), 0)
        if df_ab == 0:
            # define NPMI=-1 when never co-occur
            vals.append(-1.0)
            continue
        p_ab = df_ab / N
        p_a  = df_word.get(a, 0) / N
        p_b  = df_word.get(b, 0) / N
        # numerical guard
        if p_a <= 0 or p_b <= 0 or p_ab <= 0:
            vals.append(-1.0)
            continue
        pmi = math.log(p_ab / (p_a * p_b))
        npmi = pmi / (-math.log(p_ab))
        vals.append(npmi)
    return float(np.mean(vals)) if vals else np.nan

def umass_from_counts(words, df_word, df_pair):
    """
    UMass coherence (doc-based):
    C_UMass = average over i>j of log( (D(w_i, w_j) + 1) / D(w_j) )
    Using the ranking order given by 'words'.
    """
    vals = []
    for i in range(1, len(words)):
        wi = words[i]
        for j in range(0, i):
            wj = words[j]
            df_wj = df_word.get(wj, 0)
            df_wiwj = df_pair.get((wi, wj), 0) if (wi, wj) in df_pair else df_pair.get((wj, wi), 0)
            # smoothing +1 in numerator; denominator must be >=1 to be meaningful
            if df_wj <= 0:
                continue
            val = math.log((df_wiwj + 1.0) / df_wj)
            vals.append(val)
    return float(np.mean(vals)) if vals else np.nan

# Try to load saved models per cs; if not present, re-fit (light) using existing embeddings
# We keep the spaCy-disabled import trick already in your notebook.
import builtins as _builtins
__real_import = _builtins.__import__
def _no_spacy_import(name, *args, **kwargs):
    if name.startswith("spacy"):
        raise ModuleNotFoundError("spaCy intentionally disabled for BERTopic import.")
    return __real_import(name, *args, **kwargs)
_builtins.__import__ = _no_spacy_import
from bertopic import BERTopic
_builtins.__import__ = __real_import

rows = []

for cs in CLUSTER_SIZES:
    outdir = os.path.join(RESULTS_DIR, f"mincs_{cs:03d}")
    model_path = os.path.join(outdir, "bertopic_model")

    # Load or (if missing) fit a light model for this cs
    try:
        model = BERTopic.load(model_path)
    except Exception:
        model = BERTopic(
            embedding_model=embedding_model,  # keep embedder for consistency
            umap_model=UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42),
            hdbscan_model=HDBSCAN(min_cluster_size=cs, metric="euclidean", cluster_selection_method="eom"),
            verbose=False,
        ).fit(abstracts, embeddings)

    info = model.get_topic_info()
    topic_ids = [t for t in info.Topic.tolist() if t != -1]

    # Collect top-k words per topic
    topics_words = []
    for t in topic_ids:
        candidates = [w for (w, s) in model.get_topic(t)[:TOPK]]
        if len(candidates) >= 2:
            topics_words.append(candidates)

    if not topics_words:
        rows.append({"min_cluster_size": cs, "coherence_npmi": np.nan, "coherence_umass": np.nan,
                     "n_topics_evaluated": 0})
        continue

    # Restrict DF counting to the union vocab across topics for speed
    vocab = set(w for topic in topics_words for w in topic)
    df_word, df_pair = build_df_counts(texts, vocab)

    # Compute coherences
    coh_vals_npmi  = [npmi_from_counts(topic, df_word, df_pair, N_docs) for topic in topics_words] if "npmi" in MEASURES else []
    coh_vals_umass = [umass_from_counts(topic, df_word, df_pair)        for topic in topics_words] if "umass" in MEASURES else []

    coh_npmi  = float(np.nanmean(coh_vals_npmi))  if coh_vals_npmi  else np.nan
    coh_umass = float(np.nanmean(coh_vals_umass)) if coh_vals_umass else np.nan

    rows.append({
        "min_cluster_size": cs,
        "coherence_npmi": coh_npmi,
        "coherence_umass": coh_umass,
        "n_topics_evaluated": len(topics_words)
    })

metrics_coherence = pd.DataFrame(rows).sort_values("min_cluster_size")
metrics_coherence



# %% [J] Collect everything into one summary and save CSVs
summary = metrics_sizes.merge(metrics_sep, on="min_cluster_size", how="left") \
                       .merge(metrics_dbcv, on="min_cluster_size", how="left") \
                       .merge(metrics_persist, on="min_cluster_size", how="left")

if not metrics_coherence.empty:
    summary = summary.merge(metrics_coherence, on="min_cluster_size", how="left")

print(summary)
summary_path = os.path.join(RESULTS_DIR, "evaluation_summary_by_min_cluster_size.csv")
summary.to_csv(summary_path, index=False)
print(f"✔ Saved summary: {summary_path}")

pairs_path = os.path.join(RESULTS_DIR, "pairwise_adjacent_ARI_VI.csv")
metrics_pairwise.to_csv(pairs_path, index=False)
print(f"✔ Saved ARI/VI (adjacent): {pairs_path}")

consensus_path = os.path.join(RESULTS_DIR, "consensus_PAC_per_cs.csv")
consensus_pac.to_csv(consensus_path, index=False)
print(f"✔ Saved PAC per cs: {consensus_path}")










