

import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Optional: silence tqdm's IProgress warning in Spyder ----
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")

# ---------- Paths (EDIT THIS) ----------
FOLDER = r"C:\PhD Files\Organized\SPARK\July 7 - 2025 - Data Sent by Barbora\XML of PubMed Articles\MaterialsMethods"
CSV_NAME = "pmc_metadata.csv"

in_csv  = os.path.join(FOLDER, CSV_NAME)
out_csv = os.path.join(FOLDER, "Clinical Dual Tracer_cleaned.csv")
titles_by_topic_txt = os.path.join(
    FOLDER, "n_components_20_min_cluster_size_remainingsfromtopic0.txt"
)

# ---------- Safe-import BERTopic with spaCy disabled ----------
# BERTopic tries to import spaCy for optional POS representations.
# If spaCy is installed but binary-incompatible, import crashes.
# We temporarily pretend spaCy is missing so BERTopic falls back cleanly.
import builtins as _builtins
__real_import = _builtins.__import__

def _no_spacy_import(name, *args, **kwargs):
    if name.startswith("spacy"):
        raise ModuleNotFoundError("spaCy intentionally disabled for BERTopic import.")
    return __real_import(name, *args, **kwargs)

_builtins.__import__ = _no_spacy_import
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, TextGeneration
_builtins.__import__ = __real_import  # restore normal importing immediately

# ---------- Rest of imports ----------
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


def main():
    # ---------- Load & clean ----------
    if not os.path.isfile(in_csv):
        raise FileNotFoundError(f"Could not find CSV at: {in_csv}")

    df = pd.read_csv(in_csv, encoding="latin-1")

    # Normalize column names just in case (Title/Abstract expected)
    cols = {c.lower(): c for c in df.columns}
    title_col = cols.get("title", "Title" if "Title" in df.columns else None)
    abstract_col = cols.get("abstract", "Abstract" if "Abstract" in df.columns else None)
    if title_col is None or abstract_col is None:
        raise ValueError("CSV must contain 'Title' and 'Abstract' columns (case-insensitive).")

    df_cleaned = df.dropna(subset=[abstract_col])
    df_cleaned = df_cleaned[df_cleaned[abstract_col].astype(str).str.strip().astype(bool)]
    df_cleaned.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✔ Saved cleaned CSV to: {out_csv}")

    titles = df_cleaned[title_col].fillna("").astype(str)
    abstracts = df_cleaned[abstract_col].fillna("").astype(str).tolist()

    # ---------- Embeddings ----------
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    embeddings = embedding_model.encode(
        abstracts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32,
    )
    print("Embeddings shape:", embeddings.shape)

    # ---------- Dimensionality reduction & clustering ----------
    umap_5d = UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    emb_5d = umap_5d.fit_transform(embeddings)

    hdbscan_model = HDBSCAN(
        min_cluster_size=20,
        metric="euclidean",
        cluster_selection_method="eom"
    ).fit(emb_5d)
    clusters = hdbscan_model.labels_

    umap_2d = UMAP(n_components=2, min_dist=0.0, metric="cosine", random_state=42)
    emb_2d = umap_2d.fit_transform(embeddings)

    viz_df = pd.DataFrame(emb_2d, columns=["x", "y"])
    viz_df["title"] = titles.values
    viz_df["cluster"] = [str(c) for c in clusters]
    clusters_df = viz_df.loc[viz_df.cluster != "-1", :]
    outliers_df = viz_df.loc[viz_df.cluster == "-1", :]

    plt.figure(figsize=(8, 6))
    plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2, c="grey")
    plt.scatter(
        clusters_df.x,
        clusters_df.y,
        c=clusters_df.cluster.astype(int),
        alpha=0.6,
        s=2,
        cmap="tab20b",
    )
    plt.axis("off")
    plt.title("UMAP (2D) with HDBSCAN clusters")
    plt.tight_layout()
    plt.show()

    # ---------- BERTopic (fit on clean texts/embeddings) ----------
    mask = [isinstance(a, str) for a in abstracts]
    clean_abstracts = [a for a, m in zip(abstracts, mask) if m]
    clean_embeddings = embeddings[np.array(mask)]

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=20, metric="euclidean", cluster_selection_method="eom"),
        verbose=True,
    ).fit(clean_abstracts, clean_embeddings)

    # Inspect topics
    info = topic_model.get_topic_info()
    print(info.head(10))

    # ---------- Plotly visualizations ----------
    fig = topic_model.visualize_documents(
        titles.tolist(),
        reduced_embeddings=emb_2d,
        width=1200,
        hide_annotations=True,
    )
    fig.update_layout(font=dict(size=16))
    fig.show()

    topic_model.visualize_barchart().show()
    topic_model.visualize_heatmap(n_clusters=15).show()
    topic_model.visualize_hierarchy().show()

    # ---------- Save titles by topic ----------
    titles_by_category = defaultdict(list)
    for title, topic in zip(titles, topic_model.topics_):
        titles_by_category[topic].append(title)

    with open(titles_by_topic_txt, "w", encoding="utf-8") as f:
        for cat, papers in sorted(titles_by_category.items()):
            f.write(f"Topic {cat}:\n")
            for p in papers:
                f.write(f"- {p}\n")
            f.write("\n")
    print(f"✔  Written: {titles_by_topic_txt}")

    # ---------- Track differences after updating representation ----------
    original_topics = deepcopy(topic_model.topic_representations_)

    def topic_differences(model, original_topics, nr_topics=None):
        """Compare top words before/after updating representations."""
        if nr_topics is None:
            nr_topics = len([t for t in model.get_topic_info().Topic.unique() if t != -1])

        rows = []
        for topic in range(nr_topics):
            og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
            new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
            rows.append([topic, og_words, new_words])

        return pd.DataFrame(rows, columns=["Topic", "Original", "Updated"])

    # Update with KeyBERTInspired (no spaCy needed)
    representation_model = KeyBERTInspired()
    topic_model.update_topics(clean_abstracts, representation_model=representation_model)
    diff_df = topic_differences(topic_model, original_topics)
    print(diff_df.head(10))

    # ---------- Optional: auto-label topics with FLAN-T5 ----------
    prompt = """
    I have a topic that contains the following documents:
    [DOCUMENTS]

    The topic is described by the following keywords: [KEYWORDS]

    Based on the information above, extract a short topic label in the following format:
    topic: <short topic label>
    """

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    tg_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    hf_generator = pipeline(
        "text2text-generation",
        model=tg_model,
        tokenizer=tokenizer,
        max_length=64,
        do_sample=False,
    )

    tg_representation = TextGeneration(hf_generator, prompt=prompt)

    original_labels = topic_model.topic_labels_.copy()
    clean_topics = [t for t, m in zip(topic_model.topics_, mask) if m]

    topic_model.update_topics(
        clean_abstracts,
        clean_topics,
        representation_model=tg_representation,
    )

    for topic_id, old_label in original_labels.items():
        new_label = topic_model.topic_labels_.get(topic_id, "")
        if old_label != new_label:
            print(f"Topic {topic_id}: '{old_label}' → '{new_label}'")

    # ---------- Document datamap ----------
    freq_df = topic_model.get_topic_freq()
    valid_topics = sorted(freq_df.loc[freq_df.Topic != -1, "Topic"].tolist())
    topics_to_show = valid_topics[:17] if len(valid_topics) >= 1 else []

    fig = topic_model.visualize_document_datamap(
        titles.tolist(),
        topics=topics_to_show,
        reduced_embeddings=emb_2d,
        width=1200,
    )
    fig.show()

    # Re-label topics numerically and show datamap again
    num_labels = {topic: str(topic) for topic in freq_df.Topic}
    topic_model.set_topic_labels(num_labels)
    fig = topic_model.visualize_document_datamap(
        titles.tolist(),
        topics=topics_to_show,
        reduced_embeddings=emb_2d,
        width=1200,
        custom_labels=True,
        topic_prefix=False,
    )
    fig.show()

    print("✔ Done.")


if __name__ == "__main__":
    main()
