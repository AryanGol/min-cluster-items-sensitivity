# %% by Aryan Golzaryan - aryan.golzaryan@gmail.com

"""
HELP: Text Clustering and Topic Modeling

PURPOSE:
This script performs text clustering and topic modeling on a collection of research articles.

INPUT:
- A .csv file containing at least two columns:
    1. "title"    → the article title
    2. "abstract" → the article abstract
"""
# %% [0] Hardware Setup: Configure PyTorch to Use Intel Arc GPU (XPU Backend)
"""
HELP: Running PyTorch on Intel Arc GPU (XPU)

This cell configures your environment to use Intel's Arc GPU via the XPU backend.

INPUT REQUIREMENTS:
- A system with Intel Arc GPU hardware.
- Intel oneAPI drivers and runtime installed.
- A Python environment (e.g., C:\envs\arcxpu\python.exe) where PyTorch has been
  built/installed with XPU support.

PROCESS:
1. Checks if PyTorch detects an available XPU device.
2. If no XPU is found, raises a RuntimeError with setup guidance.
3. If XPU is available, sets device to GPU 0 explicitly.

OUTPUT:
- On success, prints: "✅ XPU ready: <GPU Name>".
- On failure, prints a clear error message with troubleshooting steps.

NOTE:
- Modify `torch.xpu.set_device(0)` if you want to select a different GPU index.
- If your system does not have Intel Arc GPU, you will need to adapt this cell
  for CUDA (NVIDIA GPUs) or fallback to CPU.
"""
import torch, sys
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    raise RuntimeError(
        f"XPU not available. Interpreter: {sys.executable}. "
        "Make sure Spyder is using C:\\envs\\arcxpu\\python.exe and drivers/oneAPI are installed."
    )
torch.xpu.set_device(0)  # pick GPU 0 explicitly
print("✅ XPU ready:", torch.xpu.get_device_name(0))
# %% [1] Configuration & Basic Imports

"""
HELP: Configure Input/Output Paths and Settings

PURPOSE:
Set up the working directory, specify the CSV file containing article metadata, 
and prepare the results folder.

INPUT REQUIRED:
- A CSV file with article data (must include at least "title" and "abstract" columns).
- Path to the folder (FOLDER) where the CSV file is located.
- CSV_NAME must match the name of your file (e.g., "pmc_metadata.csv").

PROCESS:
- Creates a "results" subfolder in the same directory to store output files.
- Suppresses tqdm-related warnings for cleaner logs.
#########################################################################
- Allows optional automatic topic labeling using the "FLAN-T5" model (downloads on first use).
#########################################################################

OUTPUT:
- A new folder called "results" inside your FOLDER directory, where Excel files, 
  visualizations, and other outputs will be saved.

USER ACTION:
1. Change the value of FOLDER to the path of your CSV file.
2. Change the value of CSV_NAME to the exact name of your CSV file.
#########################################################################
3. (Optional) Set AUTO_LABEL = False if you do not want automatic topic labeling.
#########################################################################
"""
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")  # quiet IProgress warning
FOLDER = r"C:\PhD Files\Organized\SPARK\July 7 - 2025 - Data Sent by Barbora\XML of PubMed Articles\MaterialsMethods\results"
CSV_NAME = "pmc_metadata.csv"
RESULTS_DIR = os.path.join(FOLDER, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
# Choose whether to auto-label topics with FLAN-T5 (downloads once)
AUTO_LABEL = True
# %% [2] Core Libraries (Independent of BERTopic)
"""
HELP: Import core libs and choose compute device

PURPOSE:
Load the minimum set of libraries needed for embedding, dimensionality reduction,
and clustering (without depending on BERTopic). Also decide which compute device
to use for inference.

INPUT:
- None (this cell only sets up imports and global config).

PROCESS:
- Import NumPy, pandas, matplotlib, PyTorch, SentenceTransformers, UMAP, HDBSCAN.
- Prefer Intel XPU (Arc GPU) if available; otherwise fall back to CPU.
- Use bfloat16 on XPU to reduce VRAM usage (keeps CPU in default dtype).

OUTPUT:
- Globals:
    DEVICE      → "xpu" or "cpu"
    MODEL_KWARGS→ dict passed when loading SentenceTransformer (dtype hint on XPU)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# Prefer Intel Arc (XPU) if available; otherwise CPU
DEVICE = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
# Use bfloat16 on XPU to save VRAM; no dtype override on CPU
MODEL_KWARGS = {"torch_dtype": torch.bfloat16} if DEVICE == "xpu" else {}

# %% [3] Safe-import BERTopic (temporarily block spaCy to avoid NumPy/Thinc issues)
"""
HELP: Import BERTopic without spaCy

PURPOSE
Some BERTopic extras try to import spaCy. On machines without a compatible spaCy/
Thinc/NumPy stack, that import can crash. This cell temporarily blocks spaCy
imports only while importing BERTopic, then restores normal import behavior.

INPUT
- None (environment-level concern only).

PROCESS
- Monkey-patch __import__ inside a context manager to raise if 'spacy' is requested.
- Import BERTopic and (optionally) its representation helpers.
- If AUTO_LABEL = True, enable BERTopic's TextGeneration representation so topics
  can receive short, human-readable labels (e.g., “Cancer Therapy”) after modeling.

OUTPUT
- BERTopic, KeyBERTInspired, (optionally) TextGeneration are ready to use.
- No spaCy features (e.g., lemmatization via spaCy) are used during import.
"""

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
"""
HELP: Utility helpers for results folders, 2D scatter plots, and topic diffs

FUNCTIONS
- ensure_dir(path)
    Create a directory (no error if it already exists).

- make_scatter_png(emb_2d, clusters, titles, out_png, *, figsize=(8,6), dpi=300,
                   sample=None, annotate=False, title="UMAP (2D) with HDBSCAN clusters")
    INPUTS:
        emb_2d   : array-like shape (n, 2) — 2D embeddings (e.g., UMAP outputs)
        clusters : array-like length n     — HDBSCAN labels; -1 is outlier
        titles   : sequence length n       — document titles (used only for future extensions)
        out_png  : str                     — file path to save the PNG
        sample   : Optional[int]           — if set, randomly subsample this many points for plotting
        annotate: bool                     — if True, place cluster IDs near centroids
    OUTPUT:
        Saves a PNG at out_png. Returns out_png.
#########################################################################
- topic_differences(model, original_topics, *, top_n=5, include_outlier=False, nr_topics=None)
#########################################################################
    INPUTS:
        model           : BERTopic model (already fit)
        original_topics : dict[int, list[(word, score)]] — snapshot from before update
        top_n           : how many top words to compare
        include_outlier : include topic -1 if True
        #########################################################################
        nr_topics       : optional cap on how many topics to compare
        #########################################################################
    OUTPUT:
        pandas.DataFrame with columns ["Topic", "Original", "Updated"]
"""

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
    plt.savefig(out_png, dpi=300)
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

# %% [5] Load & clean CSV (writes a cleaned copy next to the original)
"""
HELP: Load the metadata CSV and produce a cleaned version

PURPOSE
- Read the input CSV, normalize whitespace, remove rows with empty abstracts, and (optionally) drop duplicates.

INPUT
- FOLDER / CSV_NAME → CSV must contain article titles and abstracts (case-insensitive).

PROCESS
1) Read CSV (tries UTF-8, falls back to Latin-1).
2) Resolve Title/Abstract columns case-insensitively (supports common variants).
3) Normalize whitespace and strip.
4) Remove rows with empty abstracts.
5) Drop exact duplicates by (Title, Abstract).
6) Save <CSV_NAME>_cleaned.csv next to the original.

OUTPUT
- Cleaned CSV and basic stats printed to console.
"""

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

cleaned_out = os.path.join(FOLDER, "cleaned.csv")
df_clean.to_csv(cleaned_out, index=False, encoding="utf-8")
print(f"✔ Saved cleaned CSV to: {cleaned_out}")

titles = df_clean[title_col].fillna("").astype(str)
abstracts = df_clean[abstract_col].fillna("").astype(str).tolist()
# %% [6] Top 10 most frequent words per Title+Abstract -> Excel
"""
HELP: Top 10 words per document (Title+Abstract)

INPUTS (from previous cell [5], in memory):
- df_clean      : cleaned DataFrame loaded from FOLDER/CSV_NAME
- titles        : Series of titles from df_clean
- abstract_col  : name of the abstract column in df_clean

PROCESS:
- Tokenize Title+Abstract, remove exact stopwords and pure digits, keep abbreviations (e.g., 18F, 68Ga).
- Count word frequencies per document.
- Export the top 10 words + counts for each document.

OUTPUT:
- Excel: RESULTS_DIR / "top10_words_per_title_abstract.xlsx"
"""

import re
from collections import Counter

# Conservative English stopword list (no downloads needed)
STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
    "be","because","been","before","being","below","between","both","but","by",
    "can","can't","cannot","could","couldn't",
    "did","didn't","do","does","doesn't","doing","don't","down","during",
    "each",
    "few","for","from","further",
    "had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's",
    "hers","herself","him","himself","his","how","how's",
    "i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
    "let's",
    "me","more","most","mustn't","my","myself",
    "no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
    "same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such",
    "than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they",
    "they'd","they'll","they're","they've","this","those","through","to","too",
    "under","until","up",
    "very",
    "was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where",
    "where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't",
    "you","you'd","you'll","you're","you've","your","yours","yourself","yourselves",
    "background","methods","method","results","conclusions","conclusion","objective","objectives","aim","aims",
    "study","studies","based","using","use","used",
    "paper","however","therefore","thus","may","also","could","within","among"
}

def tokenize(text: str):
    text = str(text)

    raw = re.findall(r"[A-Za-z0-9]+", text)
    toks = []
    for t in raw:
        lt = t.lower()
        if lt.isdigit():        # drop pure numbers like "2024", "123"
            continue
        if lt in STOPWORDS:     # drop ONLY exact stopwords
            continue
        toks.append(lt)
    return toks

# Combine Title + Abstract for each row
combos = (titles.astype(str) + " " + df_clean[abstract_col].astype(str)).str.strip()

rows = []
titles_list = titles.tolist()
combos_list = combos.tolist()
orig_index = list(df_clean.index)

for i in range(len(combos_list)):
    words = tokenize(combos_list[i])
    freq = Counter(words).most_common(10)
    # pad to exactly 10
    while len(freq) < 10:
        freq.append(("", 0))
    row = {
        "DocIndex": orig_index[i],                # original row index from df_clean
        "Title": titles_list[i],
    }
    for j, (w, c) in enumerate(freq, start=1):
        row[f"Word_{j}"] = w
        row[f"Count_{j}"] = c
    rows.append(row)

out_df = pd.DataFrame(rows, columns=[
    "DocIndex","Title",
    "Word_1","Count_1","Word_2","Count_2","Word_3","Count_3","Word_4","Count_4","Word_5","Count_5",
    "Word_6","Count_6","Word_7","Count_7","Word_8","Count_8","Word_9","Count_9","Word_10","Count_10"
])

ensure_dir(RESULTS_DIR)
out_xlsx = os.path.join(RESULTS_DIR, "top10_words_per_title_abstract.xlsx")
out_df.to_excel(out_xlsx, index=False)
print(f"✅ Saved Excel to: {out_xlsx}")

# %% [7] Save Title/Abstract BEFORE and AFTER word removal -> two CSVs
"""
HELP: Export raw vs. cleaned Title/Abstract strings

PURPOSE:
Create two exports so you can compare the original text with the cleaned text
after token removal.

REQUIRES (from previous cells):
- df_clean      : cleaned DataFrame (from cell [5])
- titles        : Series of titles aligned with df_clean
- abstract_col  : name of the abstract column in df_clean
- tokenize()    : tokenizer from cell [6] (lowercases, splits on [A-Za-z0-9]+,
                  drops pure digits, removes exact STOPWORDS, keeps alphanumeric
                  abbreviations like 18F, 68Ga)
- ensure_dir(), RESULTS_DIR

PROCESS:
1) BEFORE export:
   - Use raw Title and Abstract from df_clean (no token removal).
   - Add a combined "Title_Abstract" column.
2) AFTER export:
   - Build cleaned strings by joining tokens returned by tokenize() for each field.
   - Add a combined "Title_Abstract_Clean" column.

OUTPUT:
- RESULTS_DIR / "titles_abstracts_before.csv"
    Columns: DocIndex, Title, Abstract, Title_Abstract
- RESULTS_DIR / "titles_abstracts_after.csv"
    Columns: DocIndex, Title_Clean, Abstract_Clean, Title_Abstract_Clean

NOTES:
- Row order and DocIndex match df_clean.index.
- Files are saved with UTF-8 BOM (utf-8-sig) for Excel compatibility.
- Cleaned text is lowercased and punctuation is removed by design.
"""
import os
import pandas as pd

ensure_dir(RESULTS_DIR)

# BEFORE: raw strings from df_clean (no token removal)
before_df = pd.DataFrame({
    "DocIndex": df_clean.index,
    "Title": titles.astype(str).values,
    "Abstract": df_clean[abstract_col].astype(str).values,
})
before_df["Title_Abstract"] = (before_df["Title"] + " " + before_df["Abstract"]).str.strip()
before_csv = os.path.join(RESULTS_DIR, "titles_abstracts_before.csv")
before_df.to_csv(before_csv, index=False, encoding="utf-8-sig")

# AFTER: cleaned strings built from tokens returned by your `tokenize` (stopwords & pure numbers removed)
def clean_text(s: str) -> str:
    return " ".join(tokenize(s))

after_df = pd.DataFrame({
    "DocIndex": df_clean.index,
    "Title_Clean": [clean_text(t) for t in titles.astype(str).values],
    "Abstract_Clean": [clean_text(a) for a in df_clean[abstract_col].astype(str).values],
})
after_df["Title_Abstract_Clean"] = (after_df["Title_Clean"] + " " + after_df["Abstract_Clean"]).str.strip()
after_csv = os.path.join(RESULTS_DIR, "titles_abstracts_after.csv")
after_df.to_csv(after_csv, index=False, encoding="utf-8-sig")

print(f"✅ Saved BEFORE CSV to: {before_csv}")
print(f"✅ Saved AFTER  CSV to: {after_csv}")

# %% [8] Word counts (Title+Abstract) BEFORE vs AFTER word removal -> Excel
"""
HELP: Count words before/after token filtering and export

PURPOSE
Compute, for each document, the number of tokens in Title+Abstract
(a) BEFORE stopword removal and (b) AFTER stopword removal, then save to Excel.

REQUIRES (from previous cells)
- df_clean, titles, abstract_col, RESULTS_DIR, ensure_dir
- tokenize() and STOPWORDS from cell [6] (alnum split, drop pure digits, remove exact stopwords)

PROCESS
1) Build Title+Abstract per row.
2) BEFORE count: tokens from alphanumeric splitting, lowercased, pure digits removed.
3) AFTER  count: tokens from tokenize() (same as BEFORE + stopword removal).
4) Compute Pct_Removed = 100 * (1 - AFTER/BEFORE).
5) Append a TOTAL row.
6) Save to Excel (fallback to CSV if no Excel engine installed).

OUTPUT
- RESULTS_DIR / "word_counts_before_after.xlsx" (or .csv fallback)
"""

import re, os, importlib.util
import pandas as pd

# BEFORE: same prefilter basis you used earlier (keep alnum, drop pure digits, lowercase)
def _tokens_prefilter(text: str):
    raw = re.findall(r"[A-Za-z0-9]+", str(text))
    return [t.lower() for t in raw if not t.isdigit()]

# Build combined text per row
combos = (titles.astype(str) + " " + df_clean[abstract_col].astype(str)).str.strip()

rows = []
for idx, title, combo in zip(df_clean.index, titles.astype(str).tolist(), combos.tolist()):
    before_tokens = _tokens_prefilter(combo)   # BEFORE stopword removal
    after_tokens  = tokenize(combo)            # AFTER stopword removal (your function)
    b, a = len(before_tokens), len(after_tokens)
    pct = (1 - (a / b)) * 100 if b else 0.0
    rows.append({
        "DocIndex": idx,
        "Title": title,
        "Words_Before": b,
        "Words_After": a,
        "Pct_Removed": round(pct, 2),
    })

out_df = pd.DataFrame(rows, columns=["DocIndex","Title","Words_Before","Words_After","Pct_Removed"])

# TOTAL row
tot_b = int(out_df["Words_Before"].sum())
tot_a = int(out_df["Words_After"].sum())
tot_p = round((1 - (tot_a / tot_b)) * 100, 2) if tot_b else 0.0
out_df = pd.concat([
    out_df,
    pd.DataFrame([{
        "DocIndex": "",
        "Title": "__TOTAL__",
        "Words_Before": tot_b,
        "Words_After":  tot_a,
        "Pct_Removed":  tot_p,
    }])
], ignore_index=True)

# Save (Excel if possible; else CSV)
def _has(pkg): 
    return importlib.util.find_spec(pkg) is not None

ensure_dir(RESULTS_DIR)
base = os.path.join(RESULTS_DIR, "word_counts_before_after")
if _has("openpyxl") or _has("xlsxwriter"):
    engine = "openpyxl" if _has("openpyxl") else "xlsxwriter"
    out_path = f"{base}.xlsx"
    out_df.to_excel(out_path, index=False, engine=engine)
    print(f"✅ Saved Excel with {engine} to: {out_path}")
else:
    out_path = f"{base}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"ℹ️ Saved CSV (install openpyxl or xlsxwriter for .xlsx): {out_path}")

# %% [Before 9 - check data]
# %% [9a] Token counts for CLEAN Title+Abstract (only title_abstract_clean)
"""
HELP: Count words & transformer tokens for CLEAN Title+Abstract

PURPOSE
Compute, for each document, the number of WORDS and TRANSFORMER TOKENS in the
cleaned Title+Abstract (after your tokenize(): lowercase, alnum split, drop pure
digits, remove exact STOPWORDS). Use this to check truncation risk.

KEY POINTS
- SentenceTransformer tokenizes into subword tokens and truncates to a max length.
- Common practice: use ~384 tokens as working max; the hard cap is 512.
- This cell flags docs that exceed 384 and 512 tokens (including special tokens).

REQUIRES
- tokenize(), STOPWORDS (from your cell [6])
- df_clean, titles, abstract_col, RESULTS_DIR, ensure_dir

OUTPUT
- RESULTS_DIR / token_counts_title_abstract_clean.xlsx (or .csv)
  Columns: DocIndex, Title, Words_in_Clean_Input, Tokens_in_Clean_Input,
           Exceeds_384, Exceeds_512
"""

import os, importlib.util
import pandas as pd
from transformers import AutoTokenizer

def _has(pkg): 
    return importlib.util.find_spec(pkg) is not None

ensure_dir(RESULTS_DIR)

# 1) Build CLEAN Title+Abstract strings using your tokenizer
def _clean_text(s: str) -> str:
    return " ".join(tokenize(s))  # uses your cell [6] tokenize()

title_clean = titles.astype(str).apply(_clean_text)
abs_clean   = df_clean[abstract_col].astype(str).apply(_clean_text)
texts_clean = (title_clean + " " + abs_clean).str.strip().tolist()

# 2) Word counts (on the exact cleaned strings you'll embed)
def _word_count(s: str) -> int:
    return len([w for w in s.split() if w])

# 3) Token counts using the embedding model's tokenizer
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2", use_fast=True)

rows = []
for doc_idx, title, text in zip(df_clean.index.tolist(), titles.astype(str).tolist(), texts_clean):
    # Count words
    wcount = _word_count(text)
    # Count tokens INCLUDING special tokens, with NO truncation
    enc = tok(text, add_special_tokens=True, truncation=False)
    tcount = len(enc["input_ids"])

    rows.append({
        "DocIndex": doc_idx,
        "Title": title,
        "Words_in_Clean_Input": wcount,
        "Tokens_in_Clean_Input": tcount,
        "Exceeds_384": tcount > 384,
        "Exceeds_512": tcount > 512,
    })

df_tokens = pd.DataFrame(rows, columns=[
    "DocIndex","Title","Words_in_Clean_Input","Tokens_in_Clean_Input","Exceeds_384","Exceeds_512"
])

# 4) Save to Excel (fallback to CSV)
out_base = os.path.join(RESULTS_DIR, "token_counts_title_abstract_clean")
engine = "openpyxl" if _has("openpyxl") else ("xlsxwriter" if _has("xlsxwriter") else None)

if engine:
    with pd.ExcelWriter(f"{out_base}.xlsx", engine=engine) as w:
        df_tokens.to_excel(w, index=False, sheet_name="token_counts")
        meta = pd.DataFrame([
            ["model", "all-mpnet-base-v2"],
            ["input", "title_abstract_clean"],
            ["notes", "Token counts include special tokens; texts exceeding 384/512 may be truncated."]
        ], columns=["key","value"])
        meta.to_excel(w, index=False, sheet_name="meta")
    print(f"✅ Saved Excel to: {out_base}.xlsx")
else:
    df_tokens.to_csv(f"{out_base}.csv", index=False)
    print(f"ℹ️ Saved CSV (install openpyxl/xlsxwriter for .xlsx): {out_base}.csv")


# %% [9] Embed CLEAN Title+Abstract (max 512 tokens) → NPY + Excel
"""
HELP: Embed cleaned Title+Abstract with 512-token cap

PURPOSE
Use SentenceTransformer ("all-mpnet-base-v2") to embed the CLEAN text
(Title + Abstract after your tokenize()) with a max input of 512 tokens.

REQUIRES
- tokenize(), STOPWORDS
- df_clean, titles, abstract_col
- DEVICE, MODEL_KWARGS, RESULTS_DIR, ensure_dir
"""

import os, importlib.util
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def _has(pkg): 
    return importlib.util.find_spec(pkg) is not None

def _clean_text(s: str) -> str:
    # your tokenizer: lowercases, splits on [A-Za-z0-9]+, drops pure digits, removes exact STOPWORDS
    return " ".join(tokenize(s))

# Build CLEAN Title+Abstract strings
titles_list    = titles.astype(str).tolist()
abstracts_list = df_clean[abstract_col].astype(str).tolist()
texts_for_embeddings = [
    (_clean_text(t) + " " + _clean_text(a)).strip()
    for t, a in zip(titles_list, abstracts_list)
]

print(f"Using device for embeddings: {DEVICE} | input='title_abstract_clean' | n={len(texts_for_embeddings)}")

# Model (force 512-token max)
embedding_model = SentenceTransformer(
    "all-mpnet-base-v2",
    device=DEVICE,
    model_kwargs=MODEL_KWARGS
)
embedding_model.max_seq_length = 512
print("Max sequence length set to:", embedding_model.max_seq_length)

# Encode
embeddings = embedding_model.encode(
    texts_for_embeddings,
    show_progress_bar=True,
    convert_to_numpy=True,
    batch_size=32,
    normalize_embeddings=True  # cosine-ready
)
print("Embeddings shape:", embeddings.shape)

# Save NPY
ensure_dir(RESULTS_DIR)
emb_npy_path = os.path.join(RESULTS_DIR, "embeddings_title_abstract_clean_max512.npy")
np.save(emb_npy_path, embeddings)
print("✅ Saved NPY to:", emb_npy_path)

# Save Excel (DocIndex, Title, d0..d767) + meta
dim_cols = [f"d{i}" for i in range(embeddings.shape[1])]
emb_df = pd.DataFrame(embeddings, columns=dim_cols)
emb_df.insert(0, "Title", titles_list)
emb_df.insert(0, "DocIndex", df_clean.index.tolist())

excel_engine = "openpyxl" if _has("openpyxl") else ("xlsxwriter" if _has("xlsxwriter") else None)
excel_path = os.path.join(RESULTS_DIR, "embeddings_title_abstract_clean_max512.xlsx")

if excel_engine:
    with pd.ExcelWriter(excel_path, engine=excel_engine) as writer:
        emb_df.to_excel(writer, index=False, sheet_name="embeddings")
        meta = pd.DataFrame([
            ["model", "all-mpnet-base-v2"],
            ["device", DEVICE],
            ["input", "title_abstract_clean"],
            ["normalize_embeddings", True],
            ["n_docs", len(texts_for_embeddings)],
            ["dim", embeddings.shape[1]],
            ["max_seq_length", int(embedding_model.max_seq_length)],
            ["created", pd.Timestamp.now().isoformat()],
        ], columns=["key", "value"])
        meta.to_excel(writer, index=False, sheet_name="meta")
    print(f"✅ Saved Excel to: {excel_path}")
else:
    csv_path = os.path.join(RESULTS_DIR, "embeddings_title_abstract_clean_max512.csv")
    emb_df.to_csv(csv_path, index=False)
    print(f"ℹ️ Saved CSV (install 'openpyxl' or 'xlsxwriter' for .xlsx): {csv_path}")

# %% [9+] Token "chessboard": rows=docs, cols=512 positions, value=TOKEN STRING (coded)
"""
HELP: Visualize embeddings as a chessboard heatmap

WHAT THIS DOES
- Loads results/embeddings_title_abstract_clean_max512.xlsx (sheet "embeddings")
- Extracts columns d0..d767 (768 dims)
- Plots a chessboard-style heatmap (no smoothing) with:
    y-axis = document index (one row per paper)
    x-axis = embedding dimension (0..767)
    cell value = embedding value (small floats)
- Adds a colorbar.

OUTPUT
- results/embeddings_heatmap_docs_x_768.png
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Excel (fallback to CSV if needed)
excel_path = os.path.join(RESULTS_DIR, "embeddings_title_abstract_clean_max512.xlsx")
csv_path   = os.path.join(RESULTS_DIR, "embeddings_title_abstract_clean_max512.csv")
if os.path.isfile(excel_path):
    df_emb = pd.read_excel(excel_path, sheet_name="embeddings")
elif os.path.isfile(csv_path):
    df_emb = pd.read_csv(csv_path)
else:
    raise FileNotFoundError(
        "Embedding file not found. Expected one of:\n"
        f" - {excel_path}\n"
        f" - {csv_path}"
    )

# Get embedding columns d0..d767 in numeric order
dim_cols = sorted([c for c in df_emb.columns if re.fullmatch(r"d\d+", c)],
                  key=lambda x: int(x[1:]))
if len(dim_cols) != 768:
    print(f"⚠️ Found {len(dim_cols)} embedding columns (expected 768). Proceeding with found columns.")

mat = df_emb[dim_cols].to_numpy(dtype=float)   # shape: (N_docs, N_dims)

# Plot heatmap (no smoothing)
plt.figure(figsize=(12, max(6, min(24, mat.shape[0] * 0.01))))  # scale height with #docs
im = plt.imshow(mat, aspect="auto", interpolation="nearest", origin="upper")

# X ticks every 64 dims for readability
xticks = np.arange(0, len(dim_cols), 64)
plt.xticks(xticks, xticks)
plt.xlabel("Embedding dimension (0 … {})".format(len(dim_cols)-1))
plt.ylabel("Document row (0 … {})".format(mat.shape[0]-1))

cbar = plt.colorbar(im)
cbar.set_label("Embedding value")

plt.title("Embeddings heatmap (rows: papers, cols: 768 dims)")
plt.tight_layout()

out_png = os.path.join(RESULTS_DIR, "embeddings_heatmap_docs_x_768.png")
plt.savefig(out_png, dpi=300)
plt.close()
print(f"✅ Saved heatmap to: {out_png}")

# %% [10] Compute UMAP reductions ONCE (CPU)
umap_5d = UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42)
emb_5d = umap_5d.fit_transform(embeddings)

umap_2d = UMAP(n_components=2, min_dist=0.0, metric="cosine", random_state=42)
emb_2d = umap_2d.fit_transform(embeddings)

print("UMAP 5D:", emb_5d.shape, "UMAP 2D:", emb_2d.shape)

# %% [10+] Visualize UMAP: 5D chessboard heatmap + 2D scatter
"""
HELP: Plot UMAP results
- 5D: chessboard heatmap (rows = docs, cols = 5 UMAP dims, cell = value)
- 2D: scatter (x,y)

REQUIRES
- emb_5d, emb_2d (from cell [10])
- RESULTS_DIR, ensure_dir
- df_clean, titles (for CSV exports)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure previous step ran
if "emb_5d" not in globals() or "emb_2d" not in globals():
    raise RuntimeError("Run cell [10] first to compute emb_5d and emb_2d.")

ensure_dir(RESULTS_DIR)

# ---------------- 5D chessboard heatmap ----------------
mat5 = np.asarray(emb_5d, dtype=float)
if mat5.ndim != 2 or mat5.shape[1] != 5:
    raise ValueError(f"emb_5d must be (n_docs, 5); got {mat5.shape}")
mat5 = np.nan_to_num(mat5, copy=False)

plt.figure(figsize=(6, max(6, min(24, mat5.shape[0] * 0.01))))  # scale height with #docs
im = plt.imshow(mat5, aspect="auto", interpolation="nearest", origin="upper")
plt.xticks(range(5), [f"u{i}" for i in range(5)])
plt.xlabel("UMAP dimension")
plt.ylabel("Document row")
cbar = plt.colorbar(im)
cbar.set_label("UMAP value")
plt.title("UMAP 5D — chessboard heatmap")
plt.tight_layout()
heat_png = os.path.join(RESULTS_DIR, "umap_5d_heatmap.png")
plt.savefig(heat_png, dpi=300)
plt.close()
print(f"✅ Saved 5D heatmap: {heat_png}")

# Save a tidy CSV for 5D (DocIndex, Title, u0..u4)
u5_df = pd.DataFrame({
    "DocIndex": df_clean.index.tolist(),
    "Title": titles.astype(str).tolist(),
    "u0": mat5[:, 0], "u1": mat5[:, 1], "u2": mat5[:, 2], "u3": mat5[:, 3], "u4": mat5[:, 4],
})
u5_csv = os.path.join(RESULTS_DIR, "umap_5d.csv")
u5_df.to_csv(u5_csv, index=False, encoding="utf-8-sig")
print(f"📄 Saved 5D UMAP CSV: {u5_csv}")

# ---------------- 2D scatter ----------------
xy = np.asarray(emb_2d, dtype=float)
if xy.ndim != 2 or xy.shape[1] != 2:
    raise ValueError(f"emb_2d must be (n_docs, 2); got {xy.shape}")

plt.figure(figsize=(8, 6))
plt.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.2)
plt.title("UMAP 2D (cosine)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
scat_png = os.path.join(RESULTS_DIR, "umap_2d_scatter.png")
plt.savefig(scat_png, dpi=300)
plt.close()
print(f"🗺️ Saved 2D scatter: {scat_png}")

# Save a tidy CSV for 2D (DocIndex, Title, x, y)
u2_df = pd.DataFrame({
    "DocIndex": df_clean.index.tolist(),
    "Title": titles.astype(str).tolist(),
    "x": xy[:, 0], "y": xy[:, 1],
})
u2_csv = os.path.join(RESULTS_DIR, "umap_2d.csv")
u2_df.to_csv(u2_csv, index=False, encoding="utf-8-sig")
print(f"📄 Saved 2D UMAP CSV: {u2_csv}")

# %% [11] RUN ONE CLUSTER SIZE (clean Title+Abstract ONLY) — HDBSCAN + BERTopic
"""
HELP: Run clustering/topic modeling with CLEAN Title+Abstract

WHAT CHANGED
- Uses cleaned Title+Abstract texts (after your tokenize()) for ALL BERTopic calls.
- Ensures embeddings match those cleaned texts (loads from file if needed).
- Keeps UMAP/HDBSCAN settings identical to your previous code.

REQUIRES
- tokenize(), STOPWORDS
- df_clean, titles, abstract_col
- embeddings (preferably the clean Title+Abstract embeddings you saved earlier)
- emb_5d, emb_2d (UMAP of the same embeddings)
- RESULTS_DIR, ensure_dir
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from umap import UMAP
from hdbscan import HDBSCAN

# ------------ 0) Build the CLEAN Title+Abstract texts ------------
def _clean_text(s: str) -> str:
    return " ".join(tokenize(s))  # your cell [6] tokenizer

if "after_df" in globals() and "Title_Abstract_Clean" in after_df.columns:
    doc_texts = after_df["Title_Abstract_Clean"].astype(str).tolist()
else:
    # Rebuild on the fly if after_df isn't available
    t_clean = titles.astype(str).apply(_clean_text)
    a_clean = df_clean[abstract_col].astype(str).apply(_clean_text)
    doc_texts = (t_clean + " " + a_clean).str.strip().tolist()

# ------------ 1) Make sure embeddings line up with cleaned texts ------------
# Prefer the already-in-memory 'embeddings' if it matches; otherwise load the saved clean embeddings.
if "embeddings" not in globals() or getattr(embeddings, "shape", (0,))[0] != len(doc_texts):
    emb_path = os.path.join(RESULTS_DIR, "embeddings_title_abstract_clean_max512.npy")
    if not os.path.isfile(emb_path):
        raise FileNotFoundError(
            "Clean embeddings not found in memory and not on disk.\n"
            f"Expected: {emb_path}\nRun the embedding cell for title_abstract_clean first."
        )
    embeddings = np.load(emb_path)

# Basic sanity check
if embeddings.shape[0] != len(doc_texts):
    raise ValueError(
        f"Embeddings/doc_texts length mismatch: {embeddings.shape[0]} vs {len(doc_texts)}.\n"
        "Ensure embeddings were computed from title_abstract_clean."
    )

# ------------ 2) Choose cluster size & output dir ------------
MIN_CS = 5  # <-- change to 5, 10, 20, 30, 50, 75, or 100
outdir = os.path.join(RESULTS_DIR, f"mincs_{MIN_CS:03d}")
ensure_dir(outdir)
print(f"\n==== Running min_cluster_size={MIN_CS} with CLEAN Title+Abstract ====")

# ------------ 3) HDBSCAN labels for scatter (fit on 5D UMAP) ------------
labels = HDBSCAN(
    min_cluster_size=MIN_CS,
    metric="euclidean",
    cluster_selection_method="eom"
).fit_predict(emb_5d)

# ------------ 4) Save scatter plot (2D UMAP) ------------
scatter_path = os.path.join(outdir, f"scatter_umap2d_mincs_{MIN_CS:03d}.png")
make_scatter_png(emb_2d, labels, titles, scatter_path)

# ------------ 5) Fit BERTopic on CLEAN texts + precomputed embeddings ------------
topic_model = BERTopic(
    embedding_model=embedding_model,  # stays on XPU/CPU as set above
    umap_model=UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42),
    hdbscan_model=HDBSCAN(min_cluster_size=MIN_CS, metric="euclidean", cluster_selection_method="eom"),
    verbose=True,
)
topic_model = topic_model.fit(doc_texts, embeddings)

# ------------ 6) Save topic info + titles by topic ------------
info = topic_model.get_topic_info()
info.to_csv(os.path.join(outdir, "topic_info.csv"), index=False)

titles_by_category = defaultdict(list)
for title, topic in zip(titles, topic_model.topics_):
    titles_by_category[topic].append(str(title))

with open(os.path.join(outdir, "titles_by_topic.txt"), "w", encoding="utf-8") as f:
    for cat, papers in sorted(titles_by_category.items()):
        f.write(f"Topic {cat}:\n")
        for p in papers:
            f.write(f"- {p}\n")
        f.write("\n")

# ------------ 7) Plotly visualizations (documents, barchart, heatmap, hierarchy) ------------
# Use CLEAN doc_texts for the documents view; titles are shown as labels/hover via 'custom_labels' if desired
docs_fig = topic_model.visualize_documents(
    doc_texts, reduced_embeddings=emb_2d, width=1200, hide_annotations=True
)
docs_fig.update_layout(font=dict(size=16))
docs_fig.write_html(os.path.join(outdir, "documents.html"))

topic_model.visualize_barchart().write_html(os.path.join(outdir, "barchart.html"))
topic_model.visualize_heatmap(n_clusters=15).write_html(os.path.join(outdir, "heatmap.html"))
topic_model.visualize_hierarchy().write_html(os.path.join(outdir, "hierarchy.html"))

# ------------ 8) Update representations with KeyBERTInspired (on CLEAN texts) ------------
original_topics = __import__("copy").deepcopy(topic_model.topic_representations_)
kb_model = KeyBERTInspired()  # uses topic_model.embedding_model internally
topic_model.update_topics(doc_texts, representation_model=kb_model)

diffs = topic_differences(topic_model, original_topics)
diffs.to_csv(os.path.join(outdir, "diffs_after_keybert.csv"), index=False)

# ------------ 9) Optional: Auto-label with FLAN-T5 (on CLEAN texts) ------------
if AUTO_LABEL:
    try:
        print("Auto-labeling with FLAN-T5-small…")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        tg_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(DEVICE)
        hf_pipe = pipeline(
            "text2text-generation",
            model=tg_model,
            tokenizer=tokenizer,
            max_length=64,
            do_sample=False
        )
        prompt = (
            "I have a topic that contains the following documents: [DOCUMENTS]\n"
            "The topic is described by the following keywords: [KEYWORDS]\n"
            "Based on the information above, extract a short topic label in the following format:\n"
            "topic: <short topic label>"
        )
        tg_representation = TextGeneration(hf_pipe, prompt=prompt)

        original_labels = topic_model.topic_labels_.copy()
        topic_model.update_topics(doc_texts, topic_model.topics_, representation_model=tg_representation)

        for topic_id, old_label in original_labels.items():
            new_label = topic_model.topic_labels_.get(topic_id, "")
            if old_label != new_label:
                print(f"Topic {topic_id}: '{old_label}' → '{new_label}'")
    except Exception as e:
        print(f"(Skipping auto-label: {e})")

# ------------ 10) Datamap visualizations (if installed) ------------
try:
    freq_df = topic_model.get_topic_freq()
    valid_topics = sorted(freq_df.loc[freq_df.Topic != -1, "Topic"].tolist())
    topics_to_show = valid_topics[:17] if valid_topics else []

    topic_model.visualize_document_datamap(
        doc_texts, topics=topics_to_show, reduced_embeddings=emb_2d, width=1200
    ).write_html(os.path.join(outdir, "datamap.html"))

    num_labels = {topic: str(topic) for topic in freq_df.Topic}
    topic_model.set_topic_labels(num_labels)
    topic_model.visualize_document_datamap(
        doc_texts, topics=topics_to_show, reduced_embeddings=emb_2d, width=1200,
        custom_labels=True, topic_prefix=False
    ).write_html(os.path.join(outdir, "datamap_numeric.html"))
except Exception as e:
    print(f"(Skipping datamap: {e})")

print(f"✔ Done for min_cluster_size={MIN_CS}. Results → {outdir}")




# %% [12] OPTIONAL: Run ALL sizes automatically (clean Title+Abstract only)
"""
HELP: Batch BERTopic runs over multiple min_cluster_size values
- Uses CLEAN Title+Abstract texts (after your tokenize()) for ALL BERTopic calls.
- Uses the matching embeddings (title_abstract_clean_max512) for fit/update/visualizations.
- Saves per-run outputs under results/mincs_XXX/.

REQUIRES
- tokenize(), STOPWORDS
- df_clean, titles, abstract_col
- embeddings, emb_5d, emb_2d computed FROM the same cleaned texts (or will be loaded)
- RESULTS_DIR, ensure_dir
- BERTopic, KeyBERTInspired, TextGeneration (optional), UMAP, HDBSCAN
- AUTO_LABEL (bool), DEVICE set earlier
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from umap import UMAP
from hdbscan import HDBSCAN

# ---------- 0) Build CLEAN Title+Abstract texts ----------
def _clean_text(s: str) -> str:
    return " ".join(tokenize(s))  # your tokenizer from cell [6]

if "after_df" in globals() and "Title_Abstract_Clean" in after_df.columns:
    doc_texts = after_df["Title_Abstract_Clean"].astype(str).tolist()
else:
    t_clean = titles.astype(str).apply(_clean_text)
    a_clean = df_clean[abstract_col].astype(str).apply(_clean_text)
    doc_texts = (t_clean + " " + a_clean).str.strip().tolist()

N_docs = len(doc_texts)

# ---------- 1) Ensure embeddings/UMAPs match the CLEAN texts ----------
# Prefer in-memory 'embeddings'; if missing or mismatched, load the clean file
clean_emb_path = os.path.join(RESULTS_DIR, "embeddings_title_abstract_clean_max512.npy")
if "embeddings" not in globals() or getattr(embeddings, "shape", (0,))[0] != N_docs:
    if not os.path.isfile(clean_emb_path):
        raise FileNotFoundError(
            "Clean embeddings not found in memory and not on disk.\n"
            f"Expected: {clean_emb_path}\nRun the clean embedding cell first."
        )
    embeddings = np.load(clean_emb_path)

if embeddings.shape[0] != N_docs:
    raise ValueError(f"Embeddings/doc_texts mismatch: {embeddings.shape[0]} vs {N_docs}. "
                     "Recompute embeddings for title_abstract_clean.")

# Ensure UMAP reductions exist and align; if not, compute quickly on CPU
if "emb_5d" not in globals() or getattr(emb_5d, "shape", (0, 0))[0] != N_docs:
    umap_5d = UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    emb_5d = umap_5d.fit_transform(embeddings)

if "emb_2d" not in globals() or getattr(emb_2d, "shape", (0, 0))[0] != N_docs:
    umap_2d = UMAP(n_components=2, min_dist=0.0, metric="cosine", random_state=42)
    emb_2d = umap_2d.fit_transform(embeddings)

# ---------- 2) Optional: prepare FLAN-T5 once ----------
hf_pipe = None
if AUTO_LABEL:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    tokenizer.model_max_length = 512  # keep prompts tidy
    tg_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(DEVICE)
    hf_pipe = pipeline(
        "text2text-generation",
        model=tg_model,
        tokenizer=tokenizer,
        max_length=64,
        do_sample=False,
        truncation=True,
    )

# ---------- 3) Cluster sizes to run ----------
CLUSTER_SIZES = [5, 10, 15, 30, 50]

for MIN_CS in CLUSTER_SIZES:
    outdir = os.path.join(RESULTS_DIR, f"mincs_{MIN_CS:03d}")
    ensure_dir(outdir)
    print(f"\n==== Running min_cluster_size={MIN_CS} (CLEAN Title+Abstract) ====")

    # 3.1 HDBSCAN labels for static scatter (fit on 5D UMAP)
    labels = HDBSCAN(
        min_cluster_size=MIN_CS,
        metric="euclidean",
        cluster_selection_method="eom"
    ).fit_predict(emb_5d)

    scatter_path = os.path.join(outdir, f"scatter_umap2d_mincs_{MIN_CS:03d}.png")
    make_scatter_png(emb_2d, labels, titles, scatter_path)

    # 3.2 Fit BERTopic with precomputed embeddings on CLEAN texts
    topic_model = BERTopic(
        embedding_model=embedding_model,  # XPU/CPU as configured earlier
        umap_model=UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=MIN_CS, metric="euclidean", cluster_selection_method="eom"),
        verbose=True,
    ).fit(doc_texts, embeddings)

    # Save model + doc assignments
    topic_model.save(os.path.join(outdir, "bertopic_model"))
    pd.DataFrame({
        "DocIndex": df_clean.index.tolist(),
        "Title": list(titles.astype(str)),
        "Topic": topic_model.topics_,
    }).to_csv(os.path.join(outdir, "doc_assignments.csv"), index=False, encoding="utf-8-sig")

    # 3.3 Topic info + titles by topic
    info = topic_model.get_topic_info()
    info.to_csv(os.path.join(outdir, "topic_info.csv"), index=False)

    titles_by_category = defaultdict(list)
    for title, topic in zip(titles.astype(str), topic_model.topics_):
        titles_by_category[topic].append(title)

    with open(os.path.join(outdir, "titles_by_topic.txt"), "w", encoding="utf-8") as f:
        for cat, papers in sorted(titles_by_category.items()):
            f.write(f"Topic {cat}:\n")
            for p in papers:
                f.write(f"- {p}\n")
            f.write("\n")

    # 3.4 Plotly visualizations
    topic_model.visualize_barchart().write_html(os.path.join(outdir, "barchart.html"))

    n_topics = int((info.Topic != -1).sum())
    if n_topics > 1:
        heatmap_k = max(1, min(15, n_topics - 1))
        topic_model.visualize_heatmap(n_clusters=heatmap_k).write_html(
            os.path.join(outdir, "heatmap.html")
        )
    else:
        print("Skipping heatmap: need at least 2 topics.")

    topic_model.visualize_hierarchy().write_html(os.path.join(outdir, "hierarchy.html"))

    # IMPORTANT: use CLEAN texts for documents view
    docs_fig = topic_model.visualize_documents(
        doc_texts, reduced_embeddings=emb_2d, width=1200, hide_annotations=True
    )
    docs_fig.update_layout(font=dict(size=16))
    docs_fig.write_html(os.path.join(outdir, "documents.html"))

    # 3.5 KeyBERT refinement (on CLEAN texts)
    original_topics = __import__("copy").deepcopy(topic_model.topic_representations_)
    kb_model = KeyBERTInspired()
    topic_model.update_topics(doc_texts, representation_model=kb_model)
    diffs = topic_differences(topic_model, original_topics)
    diffs.to_csv(os.path.join(outdir, "diffs_after_keybert.csv"), index=False)

    # 3.6 Optional auto-label (on CLEAN texts)
    if AUTO_LABEL and hf_pipe is not None:
        from bertopic.representation import TextGeneration
        prompt = (
            "I have a topic that contains the following documents: [DOCUMENTS]\n"
            "The topic is described by the following keywords: [KEYWORDS]\n"
            "Based on the information above, extract a short topic label in the following format:\n"
            "topic: <short topic label>"
        )
        tg_representation = TextGeneration(hf_pipe, prompt=prompt)
        topic_model.update_topics(doc_texts, topic_model.topics_, representation_model=tg_representation)

    # 3.7 Datamap visualizations (if available)
    try:
        freq_df = topic_model.get_topic_freq()
        valid_topics = sorted(freq_df.loc[freq_df.Topic != -1, "Topic"].tolist())
        topics_to_show = valid_topics[:17] if valid_topics else []

        fig_dm = topic_model.visualize_document_datamap(
            doc_texts, topics=topics_to_show, reduced_embeddings=emb_2d, width=1200
        )
        if hasattr(fig_dm, "write_html"):
            fig_dm.write_html(os.path.join(outdir, "datamap.html"))
        elif hasattr(fig_dm, "savefig"):
            fig_dm.savefig(os.path.join(outdir, "datamap.png"), dpi=200, bbox_inches="tight")

        # Numeric labels variant
        num_labels = {topic: str(topic) for topic in freq_df.Topic}
        topic_model.set_topic_labels(num_labels)
        fig_dm2 = topic_model.visualize_document_datamap(
            doc_texts, topics=topics_to_show, reduced_embeddings=emb_2d, width=1200,
            custom_labels=True, topic_prefix=False
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
        for cl in np.unique(lab):
            if cl == -1: 
                continue
            members = np.where(lab == cl)[0]
            if len(members) < 2: 
                continue
            for a_i in members:
                a = idx_run[a_i]
                pa = np.where(idx_all == a)[0][0]
                pb_idx = [np.where(idx_all == idx_run[b_i])[0][0] for b_i in members]
                C[pa, pb_idx] += 1
                counts[pa, pb_idx] += 1

        # also count co-presence (both sampled in this run) even if not co-clustered
        for a in idx_run:
            pa = np.where(idx_all == a)[0][0]
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
            vals.append(-1.0); continue
        p_ab = df_ab / N
        p_a  = df_word.get(a, 0) / N
        p_b  = df_word.get(b, 0) / N
        if p_a <= 0 or p_b <= 0 or p_ab <= 0:
            vals.append(-1.0); continue
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
            embedding_model=embedding_model,  # keep embedder for consistency (XPU/CPU)
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
