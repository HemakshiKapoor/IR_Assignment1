# indexer.py
# Build inverted index: postings(term -> [[docID, tf], ...]),
# document frequencies (df), and document lengths (lnc-based).
# Usage: python indexer.py Corpus index.json

import os, re, sys, json, math
from collections import Counter, defaultdict

# --- PREPROCESSING ---
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    _have_nltk = True
except Exception:
    _have_nltk = False

# Fallback stopwords & stemmer if nltk not available
FALLBACK_STOP = set([
    "the","is","at","which","on","and","a","an","for","to","in","of","that","this","it","with"
])
def _simple_stem(tok):
    for suf in ('ing','ly','ed','s'):
        if tok.endswith(suf) and len(tok) > len(suf)+2:
            return tok[:-len(suf)]
    return tok

if _have_nltk:
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = FALLBACK_STOP
    stemmer = PorterStemmer()
    def stem(tok): return stemmer.stem(tok)
    def tokenize(text):
        try:
            return nltk.word_tokenize(text)
        except Exception:
            return re.findall(r"[A-Za-z0-9']+", text)
else:
    stop_words = FALLBACK_STOP
    def stem(tok): return _simple_stem(tok)
    def tokenize(text): return re.findall(r"[A-Za-z0-9']+", text)

def preprocess_text(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords, stem"""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
    toks = tokenize(text)
    toks = [t for t in toks if t and t not in stop_words and len(t) > 1]
    toks = [stem(t) for t in toks]
    return toks

def read_corpus(corpus_path):
    docs = {}
    files = sorted([f for f in os.listdir(corpus_path) if f.lower().endswith('.txt')])
    for fname in files:
        path = os.path.join(corpus_path, fname)
        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
            txt = fh.read()
        toks = preprocess_text(txt)
        docs[fname] = toks
    return docs

# --- INDEX BUILDING ---
def build_index(docs):
    postings = defaultdict(list)   # term -> list of [docID, tf]
    docs_map = {}                  # docID -> filename
    doc_len = {}                   # docID -> document length
    N = len(docs)

    for i, fname in enumerate(sorted(docs.keys()), start=1):
        toks = docs[fname]
        docs_map[str(i)] = fname
        tf = Counter(toks)

        # compute lnc weights for doc length
        weights = {t: 1.0 + math.log10(c) for t, c in tf.items() if c > 0}
        norm = math.sqrt(sum(w*w for w in weights.values())) if weights else 1.0
        doc_len[str(i)] = norm

        # store raw tf in postings
        for t, c in tf.items():
            postings[t].append([i, int(c)])

    df = {t: len(plist) for t, plist in postings.items()}

    index_data = {
        "N": N,
        "docs": docs_map,
        "postings": dict(postings),
        "df": df,
        "doc_len": doc_len
    }
    return index_data

# --- MAIN ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python indexer.py Corpus index.json")
        sys.exit(1)

    corpus = sys.argv[1]
    out = sys.argv[2]

    if not os.path.exists(corpus):
        print("Error: corpus folder not found:", corpus)
        sys.exit(1)

    print("Reading corpus from:", corpus)
    docs = read_corpus(corpus)
    print(f"Documents found: {len(docs)}")

    print("Building index ...")
    idx = build_index(docs)

    with open(out, 'w', encoding='utf-8') as fo:
        json.dump(idx, fo)

    print("Index written to", out)
    print("N =", idx["N"])
    print("Vocabulary size =", len(idx["postings"]))

    # sample debug prints
    sample = sorted(idx["df"].items(), key=lambda x: -x[1])[:10]
    print("Top-10 terms by df (term, df):")
    for t, d in sample:
        print(" ", t, d)

    if sample:
        t0 = sample[0][0]
        print(f"Sample postings for term '{t0}':", idx["postings"][t0][:10])
    print("Doc length for docID 1:", idx["doc_len"].get("1", None))
