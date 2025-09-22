import sys
import json
import math
import re
from collections import Counter, defaultdict
import os

# Preprocessing stuff
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    _have_nltk = True
except Exception:
    _have_nltk = False

fallback_stop = set(["the","is","at","which","on","and","a","an","for","to","in","of","that","this","it","with"])

def simple_stem(word):
    for suf in ('ing','ly','ed','s'):
        if word.endswith(suf) and len(word) > len(suf)+2:
            return word[:-len(suf)]
    return word

if _have_nltk:
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = fallback_stop
    stemmer = PorterStemmer()
    def stem(word): return stemmer.stem(word)
    def tokenize(text):
        try:
            return nltk.word_tokenize(text)
        except Exception:
            return re.findall(r"[A-Za-z0-9']+", text)
else:
    stop_words = fallback_stop
    def stem(word): return simple_stem(word)
    def tokenize(text): return re.findall(r"[A-Za-z0-9']+", text)

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
    tokens = tokenize(text)
    tokens = [t for t in tokens if t and t not in stop_words and len(t) > 1]
    tokens = [stem(t) for t in tokens]
    return tokens

def calc_query_weights(tokens, df, N):
    tf_counts = Counter(tokens)
    weights = {}
    norm_sq = 0.0
    for term, tf_raw in tf_counts.items():
        if term in df:
            tf_w = 1 + math.log10(tf_raw) if tf_raw > 0 else 0
            idf_w = math.log10(N / df[term]) if df[term] > 0 else 0
            weights[term] = tf_w * idf_w
            norm_sq += weights[term] ** 2
    norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
    for term in weights:
        weights[term] /= norm
    return weights

def calc_doc_weight(tf_raw):
    return 1 + math.log10(tf_raw) if tf_raw > 0 else 0

def compute_cosine_sim(query_weights, index):
    sims = defaultdict(float)  # doc id => score
    docs_map = index["docs"]
    postings = index["postings"]
    df = index["df"]
    doc_len = index["doc_len"]

    for term, q_w in query_weights.items():
        if term in postings:
            for doc_id_raw, tf_raw in postings[term]:
                doc_id = str(doc_id_raw)
                doc_w = calc_doc_weight(tf_raw)
                sims[doc_id] += q_w * doc_w

    for doc_id in sims:
        sims[doc_id] /= doc_len.get(doc_id, 1.0)

    results = [(docs_map[doc_id], score) for doc_id, score in sims.items() if score > 0]
    results.sort(key=lambda x: (-x[1], x[0]))

    return results

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python searcher.py index.json \"Your query here\"")
        sys.exit(1)

    index_file = sys.argv[1]
    query = sys.argv[2]

    if not os.path.exists(index_file):
        print(f"Index file '{index_file}' not found.")
        sys.exit(1)

    print(f"Loading index from {index_file}...")
    with open(index_file, 'r', encoding='utf-8') as f:
        idx = json.load(f)
    print("Index loaded.")

    print(f"\nSearching for query: '{query}'")
    tokens = preprocess_text(query)
    if not tokens:
        print("Sorry, no valid terms in query. Try again.")
        sys.exit(0)
    print("Query tokens:", tokens)

    q_weights = calc_query_weights(tokens, idx["df"], idx["N"])
    print("Query weights:", q_weights)

    results = compute_cosine_sim(q_weights, idx)

    print("\nTop 10 results:")
    if not results:
        print("No matches found.")
    else:
        for i, (fname, score) in enumerate(results[:10]):
            print(f"{i+1}. {fname} (Score: {score})")
