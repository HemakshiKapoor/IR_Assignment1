import os
import re
import sys
import json
import math
from collections import Counter, defaultdict

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    stop_words = set(stopwords.words('english'))
    stem = PorterStemmer().stem
    def tokenize(txt): return nltk.word_tokenize(txt)
except:
    stop_words = set(['the','is','at','which','on','and','a','an','for','to','in','of','that','this','it','with'])
    def stem(word):
        for suf in ('ing','ly','ed','s'): 
            if word.endswith(suf) and len(word) > len(suf) + 2:
                return word[:-len(suf)]
        return word
    def tokenize(txt):
        return re.findall(r"[A-Za-z0-9']+", txt)

def preprocess(txt):
    txt = re.sub(r'[^a-zA-Z0-9\s]', ' ', txt).lower()
    tokens = tokenize(txt)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [stem(t) for t in tokens]
    return tokens

def get_docs(folder):
    docs = {}
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.txt')])
    for fname in files:
        with open(os.path.join(folder, fname), 'r', encoding='utf-8', errors='ignore') as f:
            docs[fname] = preprocess(f.read())
    return docs

def make_index(docs):
    postings = defaultdict(list)
    num_to_doc = {}
    lens = {}
    for docid, name in enumerate(sorted(docs.keys()), 1):
        toks = docs[name]
        num_to_doc[str(docid)] = name
        freq = Counter(toks)
        wts = {t: 1.0 + math.log10(c) for t, c in freq.items() if c > 0}
        norm = math.sqrt(sum(w**2 for w in wts.values())) or 1.0
        lens[str(docid)] = norm
        for t, c in freq.items():
            postings[t].append([docid, int(c)])
    docs_freq = {t: len(lst) for t, lst in postings.items()}
    return {
        "N": len(docs),
        "docs": num_to_doc,
        "postings": dict(postings),
        "df": docs_freq,
        "doc_len": lens
    }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python indexer.py Corpus index.json")
        sys.exit(1)
    folder, out_file = sys.argv[1], sys.argv[2]
    if not os.path.exists(folder):
        print("Corpus folder not found:", folder)
        sys.exit(1)
    docs = get_docs(folder)
    idx = make_index(docs)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(idx, f)
    print("Index written to", out_file)
    print("Docs:", idx['N'], "| Vocab:", len(idx["postings"]))
