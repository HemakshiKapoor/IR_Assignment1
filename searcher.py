import sys
import json
import math
import re
from collections import Counter, defaultdict
import os


# --- PREPROCESSING (re-using from indexer.py to ensure consistency) ---
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

# --- RANKING FUNCTIONS ---

def calculate_query_weights(query_tokens, df, N):
    """
    Calculates the ltc weights for the query vector:
    Term Frequency (tf): 1 + log10(tf)
    Inverse Document Frequency (idf): log10(N/df)
    Normalization: Cosine normalization (L2 norm)
    Returns: {term: weight}
    """
    query_tf = Counter(query_tokens)
    query_weights = {}
    query_norm_sq = 0.0

    for term, tf_raw in query_tf.items():
        if term in df: # Only consider terms present in our corpus
            tf_weight = 1 + math.log10(tf_raw) if tf_raw > 0 else 0
            idf_weight = math.log10(N / df[term]) if df[term] > 0 else 0 # Should always be > 0 if term in df
            query_weights[term] = tf_weight * idf_weight
            query_norm_sq += query_weights[term] ** 2
    
    query_norm = math.sqrt(query_norm_sq) if query_norm_sq > 0 else 1.0

    # Apply cosine normalization (divide by L2 norm)
    for term in query_weights:
        query_weights[term] /= query_norm

    return query_weights

def calculate_document_weights(term, doc_id, tf_raw, doc_len):
    """
    Calculates the lnc weight for a document term:
    Term Frequency (tf): 1 + log10(tf)
    Normalization: Use pre-calculated document length (lnc-based)
    Returns: weighted_tf / doc_len
    """
    tf_weight = 1 + math.log10(tf_raw) if tf_raw > 0 else 0
    # No IDF for document in lnc.ltc scheme (ddd.qqq)
    # The pre-calculated doc_len already accounts for the 'lnc' part (log tf and cosine norm for doc)
    
    # We will combine this with query weights, so we just need the TF component
    # The document vector is effectively normalized by doc_len during similarity calculation
    # For a term in a document, its 'lnc' weight is (1 + log10(tf)) / doc_len
    
    # Return raw (1+log(tf)) here, we will divide by doc_len in similarity calculation
    return tf_weight

def compute_cosine_similarity(query_weights, index_data):
    """
    Computes cosine similarity between the query and all relevant documents.
    Implements lnc.ltc ranking scheme.
    """
    similarities = defaultdict(float) # docID -> similarity score
    
    N = index_data["N"]
    docs_map = index_data["docs"]
    postings = index_data["postings"]
    df = index_data["df"]
    doc_lengths = index_data["doc_len"] # Already stores lnc-normalized lengths

    # Iterate through query terms to find matching documents
    for query_term, q_weight in query_weights.items():
        if query_term in postings:
            # For each document where this term appears
            for doc_id_raw, tf_raw in postings[query_term]:
                doc_id = str(doc_id_raw) # Ensure doc_id is string to match doc_lengths keys

                # Document term weight (lnc part) is (1 + log10(tf))
                doc_term_weight_tf = calculate_document_weights(query_term, doc_id, tf_raw, doc_lengths[doc_id])
                
                # The dot product part: query_weight * document_term_weight_TF
                # Then we divide by document length (doc_lengths[doc_id]) later
                similarities[doc_id] += q_weight * doc_term_weight_tf

    # Final division by document length for cosine similarity
    for doc_id in similarities:
        if doc_id in doc_lengths and doc_lengths[doc_id] > 0:
            similarities[doc_id] /= doc_lengths[doc_id]
        else:
            similarities[doc_id] = 0.0 # Should not happen if doc_lengths is correctly built

    # Convert docID back to filename for output
    final_similarities = []
    for doc_id, score in similarities.items():
        if score > 0: # Only return documents with positive similarity
            final_similarities.append((docs_map[doc_id], score))
            
    # Sort by similarity (descending), then by docID (ascending) for ties
    final_similarities.sort(key=lambda x: (-x[1], x[0]))

    return final_similarities

# --- MAIN ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python searcher.py index.json \"Your query text here\"")
        sys.exit(1)

    index_file = sys.argv[1]
    query_text = sys.argv[2]

    if not os.path.exists(index_file):
        print(f"Error: Index file '{index_file}' not found.")
        sys.exit(1)

    print(f"Loading index from '{index_file}'...")
    with open(index_file, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    print("Index loaded.")

    N = index_data["N"]
    df = index_data["df"]
    
    print(f"\nProcessing query: '{query_text}'")
    query_tokens = preprocess_text(query_text)
    if not query_tokens:
        print("No meaningful terms found in query after preprocessing. Please try a different query.")
        sys.exit(0)
    
    print(f"Query tokens: {query_tokens}")

    # Calculate query weights (ltc scheme)
    query_weights = calculate_query_weights(query_tokens, df, N)
    print(f"Query weights: {query_weights}")

    # Compute cosine similarities (lnc.ltc scheme)
    ranked_documents = compute_cosine_similarity(query_weights, index_data)

    print("\n--- Top 10 Search Results ---")
    if not ranked_documents:
        print("No matching documents found.")
    else:
        for i, (filename, score) in enumerate(ranked_documents[:10]): # Limit to top 10
            print(f"{i+1}. ('{filename}', {score})")

    # Example test cases provided in the assignment (for verification)
    # Q1: 'Developing your Zomato business account and profile is a great way to boost your restaurant's online reputation'
    # Q2: 'Warwickshire, came from an ancient family and was the heiress to some land'
    print("\nRemember to test with the provided sample queries for verification.")