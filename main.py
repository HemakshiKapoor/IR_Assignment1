import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # A common stemming algorithm

# Uncomment these two lines and run the script once to download the necessary data
# nltk.download('punkt')        # For tokenization
# nltk.download('stopwords')    # For stop word list
# nltk.download('wordnet')      # For lemmatization (we'll use stemming, but good to have)

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Performs tokenization, lowercasing, punctuation removal,
    stop word removal, and stemming on the input text.
    Returns a list of processed tokens.
    """
    # 1. Lowercasing and Punctuation Removal
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

    # 2. Tokenization
    tokens = nltk.word_tokenize(text)

    # 3. Stop Word Removal and Stemming
    processed_tokens = []
    for word in tokens:
        if word not in stop_words:
            stemmed_word = stemmer.stem(word)
            processed_tokens.append(stemmed_word)
    return processed_tokens

def read_corpus(corpus_path):
    """
    Reads all text files from the specified corpus directory,
    applies preprocessing, and returns a dictionary where keys are filenames (docIDs)
    and values are lists of preprocessed tokens.
    """
    documents = {}
    for filename in os.listdir(corpus_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(corpus_path, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Apply preprocessing here
                processed_content = preprocess_text(content)
                documents[filename] = processed_content
    return documents

if __name__ == "__main__":
    corpus_directory = "Corpus" # Make sure this matches your folder name

    if not os.path.exists(corpus_directory):
        print(f"Error: Corpus directory '{corpus_directory}' not found.")
        print("Please make sure you have unzipped the corpus into your project folder.")
    else:
        all_documents_processed = read_corpus(corpus_directory)

        print(f"Loaded and processed {len(all_documents_processed)} documents.")
        # Print processed tokens of the first document to verify
        if all_documents_processed:
            first_doc_id = list(all_documents_processed.keys())[0]
            print(f"\nProcessed tokens of '{first_doc_id}' (first 20 tokens):")
            print(all_documents_processed[first_doc_id][:20])
            print(f"\nOriginal content of '{first_doc_id}' (first 200 chars):")
            # To show original content for comparison, you might need to re-read it
            # Or store both processed and raw if needed later. For now, just to verify preprocess.
            with open(os.path.join(corpus_directory, first_doc_id), 'r', encoding='utf-8', errors='ignore') as f:
                original_content = f.read()
                print(original_content[:200])
        else:
            print("No documents found in the corpus.")