import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 

nltk.download('punkt')        #tokenization
nltk.download('stopwords')    #stop word list

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)      #remove punctuation and lowercase
    text = text.lower()

    words = nltk.word_tokenize(text)

    result = []
    for w in words :
        if w not in stop_words:
            result.append(stemmer.stem(w))
    return result

def get_docs(folder):
    
    docs = {}
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            with open(os.path.join(folder, fname), 'r', encoding='utf8', errors='ignore') as f:
                txt = f.read()
                docs[fname] = clean_text(txt)
    return docs

if __name__ == "__main__":
    folder = "Corpus" 

    if not os.path.exists(folder):
        print(" Corpus folder not found.")
    else:
        docs = get_docs(folder)
        print('Total docs loaded:', len(docs))
        
        if docs:
            first = list(docs.keys())[0]
            print('\nFirst 20 words of', first, ':')
            print(docs[first][:20])
            print('\nFirst 200 chars of original:')
            with open(os.path.join(folder, first), 'r', encoding='utf8', errors='ignore') as f:
                orig = f.read()
                print(orig[:200])
        else:
            print('No documents found.')