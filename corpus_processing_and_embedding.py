import json
import tqdm
import re
import pandas as pd
import pickle
import os
import nltk
import torch
import faiss
import numpy as np
import csv
import torch.multiprocessing as mp
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from ko_ww_stopwords.stop_words import ko_ww_stop_words
from transformers import AutoTokenizer, AutoModel
import multiprocessing as mp
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

# Download the necessary resources
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')

# Stop Words
stop_words_dict = {
    "en": set(stopwords.words("english")),
    "fr": set(stopwords.words("french")),
    "de": set(stopwords.words("german")),
    "es": set(stopwords.words("spanish")),
    "it": set(stopwords.words("italian")),
    "ar": set(stopwords.words("arabic")),
}

# Stop Words Korean
korean_stop_words = set(ko_ww_stop_words)
stop_words_dict["ko"] = korean_stop_words

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Paths
path_to_corpus_json = "data/corpus.json/corpus.json"
path_to_save_df = "data/pd_df/"
path_to_save_files = "data/saved_files/"
path_to_query = "data/dev.csv"

lemmatizer = WordNetLemmatizer()

# Preprocess the text
def preprocess_text(text, lang):
    
    # Remove Special Characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove Stop Words
    tokens = [word for word in tokens if word.lower() not in stop_words_dict[lang]]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Lowercase
    tokens = [word.lower() for word in tokens]
    # Put it back to the corpus
    return " ".join(tokens)


def preprocess_single_document(doc):
    return preprocess_text(doc["text"], doc["lang"])

# Preprocess the corpus using multiprocessing
def preprocess_corpus():
    with open(path_to_corpus_json, 'r') as f:
        corpus_json = json.load(f)
    
    preprocess_c = {}

    # Create a pool of worker processes
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm.tqdm(pool.imap(preprocess_single_document, corpus_json), total=len(corpus_json)))
    
    for doc, preprocessed_text in zip(corpus_json, results):
        preprocess_c[doc["docid"]] = preprocessed_text

    return preprocess_c

def generate_bert_embeddings(text):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

def generate_embeddings_for_document(doc):
    return generate_bert_embeddings(doc)

def generate_embeddings_in_batches(documents, batch_size=1000):
    embeddings = []
    document_values = list(documents.values())
    
    # Create a pool of worker processes
    with mp.Pool(mp.cpu_count(), initializer=init_worker) as pool:
        for i in tqdm.tqdm(range(0, len(document_values), batch_size)):
            batch = document_values[i:i+batch_size]
            batch_embeddings = pool.map(generate_embeddings_for_document, batch)
            embeddings.extend(batch_embeddings)
    
    return np.vstack(embeddings)


def init_worker():
    global model, tokenizer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # Preprocess the Corpus
    if os.path.exists(path_to_save_df + "preprocessed_corpus.pkl"):
        print("Preprocessed Corpus exists, loading it")
        with open(path_to_save_df + "preprocessed_corpus.pkl", "rb") as f:
            preprocessed_c = pickle.load(f)
            
        corpus = json.load(open(path_to_corpus_json))
        
        corpus_vocab = defaultdict(Counter)
        pre_corpus_vocab = defaultdict(Counter)
        doc_ids_and_lang = {}

        def process_main_corpus(doc):
            lang = doc["lang"]
            text = doc["text"]
            doc_id = doc["docid"]
            doc_ids_and_lang[doc_id] = lang
            
            tokens = word_tokenize(text)
            corpus_vocab[lang].update(tokens)

        def process_preprocessed_corpus(doc_id):
            lang = doc_ids_and_lang[doc_id]
            text = preprocessed_c[doc_id]
            
            tokens = word_tokenize(text)
            pre_corpus_vocab[lang].update(tokens)

        # Process the main corpus in parallel
        with ThreadPoolExecutor() as executor:
            list(tqdm.tqdm(executor.map(process_main_corpus, corpus), total=len(corpus)))

        # Process the preprocessed corpus in parallel
        with ThreadPoolExecutor() as executor:
            list(tqdm.tqdm(executor.map(process_preprocessed_corpus, preprocessed_c), total=len(preprocessed_c)))

        # Convert defaultdict(Counter) to regular dict for final output
        corpus_vocab = {lang: dict(counter) for lang, counter in corpus_vocab.items()}
        pre_corpus_vocab = {lang: dict(counter) for lang, counter in pre_corpus_vocab.items()}

        # Calculate corpus vocab lengths
        corpus_vocab_length = {lang: sum(counter.values()) for lang, counter in corpus_vocab.items()}
        pre_corpus_vocab_length = {lang: sum(counter.values()) for lang, counter in pre_corpus_vocab.items()}
                    
        # Print length of vocab of corpus for each lang & total for corpus before preprocessing and after preprocessing
        for lang in corpus_vocab:
            print(f"Lang: {lang}, Vocab Length Before Preprocessing: {corpus_vocab[lang]}, Vocab Length After Preprocessing: {corpus_vocab_length[lang]}")
            
        print("Total Vocab Length Before Preprocessing: ", sum(corpus_vocab.values()))   
        print("Total Vocab Length After Preprocessing: ", sum(corpus_vocab_length.values())) 
        
                    
        
    else:
        print("Preprocessing Corpus")
        preprocessed_c = preprocess_corpus()
                
        with open(path_to_save_files + "preprocessed_corpus.pkl", "wb") as f:
            pickle.dump(preprocessed_c, f)
        
    # Generate the BERT Embeddings 
    if os.path.exists(path_to_save_files + "embeddings.pkl"):
        print("Embeddings exist, loading them")
        with open(path_to_save_files + "embeddings.pkl", "rb") as f:
            corpus_embeddings = pickle.load(f)
    else:
        print("Generating BERT Embeddings for the Corpus")
        corpus_embeddings = generate_embeddings_in_batches(preprocessed_c)
        
        with open(path_to_save_files + "embeddings.pkl", "wb") as f:
            pickle.dump(corpus_embeddings, f)
        
    # Convert list of embeddings to a NumPy array
    corpus_embeddings = np.vstack(corpus_embeddings)
    
    # Create FAISS Index
    if os.path.exists(path_to_save_files + "faiss_index.pkl"):
        print("FAISS Index exists, loading it")
        with open(path_to_save_files + "faiss_index.pkl", "rb") as f:
            index = pickle.load(f)
    else: 
        print("Creating FAISS Index")
        d = corpus_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(corpus_embeddings)
        
        with open(path_to_save_files + "faiss_index.pkl", "wb") as f:
            pickle.dump(index, f)
    
    
    # Preprocess the Query
    print("Preprocessing the Query")
    query_df = pd.read_csv(path_to_query)
    preprocess_query = {}
    for i in tqdm.tqdm(range(len(query_df))):
        preprocess_query[query_df["query_id"][i]] = preprocess_text(query_df["query"][i], query_df["lang"][i])
    

        # Generate BERT Embeddings for the Query
    print("Generating BERT Embeddings for the Query")
    query_embeddings = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for query_id, prep_text in tqdm.tqdm(preprocess_query.items()):
        emb = generate_bert_embeddings(prep_text)
        query_embeddings[query_id] = emb

    # Convert dictionary of embeddings to a NumPy array
    query_ids = list(query_embeddings.keys())
    query_embeddings_array = np.vstack([query_embeddings[qid] for qid in query_ids])

    # Normalize the query embeddings to unit vectors
    query_embeddings_array = query_embeddings_array / np.linalg.norm(query_embeddings_array, axis=1, keepdims=True)

    # Retrieve the Top-K Results using Cosine Similarity
    k = 10
    results = []
    
    doc_ids = list(preprocessed_c.keys())

    print("Retrieving the Top-K Results using Cosine Similarity")
    # Rank the top k documents with cosine similarity
    for i in tqdm.tqdm(range(len(query_embeddings_array))):
        query_id = query_ids[i]
        query_emb = query_embeddings_array[i]
        scores, neighbors = index.search(query_emb.reshape(1, -1), k)
        results.append((query_id, [doc_ids[n] for n in neighbors[0]]))
        

    print(results[:5])
    
    possitive_docs = list(query_df["positive_docs"])
    negative_docs = list(query_df["negative_docs"])
    
    
    # Evaluate the Results and get average precision
    print("Evaluating the Results")
    possitive_d = 0
    for i in range(len(results)):
        if possitive_docs[i] in results[i][1]:
            possitive_d += 1
        
    
    print("possitive_d: ", possitive_d)
    

if __name__ == "__main__":
    main()