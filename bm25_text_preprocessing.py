import json
import re
import pandas as pd
import pickle
import os
import numpy as np
import multiprocessing as mp
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from ko_ww_stopwords.stop_words import ko_ww_stop_words
from tqdm import tqdm
import csv
from more_itertools import chunked
from multiprocessing import Pool

# BM25 Parameters
k1 = 1.5
b = 0.75

# Paths
path_to_corpus_json = "data/corpus.json/corpus.json"
path_to_save_files = "data/saved_files/"
path_to_query = "data/test.csv"

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define stop words
stop_words_dict = {
    "en": set(stopwords.words("english")),
    "fr": set(stopwords.words("french")),
    "de": set(stopwords.words("german")),
    "es": set(stopwords.words("spanish")),
    "it": set(stopwords.words("italian")),
    "ar": set(stopwords.words("arabic")),
}
stop_words_dict["ko"] = set(ko_ww_stop_words)


# Preprocess text
def preprocess_text(text, lang):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words_dict.get(lang, set())]
    tokens = [lemmatizer.lemmatize(word).lower() for word in tokens]
    return tokens


# Compute Inverse Document Frequency (IDF)
def compute_idf(corpus, doc_freq, num_docs):
    print("Starting IDF computation...")
    idf = {term: np.log((num_docs - freq + 0.5) / (freq + 0.5) + 1) for term, freq in tqdm(doc_freq.items())}
    print("IDF computation completed.")
    return idf


# BM25 Scoring function
def bm25_score(query_terms, doc, doc_length, avg_doc_length, idf):
    score = 0.0
    for term in query_terms:
        if term in doc:
            tf = doc[term]
            numerator = idf[term] * tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            score += numerator / denominator
    return score


# Define the process_doc function at the top level
def process_doc(doc):
    tokens = preprocess_text(doc["text"], doc["lang"])
    term_counts = {}
    for term in tokens:
        term_counts[term] = term_counts.get(term, 0) + 1
    unique_terms = set(tokens)
    return doc["docid"], term_counts, unique_terms, len(tokens)


# Preprocess Corpus in Parallel
def preprocess_corpus_parallel():
    with open(path_to_corpus_json, 'r') as f:
        corpus_json = json.load(f)

    doc_freq = {}
    doc_lengths = []
    corpus = {}

    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_doc, corpus_json), total=len(corpus_json)))
        
    pool.close()
    pool.join() 
    
    with open(path_to_save_files + "preprocessed_corpus.pkl", "wb") as f:
        print("Saving preprocessed corpus...")
        pickle.dump(results, f)
    

    
    print("Building term frequency matrix...")
    for doc_id, term_counts, unique_terms, doc_length in tqdm(results):
        corpus[doc_id] = term_counts
        doc_lengths.append(doc_length)
        for term in unique_terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    avg_doc_length = np.mean(doc_lengths)
    
    with open(path_to_save_files + "avg_doc_length.pkl", "wb") as f:
        print("Saving avg_doc_length...")
        pickle.dump(avg_doc_length, f)
    print("Computing IDF...")
    idf = compute_idf(corpus, doc_freq, len(corpus_json))
    
    with open(path_to_save_files + "idf.pkl", "wb") as f:
        print("Saving idf...")
        pickle.dump(idf, f)
    
    print("Preprocessing completed.")
    return corpus

# Preprocess Queries
def preprocess_queries():
    query_df = pd.read_csv(path_to_query)
    queries = {}
    for i in tqdm(range(len(query_df))):
        queries[query_df["id"][i]] = preprocess_text(query_df["query"][i], query_df["lang"][i])
    return queries


# Rank Documents using BM25 in Parallel
def score_single_query_batch(query_id, query_terms, corpus_batch, avg_doc_length, idf):
    scores = []
    for doc_id, doc_terms in corpus_batch.items():
        doc_length = sum(doc_terms.values())
        score = bm25_score(query_terms, doc_terms, doc_length, avg_doc_length, idf)
        scores.append((doc_id, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return query_id, [doc_id for doc_id, _ in scores[:10]]

def rank_documents_bm25_parallel(queries, corpus, idf, avg_doc_length, batch_size=1000):
    corpus_items = list(corpus.items())
    query_results = []
    with Pool() as pool:
        for i in tqdm(range(0, len(corpus_items), batch_size)):
            corpus_batch = dict(corpus_items[i:i+batch_size])
            batch_results = list(tqdm(pool.starmap(score_single_query_batch, [(query_id, query_terms, corpus_batch, avg_doc_length, idf) for query_id, query_terms in queries.items()]), total=len(queries)))
            query_results.extend(batch_results)
    return query_results


def main():
    # Load or preprocess corpus
    if os.path.exists(path_to_save_files + "preprocessed_corpus.pkl"):
    
        # Load the Corpus
        with open(path_to_save_files + "preprocessed_corpus.pkl", "rb") as f:
            print("Loading preprocessed corpus...")
            corpus = pickle.load(f)
        # Load the IDF
        with open(path_to_save_files + "idf.pkl", "rb") as f:
            print("Loading preprocessed idf...")
            idf = pickle.load(f)
        # Load the Average Document Length
        with open(path_to_save_files + "avg_doc_length.pkl", "rb") as f:
            print("Loading preprocessed avg_doc_length...")
            avg_doc_length = pickle.load(f)
    else:
        print("Preprocessing corpus...")
        corpus = preprocess_corpus_parallel()
        
        # Save the Corpus only
        with open(path_to_save_files + "preprocessed_corpus.pkl", "wb") as f:
            print("Saving preprocessed corpus...")
            pickle.dump(corpus, f)
        # Save the IDF
        with open(path_to_save_files + "idf.pkl", "wb") as f:
            print("Saving preprocessed idf...")
            pickle.dump(idf, f)
        # Save the Average Document Length
        with open(path_to_save_files + "avg_doc_length.pkl", "wb") as f:
            print("Saving preprocessed avg_doc_length...")
            pickle.dump(avg_doc_length, f)
            
            
    
    print(corpus.items())
    exit()
    # Load or preprocess queries
    print("Preprocessing queries...")
    queries = preprocess_queries()

    # Rank documents using BM25
    print("Ranking documents...")
    bm25_results = rank_documents_bm25_parallel(queries, corpus, idf, avg_doc_length)

    # Save results
    with open(path_to_save_files + "final_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "docids"])
        for query_id, doc_ids in bm25_results.items():
            writer.writerow([query_id, ",".join(doc_ids)])

    print("Ranking completed and saved to final_results.csv.")


if __name__ == "__main__":
    main()