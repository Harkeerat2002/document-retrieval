import os
import faiss
import numpy as np
import pandas as pd
import pickle
import gc
import string
from tqdm import tqdm
from collections import defaultdict, Counter
from scipy.sparse import lil_matrix, vstack, save_npz, load_npz, csr_matrix
from scipy.spatial.distance import cosine
from scipy.sparse.linalg import norm
from sklearn.decomposition import PCA
from multiprocessing import Pool, cpu_count
from nltk.corpus import stopwords
from ko_ww_stopwords.stop_words import ko_ww_stop_words
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import IncrementalPCA
tqdm.pandas()

# Constants and paths
path_to_save_df = "./data/pd_df/"
path_to_saved_file = "./data/saved_files/"
path_to_train_query = "./data/train.csv"
corpus_file_path = "./data/corpus.json/corpus.json"
path_save_emb = "./data/saved_files/"
k1 = 1.5
b = 0.75
BATCH_SIZE = 1000  # Batch size for processing

# Stop words and stemming setup
stop_words_dict = {
    "en": set(stopwords.words("english")),
    "fr": set(stopwords.words("french")),
    "de": set(stopwords.words("german")),
    "es": set(stopwords.words("spanish")),
    "it": set(stopwords.words("italian")),
    "ar": set(stopwords.words("arabic")),
    "ko": set(ko_ww_stop_words),
}
stemmer = PorterStemmer()

# Ensure output directories exist
os.makedirs(path_to_save_df, exist_ok=True)
os.makedirs(path_to_saved_file, exist_ok=True)

# Text preprocessing function
def preprocess(text, lang):
    text = pd.Series(text).str.lower().str.replace(f"[{string.punctuation}]", " ", regex=True).iloc[0]
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words_dict.get(lang, set())])
    return text

# Batch processing function for corpus
def preprocess_batch(corpus_chunk, batch_idx):
    # tqdm within each batch for progress tracking
    corpus_chunk["preprocessed_text"] = [
        preprocess(text, lang) 
        for text, lang in tqdm(corpus_chunk[["text", "lang"]].values, 
                               desc=f"Batch {batch_idx} Preprocessing", 
                               leave=False)
    ]
    corpus_chunk["doc_len"] = corpus_chunk["preprocessed_text"].apply(lambda x: len(x.split()))
    return corpus_chunk[["docid", "preprocessed_text", "doc_len"]]

# Function to preprocess corpus in batches with multiprocessing and batch tracking
def preprocess_corpus_in_batches(corpus_df, batch_size, n_jobs):
    num_batches = len(corpus_df) // batch_size + (1 if len(corpus_df) % batch_size != 0 else 0)
    processed_batches = []

    with Pool(processes=n_jobs) as pool:
        for i in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(corpus_df))
            corpus_chunk = corpus_df.iloc[start_idx:end_idx]
            # Track batch processing
            processed_batches.append(pool.apply_async(preprocess_batch, (corpus_chunk, i)))

        # Gather results from parallel processes
        processed_corpus = pd.concat([batch.get() for batch in processed_batches], ignore_index=True)

    return processed_corpus

# Compute TF, DF, and average document length
def compute_tf_df_and_avgdl(corpus_df, path_to_saved_file):
    tf_dict = defaultdict(lambda: defaultdict(int))
    df_dict = defaultdict(int)
    total_length = 0
    num_docs = len(corpus_df)

    for doc_id, row in tqdm(corpus_df.iterrows(), desc="Computing TF, DF, and avgdl", total=num_docs):
        doc_id = row["docid"]
        text = row["preprocessed_text"]
        words = text.split()
        total_length += len(words)
        unique_words = set(words)

        for word in words:
            tf_dict[word][doc_id] += 1

        for word in unique_words:
            df_dict[word] += 1

    avgdl = total_length / num_docs
    tf_dict, df_dict = dict(tf_dict), dict(df_dict)

    with open(path_to_saved_file + "tf_dict.pkl", "wb") as f:
        pickle.dump(tf_dict, f)
    with open(path_to_saved_file + "df_dict.pkl", "wb") as f:
        pickle.dump(df_dict, f)
    with open(path_to_saved_file + "avgdl.pkl", "wb") as f:
        pickle.dump(avgdl, f)
    with open(path_to_saved_file + "num_docs.pkl", "wb") as f:
        pickle.dump(num_docs, f)

    return tf_dict, df_dict, avgdl, num_docs

# Compute IDF scores
def compute_idf(df_dict, num_docs):
    return {term: np.log((num_docs - df + 0.5) / (df + 0.5)) for term, df in df_dict.items()}

# TF-IDF embedding creation
def create_tfidf_embedding(corpus, tf_dict, idf_dict, term_index):
    num_terms = len(term_index)
    embeddings, doc_ids = [], []

    for doc_id, text in tqdm(corpus[["docid", "preprocessed_text"]].values, desc="Creating TF-IDF embeddings"):
        term_freqs = Counter(text.split())
        embedding = lil_matrix((1, num_terms), dtype=np.float32)

        for term, freq in term_freqs.items():
            if term in term_index:
                term_tf = freq
                term_idf = idf_dict.get(term, 0)
                tfidf_score = term_tf * term_idf
                embedding[0, term_index[term]] = tfidf_score

        embeddings.append(embedding)
        doc_ids.append(doc_id)

    return vstack(embeddings),

# Save embeddings to memory-mapped file
def save_embeddings_to_mmap(embeddings, mmap_file):
    save_npz(mmap_file, embeddings)

def load_embeddings_from_mmap(mmap_file):
    return load_npz(mmap_file).tocsr()

# Generate query embedding
def generate_query_embedding(query_text, tf_dict, idf_dict, term_index):
    query_embedding = lil_matrix((1, len(term_index)), dtype=np.float32)
    for term in query_text.split():
        if term in term_index:
            query_embedding[0, term_index[term]] = idf_dict.get(term, 0)
    return query_embedding

def init_worker(tf_dict_shared, idf_dict_shared, avgdl_shared):
    global tf_dict
    global idf_dict
    global avgdl
    tf_dict = tf_dict_shared
    idf_dict = idf_dict_shared
    avgdl = avgdl_shared

def bm25_worker(args):
    query_terms, doc_id, doc_len = args
    return doc_id, bm25_score(query_terms, doc_id, doc_len)

# Modify the bm25_score function to use global variables
def bm25_score(query_terms, doc_id, doc_len):
    score = 0.0
    print("doc_id", doc_id)
    for term in query_terms:
        if term in idf_dict:
            idf = idf_dict[term]
            f_td = tf_dict.get(term, {}).get(doc_id, 0)
            denom = f_td + k1 * (1 - b + b * (doc_len / avgdl))
            if denom != 0:
                term_score = idf * ((f_td * (k1 + 1)) / denom)
                score += term_score
    return score

# In the rank_documents_with_bm25 function, use initializer and initargs
def rank_documents_with_bm25(corpus, queries, tf_dict, idf_dict, avgdl, top_k=10):
    ranked_documents = {}
    doc_ids = corpus["docid"].values
    doc_lens = corpus["doc_len"].values
    n_jobs = cpu_count()

    print("Ranking documents with BM25...")
    # Use initializer and initargs to pass shared variables
    with Pool(processes=n_jobs, initializer=init_worker, initargs=(tf_dict, idf_dict, avgdl)) as pool:
        with tqdm(total=len(queries), desc="Ranking queries with BM25") as pbar:
            for query_id, query_text in queries[["id", "preprocessed_query"]].values:
                query_terms = query_text.split()
                args_list = [(query_terms, doc_id, doc_len) for doc_id, doc_len in zip(doc_ids, doc_lens)]

                results = pool.map(bm25_worker, args_list)

                query_scores = dict(results)
                top_docs = sorted(query_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                ranked_documents[query_id] = [doc_id for doc_id, _ in top_docs]
                
                pbar.update(1)

    return ranked_documents

# Reduce dimensionality using TruncatedSVD for sparse matrices
def reduce_dimensions(embeddings, n_components=300):
    print("Reducing dimensionality of embeddings with TruncatedSVD...")
    svd = TruncatedSVD(n_components=n_components)
    
    print("Fitting TruncatedSVD...")
    # Fit and transform the sparse matrix directly
    reduced_embeddings = svd.fit_transform(embeddings)
    
    print(f"Reduced embeddings to {n_components} dimensions.")
    return reduced_embeddings

# Load and preprocess queries
def load_and_preprocess_queries(path_to_train_query):
    train_query = pd.read_csv(path_to_train_query)
    train_query["preprocessed_query"] = train_query.apply(lambda row: preprocess(row["query"], row["lang"]), axis=1)
    return train_query

# Main function
if __name__ == "__main__":
    # Load corpus
    if os.path.exists(path_to_save_df + "preprocessed_corpus.pkl"):
        print("Loading preprocessed corpus...")
        corpus_df = pd.read_pickle(path_to_save_df + "preprocessed_corpus.pkl")
    else:
        print("Loading and preprocessing corpus in batches...")
        corpus_df = pd.read_json(corpus_file_path)
        corpus_df = preprocess_corpus_in_batches(corpus_df, batch_size=BATCH_SIZE, n_jobs=cpu_count())
        corpus_df.to_pickle(path_to_save_df + "preprocessed_corpus.pkl")

    # Load queries
    if os.path.exists(path_to_save_df + "preprocessed_train_query.pkl"):
        print("Loading preprocessed queries...")
        train_query_df = pd.read_pickle(path_to_save_df + "preprocessed_train_query.pkl")
    else:
        print("Loading and preprocessing queries...")
        train_query_df = load_and_preprocess_queries(path_to_train_query)
        train_query_df.to_pickle(path_to_save_df + "preprocessed_train_query.pkl")

    # Compute TF, DF, avgdl, and IDF
    if os.path.exists(path_to_saved_file + "tf_dict.pkl"):
        print("Loading TF, DF, and IDF data...")
        with open(path_to_saved_file + "tf_dict.pkl", "rb") as f:
            tf_dict = pickle.load(f)
        with open(path_to_saved_file + "df_dict.pkl", "rb") as f:
            df_dict = pickle.load(f)
        with open(path_to_saved_file + "avgdl.pkl", "rb") as f:
            avgdl = pickle.load(f)
    else:
        print("Computing TF, DF, and avgdl...")
        tf_dict, df_dict, avgdl, num_docs = compute_tf_df_and_avgdl(corpus_df, path_to_saved_file)

    if os.path.exists(path_to_saved_file + "idf_dict.pkl"):
        with open(path_to_saved_file + "idf_dict.pkl", "rb") as f:
            idf_dict = pickle.load(f)
    else:
        idf_dict = compute_idf(df_dict, num_docs)
        with open(path_to_saved_file + "idf_dict.pkl", "wb") as f:
            pickle.dump(idf_dict, f)

    # Create term index and embeddings
         # Save the Embeddings to a memory-mapped file
    # Check if embeddings are already saved
    if os.path.exists(os.path.join(path_save_emb, "tfidf_embeddings_before_pca.npz")):
        print("Loading precomputed embeddings...")
        embeddings = load_embeddings_from_mmap(os.path.join(path_save_emb, "tfidf_embeddings_before_pca.npz"))
    else:
        print("Creating TF-IDF embeddings...")
        term_index = {term: idx for idx, term in enumerate(df_dict.keys())}
        embeddings, doc_ids = create_tfidf_embedding(corpus_df, tf_dict, idf_dict, term_index)
        mmap_file = os.path.join(path_save_emb, "tfidf_embeddings_before_pca.npz")
        save_embeddings_to_mmap(embeddings, mmap_file)
    
   

    # # Dimensionality reduction with PCA
    # embeddings = reduce_dimensions(embeddings)

    # # Save embeddings to a memory-mapped file
    # mmap_file = os.path.join(path_save_emb, "tfidf_embeddings.npz")
    # save_embeddings_to_mmap(embeddings, mmap_file)

    # # Load memory-mapped embeddings and rank with BM25
    # embeddings = load_embeddings_from_mmap(mmap_file)
    
     # Preprocess the test queries
    path_to_test_query = "./data/test.csv"
    test_query_df = load_and_preprocess_queries(path_to_test_query) 
    ranked_docs = rank_documents_with_bm25(corpus_df, test_query_df, tf_dict, idf_dict, avgdl, top_k=10)

    # Display the ranked documents for each query
    for query_id, docs in ranked_docs.items():
        print(f"Query {query_id}: Top documents {docs}")
