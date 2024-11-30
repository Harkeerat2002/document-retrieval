import pandas as pd
import numpy as np
import json
import os
import string
from collections import defaultdict
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pickle
from joblib import Parallel, delayed
from ko_ww_stopwords.stop_words import ko_ww_stop_words
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
from scipy.sparse import csr_matrix
from concurrent.futures import as_completed

# Paths
path_to_save_df = "./data/pd_df/"
path_to_saved_file = "./data/saved_files/"
path_to_train_query = "./data/train.csv"
corpus_file_path = "./data/corpus.json/corpus.json"


# Helper function to preprocess text

# Global function for scoring a single query
def score_single_query(query_id, query_terms, idf_dict, tf_sparse, idf_tensor, doc_lengths, avgdl, k1, b, num_docs, doc_ids, term_index):
    # Initialize the query scores tensor on the GPU if available
    query_scores = torch.zeros(num_docs, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Iterate over terms in the query, scoring each term for relevance
    for term in query_terms:
        if term not in idf_dict:
            continue
        term_idx = term_index[term]
        term_idf = idf_tensor[term_idx]
        
        # Directly use GPU tensor for term frequency
        term_tf = torch.tensor(tf_sparse[:, term_idx].toarray(), dtype=torch.float32).squeeze().to(query_scores.device)

        # Compute the BM25 scores
        numerator = term_tf * (k1 + 1)
        denominator = term_tf + k1 * (1 - b + b * (doc_lengths / avgdl))
        query_scores += term_idf * (numerator / denominator)

    # Retrieve the top 10 documents
    top_scores, top_indices = torch.topk(query_scores, 10)
    top_doc_ids = [doc_ids[idx] for idx in top_indices.tolist()]

    return query_id, top_doc_ids

def preprocess(text, lan):
    text = pd.Series(text).str.lower().str.replace(f"[{string.punctuation}]", " ", regex=True).iloc[0]
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words_dict[lan]])
    return " ".join(text.split())

def process_batch(batch_df, path_to_save_df, batch_index, id, query_text, preprocessed_, file_name):
    preprocessed_data = Parallel(n_jobs=-1)(
        delayed(lambda row: (row[id], preprocess(row[query_text], row["lang"])))(row)
        for _, row in tqdm(batch_df.iterrows(), total=len(batch_df))
    )
    preprocessed_text_df = pd.DataFrame(preprocessed_data, columns=[id, preprocessed_])
    batch_file_path = os.path.join(path_to_save_df, f"{file_name}{batch_index}.pkl")
    preprocessed_text_df.to_pickle(batch_file_path)
    print(f"Batch {batch_index} saved.")

# Load and preprocess corpus
def load_and_preprocess_corpus(corpus_file_path):
    corpus = pd.read_json(corpus_file_path)
    batch_size = 10000
    num_batches = len(corpus) // batch_size + 1
    for i in range(num_batches):
        batch_df = corpus.iloc[i * batch_size:(i + 1) * batch_size]
        process_batch(batch_df, path_to_save_df, i, "docid", "text", "preprocessed_text", "preprocessed_corpus")
    preprocessed_corpus = []
    for i in range(num_batches):
        batch_file_path = os.path.join(path_to_save_df, f"preprocessed_corpus{i}.pkl")
        preprocessed_corpus.append(pd.read_pickle(batch_file_path))
    
    preprocessed_corpus = pd.concat(preprocessed_corpus, ignore_index=True)
    preprocessed_corpus.to_pickle(os.path.join(path_to_save_df, "preprocessed_corpus.pkl"))
    return preprocessed_corpus

# Load and preprocess queries
def load_and_preprocess_queries(path_to_train_query):
    train_query = pd.read_csv(path_to_train_query)
    batch_size = 1000
    num_batches = len(train_query) // batch_size + 1
    for i in range(num_batches):
        batch_df = train_query.iloc[i * batch_size:(i + 1) * batch_size]
        process_batch(batch_df, path_to_save_df, i, "query_id", "query", "preprocessed_query", "preprocessed_train_query")
    preprocessed_train_query = []
    for i in range(num_batches):
        batch_file_path = os.path.join(path_to_save_df, f"preprocessed_train_query{i}.pkl")
        preprocessed_train_query.append(pd.read_pickle(batch_file_path))
    preprocessed_train_query = pd.concat(preprocessed_train_query, ignore_index=True)
    preprocessed_train_query.to_pickle(os.path.join(path_to_save_df, "preprocessed_train_query.pkl"))
    return preprocessed_train_query

# Compute TF, DF, and average document length
def compute_tf_df_and_avgdl(corpus_df, path_to_saved_file):
    # Initialize dictionaries to hold term frequencies and document frequencies
    tf_dict = defaultdict(lambda: defaultdict(int))  # {term: {doc_id: tf}}
    df_dict = defaultdict(int)  # {term: df}
    total_length = 0  # To calculate average document length
    num_docs = len(corpus_df)  # Total number of documents

    # Iterate through each document in the corpus
    for doc_id, row in tqdm(corpus_df.iterrows(), desc="Computing TF, DF, and avgdl", total=num_docs):
        doc_id = row["docid"]  # Get the document ID
        text = row["preprocessed_text"]  # Get the preprocessed text of the document
        words = text.split()  # Split the text into words
        total_length += len(words)  # Update total word count
        unique_words = set(words)  # Get unique words in the document

        # Update term frequencies in tf_dict
        for word in words:
            tf_dict[word][doc_id] += 1  # Increment term frequency for this word in the document
        
        # Update document frequencies in df_dict
        for word in unique_words:
            df_dict[word] += 1  # Increment document frequency for the unique word

    avgdl = total_length / num_docs  # Calculate average document length

    # Convert tf_dict to standard dictionary for serialization
    tf_dict = {k: dict(v) for k, v in tf_dict.items()}  # Convert defaultdict to dict
    df_dict = dict(df_dict)  # Convert defaultdict to dict

    # Save the dictionaries and average document length to files
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
    return {word: np.log((num_docs - df + 0.5) / (df + 0.5) + 1) for word, df in df_dict.items()}


def get_doc_length(doc):
    return len(doc.split())


# def rank_documents_with_bm25_optimized(corpus, train_query, tf_dict, idf_dict, avgdl, k1=1.5, b=0.75):
#     doc_ids = list(corpus["docid"])
#     print("Indexing documents...")
#     doc_id_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
#     num_docs = len(doc_ids)
#     term_index = {term: idx for idx, term in enumerate(tf_dict.keys())}

#     rows, cols, data = [], [], []
#     for term, docs in tqdm(tf_dict.items(), desc="Converting TF dict to sparse matrix", total=len(tf_dict)):
#         for doc_id, freq in docs.items():
#             rows.append(doc_id_index[doc_id])
#             cols.append(term_index[term])
#             data.append(freq)
    

#     print("Converting to sparse matrix...")
#     tf_sparse = csr_matrix((data, (rows, cols)), shape=(num_docs, len(tf_dict)))

#     print("Converting IDF dict to tensor...")
#     idf_tensor = torch.tensor([idf_dict.get(term, 0) for term in tf_dict.keys()], dtype=torch.float32)
#     idf_tensor = idf_tensor.cuda() if torch.cuda.is_available() else idf_tensor

#     print("Computing document lengths...")
#     doc_lengths = torch.tensor([get_doc_length(text) for text in corpus["preprocessed_text"]], dtype=torch.float32)
#     doc_lengths = doc_lengths.cuda() if torch.cuda.is_available() else doc_lengths

#     # Parallel processing with ProcessPoolExecutor
#     ranked_documents_dict = {}
#     print("Ranking documents...")
#     # with ProcessPoolExecutor() as executor:
#     #     futures = [
#     #         executor.submit(
#     #             score_single_query, query_id, query_terms, idf_dict, tf_sparse, idf_tensor, doc_lengths, avgdl, k1, b, num_docs, doc_ids, term_index
#     #         )
#     #         for query_id, query_terms in tqdm(train_query[["query_id", "preprocessed_query"]].values, desc="Processing queries", total=len(train_query))
#     #     ]
#     #     for future in as_completed(futures):
#     #         query_id, top_doc_ids = future.result()
#     #         ranked_documents_dict[query_id] = top_doc_ids
    
#     idf_tensor = idf_tensor.to('cuda')
#     doc_lengths = doc_lengths.to('cuda')

#     # for query_id, query_terms in tqdm(train_query[["query_id", "preprocessed_query"]].values, desc="Processing queries", total=len(train_query)):
#     #     query_id, top_doc_ids = score_single_query(query_id, query_terms, idf_dict, tf_sparse, idf_tensor, doc_lengths, avgdl, k1, b, num_docs, doc_ids, term_index)
#     #     ranked_documents_dict[query_id] = top_doc_ids
    
#     ranked_documents_dict = {}

#     for query_id, query_terms in tqdm(train_query[["query_id", "preprocessed_query"]].values, desc="Processing queries", total=len(train_query)):
#         # Assuming score_single_query is modified to handle GPU tensors
#         query_id, top_doc_ids = score_single_query(query_id, query_terms, idf_dict, tf_sparse, idf_tensor, doc_lengths, avgdl, k1, b, num_docs, doc_ids, term_index)
#         ranked_documents_dict[query_id] = top_doc_ids
    
            

#     return ranked_documents_dict

def score_single_query_sparse_chunk(query_id, query_terms, idf_tensor, tf_sparse_chunk, doc_lengths_chunk, avgdl, k1, b, doc_ids_chunk, term_index):
    # Initialize scores tensor on GPU
    scores = torch.zeros(len(doc_ids_chunk), device='cuda' if torch.cuda.is_available() else 'cpu')

    for term in query_terms:
        if term in term_index:
            term_idf = idf_tensor[term_index[term]].to(scores.device)
            term_idx = term_index[term]

            # Retrieve term frequencies from the sparse matrix chunk
            term_tf = torch.tensor(tf_sparse_chunk[:, term_idx].toarray(), dtype=torch.float32).squeeze().to(scores.device)

            # Calculate BM25 scores for this term across the chunk
            numerator = term_tf * (k1 + 1)
            denominator = term_tf + k1 * (1 - b + b * (doc_lengths_chunk / avgdl))
            scores += term_idf * (numerator / denominator)

    # Retrieve top document IDs and scores
    top_scores, top_indices = torch.topk(scores, 10)
    top_doc_ids = [doc_ids_chunk[idx] for idx in top_indices.tolist()]

    return query_id, top_doc_ids

# Inside rank_documents_with_bm25_optimized, adjust for GPU compatibility
def rank_documents_with_bm25_optimized(corpus, train_query, tf_dict, idf_dict, avgdl, k1=1.5, b=0.75):
    
    # Check if doc_id_index and term_index are already computed
    if os.path.exists(path_to_saved_file + "doc_id_index.pkl") and os.path.exists(path_to_saved_file + "term_index.pkl"):
        print("Loading doc_id_index and term_index...")
        with open(path_to_saved_file + "doc_id_index.pkl", "rb") as f:
            doc_id_index = pickle.load(f)
        with open(path_to_saved_file + "term_index.pkl", "rb") as f:
            term_index = pickle.load(f)
            
        doc_ids = list(corpus["docid"])
        num_docs = len(doc_ids)
    else:
        print("Indexing documents...")
        doc_ids = list(corpus["docid"])
        doc_id_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        num_docs = len(doc_ids)
        term_index = {}
        for term, docs in tf_dict.items():
            term_index[term] = [doc_id_index[doc_id] for doc_id in docs.keys()]
    
        # Save doc_id_index and term_index to disk
        with open(path_to_saved_file + "doc_id_index.pkl", "wb") as f:
            pickle.dump(doc_id_index, f)
        with open(path_to_saved_file + "term_index.pkl", "wb") as f:
            pickle.dump(term_index, f)

    
    # Save doc_id_index and term_index to disk
    with open(path_to_saved_file + "doc_id_index.pkl", "wb") as f:
        pickle.dump(doc_id_index, f)
    with open(path_to_saved_file + "term_index.pkl", "wb") as f:
        pickle.dump(term_index, f)
    
    # Check if tf_sparse and idf_tensor are already computed
    if os.path.exists(path_to_saved_file + "tf_sparse.pkl") and os.path.exists(path_to_saved_file + "idf_tensor.pkl"):
        print("Loading TF sparse matrix and IDF tensor...")
        with open(path_to_saved_file + "tf_sparse.pkl", "rb") as f:
            tf_sparse = pickle.load(f)
        with open(path_to_saved_file + "idf_tensor.pkl", "rb") as f:
            idf_tensor = pickle.load(f)
    else:
        print("Converting to sparse matrix...")
        
        # Sparse TF matrix construction
        print("Converting TF dict to sparse matrix...")
        rows, cols, data = [], [], []
        for term, docs in tf_dict.items():
            for doc_id, freq in docs.items():
                rows.append(doc_id_index[doc_id])
                cols.append(term_index[term])
                data.append(freq)
            
        tf_sparse = csr_matrix((data, (rows, cols)), shape=(num_docs, len(tf_dict)), dtype=np.float32)
        idf_tensor = torch.tensor([idf_dict.get(term, 0) for term in tf_dict.keys()], dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Save the sparse matrix to disk
        with open(path_to_saved_file + "tf_sparse.pkl", "wb") as f:
            pickle.dump(tf_sparse, f)
        # Save the IDF tensor to disk
        with open(path_to_saved_file + "idf_tensor.pkl", "wb") as f:
            pickle.dump(idf_tensor, f)
    
    # Document lengths tensor
    print("Computing document lengths...")
    doc_lengths = torch.tensor([get_doc_length(text) for text in corpus["preprocessed_text"]], dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize results dictionary
    ranked_documents_dict = {}
    batch_size = 64
    print("Ranking documents...")
    for i in range(0, len(train_query), batch_size):
        batch_queries = train_query.iloc[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}... of {len(train_query)//batch_size + 1}")
        for query_id, query_terms in batch_queries[["query_id", "preprocessed_query"]].values:
            # Directly score using sparse data without dense transformation
            query_id, top_doc_ids = score_single_query_sparse(query_id, query_terms, idf_tensor, tf_sparse, doc_lengths, avgdl, k1, b, num_docs, doc_ids, term_index)
            ranked_documents_dict[query_id] = top_doc_ids.cpu().numpy()  # Move to CPU if necessary

    return ranked_documents_dict

def rank_documents_with_bm25_optimized_chunked(corpus, train_query, tf_dict, idf_dict, avgdl, k1=1.5, b=0.75, chunk_size=5000):
    # Setup term index
    doc_ids = list(corpus["docid"])
    doc_id_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    term_index = {term: idx for idx, term in enumerate(tf_dict.keys())}
    num_docs = len(doc_ids)

    # Sparse TF matrix construction in chunks
    rows, cols, data = [], [], []
    for term, docs in tf_dict.items():
        for doc_id, freq in docs.items():
            rows.append(doc_id_index[doc_id])
            cols.append(term_index[term])
            data.append(freq)
    
    tf_sparse = csr_matrix((data, (rows, cols)), shape=(num_docs, len(tf_dict)), dtype=np.float32)
    idf_tensor = torch.tensor([idf_dict.get(term, 0) for term in tf_dict.keys()], dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    doc_lengths = torch.tensor([get_doc_length(text) for text in corpus["preprocessed_text"]], dtype=torch.float32)
    
    ranked_documents_dict = {}

    print("Ranking documents in chunks...")
    for i in tqdm(range(0, num_docs, chunk_size), desc="Processing chunks"):
        chunk_start = i
        chunk_end = min(i + chunk_size, num_docs)
        
        # Prepare chunk data
        tf_sparse_chunk = tf_sparse[chunk_start:chunk_end, :]
        doc_lengths_chunk = doc_lengths[chunk_start:chunk_end].to(idf_tensor.device)
        doc_ids_chunk = doc_ids[chunk_start:chunk_end]

        # Process each query in the current chunk
        # for query_id, query_terms in train_query[["query_id", "preprocessed_query"]].values:
        #     query_id, top_doc_ids = score_single_query_sparse_chunk(
        #         query_id, query_terms, idf_tensor, tf_sparse_chunk, doc_lengths_chunk, avgdl, k1, b, doc_ids_chunk, term_index
        #     )
        #     ranked_documents_dict[query_id] = ranked_documents_dict.get(query_id, []) + top_doc_ids
        
        # Proecess each query in the current chunk using TQDM
        for query_id, query_terms in tqdm(train_query[["query_id", "preprocessed_query"]].values, desc="Processing queries", total=len(train_query)):
            query_id, top_doc_ids = score_single_query_sparse_chunk(
                query_id, query_terms, idf_tensor, tf_sparse_chunk, doc_lengths_chunk, avgdl, k1, b, doc_ids_chunk, term_index
            )
            ranked_documents_dict[query_id] = top_doc_ids

    return ranked_documents_dict


# # Execution Flow
# if not os.path.exists(path_to_save_df):
#     os.makedirs(path_to_save_df)

# # Check if preprocessed corpus exists
# if os.path.exists(path_to_save_df + "preprocessed_corpus.pkl"):
#     print("Loading preprocessed corpus...")
#     corpus_df = pd.read_pickle(path_to_save_df + "preprocessed_corpus.pkl")
# else:
#     print("Loading and preprocessing corpus...")
#     corpus_df = load_and_preprocess_corpus(corpus_file_path)

# if os.path.exists(path_to_save_df + "preprocessed_train_query.pkl"):
#     print("Loading preprocessed queries...")
#     train_query_df = pd.read_pickle(path_to_save_df + "preprocessed_train_query.pkl")
# else:
#     print("Loading and preprocessing queries...")
#     train_query_df = load_and_preprocess_queries(path_to_train_query)


# if os.path.exists(path_to_saved_file + "tf_dict.pkl") and os.path.exists(path_to_saved_file + "df_dict.pkl") and os.path.exists(path_to_saved_file + "avgdl.pkl") and os.path.exists(path_to_saved_file + "num_docs.pkl"):
#     print("Loading TF, DF, and avgdl...")
#     with open(path_to_saved_file + "tf_dict.pkl", "rb") as f:
#         tf_dict = pickle.load(f)
#     with open(path_to_saved_file + "df_dict.pkl", "rb") as f:
#         df_dict = pickle.load(f)
#     with open(path_to_saved_file + "avgdl.pkl", "rb") as f:
#         avgdl = pickle.load(f)
#     with open(path_to_saved_file + "num_docs.pkl", "rb") as f:
#         num_docs = pickle.load(f)
# else:
#     print("Computing TF, DF, and avgdl...")
#     tf_dict, df_dict, avgdl, num_docs = compute_tf_df_and_avgdl(corpus_df, path_to_saved_file)


# print("Computing IDF scores...")
# idf_dict = compute_idf(df_dict, num_docs)

# # # Print 5 tf_dict
# # print("Printing 5 tf_dict")
# # # Print the keys and values of the first item in the dictionary
# # print(list(tf_dict.keys())[:5])
# # exit()


# print("Ranking documents for each query using BM25...")
# ranked_documents_dict = rank_documents_with_bm25_optimized(corpus_df, train_query_df, tf_dict, idf_dict, avgdl)

# # Save ranked documents as CSV
# ranked_documents_df = pd.DataFrame(ranked_documents_dict).T
# ranked_documents_df.columns = [f"doc_{i}" for i in range(1, 11)]
# ranked_documents_df.index.name = "query_id"
# ranked_documents_df.to_csv(os.path.join(path_to_save_df, "ranked_documents_bm25.csv"))

# print("Ranking completed. The ranked documents are saved as 'ranked_documents_bm25.csv'.")


def main():
    nltk.download("stopwords")

    # Paths
    path_to_save_df = "./data/pd_df/"
    path_to_saved_file = "./data/saved_files/"
    path_to_train_query = "./data/train.csv"
    corpus_file_path = "./data/corpus.json/corpus.json"

    # BM25 Parameters
    k1 = 1.5
    b = 0.75

    # Stop Words and Stemmer Initialization
    stop_words_dict = {
        "en": set(stopwords.words("english")),
        "fr": set(stopwords.words("french")),
        "de": set(stopwords.words("german")),
        "es": set(stopwords.words("spanish")),
        "it": set(stopwords.words("italian")),
        "ar": set(stopwords.words("arabic")),
        "ko": set(ko_ww_stop_words)
    }
    stemmer = PorterStemmer()
    # Execution Flow
    if not os.path.exists(path_to_save_df):
        os.makedirs(path_to_save_df)

    # Check if preprocessed corpus exists
    if os.path.exists(path_to_save_df + "preprocessed_corpus.pkl"):
        print("Loading preprocessed corpus...")
        corpus_df = pd.read_pickle(path_to_save_df + "preprocessed_corpus.pkl")
    else:
        print("Loading and preprocessing corpus...")
        corpus_df = load_and_preprocess_corpus(corpus_file_path)

    if os.path.exists(path_to_save_df + "preprocessed_train_query.pkl"):
        print("Loading preprocessed queries...")
        train_query_df = pd.read_pickle(path_to_save_df + "preprocessed_train_query.pkl")
    else:
        print("Loading and preprocessing queries...")
        train_query_df = load_and_preprocess_queries(path_to_train_query)

    if os.path.exists(path_to_saved_file + "tf_dict.pkl") and os.path.exists(path_to_saved_file + "df_dict.pkl") and os.path.exists(path_to_saved_file + "avgdl.pkl") and os.path.exists(path_to_saved_file + "num_docs.pkl"):
        print("Loading TF, DF, and avgdl...")
        with open(path_to_saved_file + "tf_dict.pkl", "rb") as f:
            tf_dict = pickle.load(f)
        with open(path_to_saved_file + "df_dict.pkl", "rb") as f:
            df_dict = pickle.load(f)
        with open(path_to_saved_file + "avgdl.pkl", "rb") as f:
            avgdl = pickle.load(f)
        with open(path_to_saved_file + "num_docs.pkl", "rb") as f:
            num_docs = pickle.load(f)
    else:
        print("Computing TF, DF, and avgdl...")
        tf_dict, df_dict, avgdl, num_docs = compute_tf_df_and_avgdl(corpus_df, path_to_saved_file)

    if os.path.exists(path_to_saved_file + "idf_dict.pkl"):
        print("Loading IDF scores...")
        with open(path_to_saved_file + "idf_dict.pkl", "rb") as f:
            idf_dict = pickle.load(f)
    else:
        print("Computing IDF scores...")
        idf_dict = compute_idf(df_dict, num_docs)
        # Save the IDF dictionary to disk
        with open(path_to_saved_file + "idf_dict.pkl", "wb") as f:
            pickle.dump(idf_dict, f)

    print("Ranking documents...")
    ranked_documents_dict = rank_documents_with_bm25_optimized_chunked(corpus_df, train_query_df, tf_dict, idf_dict, avgdl)
    
    # Save ranked documents as CSV
    ranked_documents_df = pd.DataFrame(ranked_documents_dict).T
    ranked_documents_df.columns = [f"doc_{i}" for i in range(1, 11)]
    ranked_documents_df.index.name = "query_id"
    ranked_documents_df.to_csv(os.path.join(path_to_save_df, "ranked_documents_bm25.csv"))

    print("Ranking completed. The ranked documents are saved as 'ranked_documents_bm25.csv'.")

if __name__ == "__main__":
    main()