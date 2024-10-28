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
from numba import njit, prange
from joblib import Parallel, delayed
from ko_ww_stopwords.stop_words import ko_ww_stop_words

nltk.download("stopwords")

# Paths
path_to_save_df = "./data/pd_df/"
path_to_saved_file = "./data/saved_files/"
path_to_train_query = "./data/train.csv"
corpus_file_path = "./data/corpus.json/corpus.json"

# Parameters for BM25
k1 = 1.5  # BM25 parameter, typically between 1.2 and 2.0
b = 0.75  # BM25 parameter, typically 0.75

# Set frequency threshold for vocabulary limitation
frequency_threshold = 5  # Minimum frequency for a term to be included

# Initialize NLTK resources
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

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

stemmer = PorterStemmer()


# Helper function to preprocess text
def preprocess(text, lan, docid):
    # print("Before Preprocessing: ", len(text), docid)
    # Lowercasing and removing punctuation using pandas vectorized operations
    text = (
        pd.Series(text)
        .str.lower()
        .str.replace(f"[{string.punctuation}]", " ", regex=True)
        .iloc[0]
    )

    # Remove Stopwords without tokenization
    text = " ".join([word for word in text.split() if word not in stop_words_dict[lan]])

    # Stemming
    for word in text.split():
        text = text.replace(word, stemmer.stem(word))

    # Extra whitespace removal
    for _ in range(10):
        text = text.replace("  ", " ")

    # print("After Preprocessing: ", len(text), docid)

    return text


def process_batch(batch_df, path_to_save_df, batch_index):
    preprocessed_data = Parallel(n_jobs=-1)(
        delayed(
            lambda row: (
                row["query_id"],
                preprocess(row["query"], row["lang"], row["query_id"]),
            )
        )(row)
        for _, row in tqdm(batch_df.iterrows(), total=len(batch_df))
    )
    preprocessed_text_df = pd.DataFrame(
        preprocessed_data, columns=["query_id", "preprocessed_query"]
    )

    batch_file_path = os.path.join(
        path_to_save_df, f"preprocessed_train_query{batch_index}.pkl"
    )
    preprocessed_text_df.to_pickle(batch_file_path)
    print(f"Batch {batch_index} saved.")


# Load and preprocess corpus
def load_and_preprocess_corpus(corpus_file_path):
    with open(corpus_file_path, "r") as f:
        corpus = json.load(f)
    return {doc_id: preprocess(text) for doc_id, text in corpus.items()}


# Compute TF, DF, and average document length
def compute_tf_df_and_avgdl(corpus):
    tf_dict = defaultdict(lambda: defaultdict(int))
    df_dict = defaultdict(int)
    total_length = 0
    num_docs = len(corpus)

    for index, row in tqdm(
        corpus.iterrows(), desc="Computing TF, DF, and avgdl", total=num_docs
    ):
        text = row["preprocessed_text"]
        words = text.split()
        unique_words = set(words)
        total_length += len(words)

        # Calculate term frequency for each word
        for word in words:
            tf_dict[index][word] += 1
        # Calculate document frequency for each unique word
        for word in unique_words:
            df_dict[word] += 1

    avgdl = total_length / num_docs

    # Filter out low-frequency terms
    frequency_threshold = 1  # Define your frequency threshold
    df_dict = {word: df for word, df in df_dict.items() if df >= frequency_threshold}

    # Convert defaultdict to dict before pickling
    tf_dict = {k: dict(v) for k, v in tf_dict.items()}
    df_dict = dict(df_dict)

    # Save the computed values
    with open(path_to_saved_file + "tf_dict.pkl", "wb") as f:
        pickle.dump(tf_dict, f)
    with open(path_to_saved_file + "df_dict.pkl", "wb") as f:
        pickle.dump(df_dict, f)
    with open(path_to_saved_file + "avgdl.pkl", "wb") as f:
        pickle.dump(avgdl, f)
    with open(path_to_saved_file + "num_docs.pkl", "wb") as f:
        pickle.dump(num_docs, f)

    return tf_dict, df_dict, avgdl, num_docs


# Compute IDF scores based on document frequency
def compute_idf(df_dict, num_docs):
    idf_dict = {}
    for word, df in df_dict.items():
        idf_dict[word] = np.log((num_docs - df + 0.5) / (df + 0.5) + 1)
    return idf_dict


# BM25 ranking function
# def rank_documents_with_bm25(corpus, train_query, tf_dict, idf_dict, avgdl):
#     ranked_documents_dict = {}
#     print("Columns in train_query: ", train_query.columns)

#     k1 = 1.5  # Typical value for k1
#     b = 0.75  # Typical value for b

#     for _, row in tqdm(
#         train_query.iterrows(), desc="Ranking documents", total=len(train_query)
#     ):
#         query_id = row["query_id"]
#         query_terms = row["preprocessed_query"].split()
#         scores = defaultdict(float)

#         for doc_id, doc_text in corpus.iterrows():
#             doc_len = len(doc_text["preprocessed_text"].split())
#             doc_score = 0

#             for term in query_terms:
#                 if term in tf_dict[doc_id]:
#                     term_freq = tf_dict[doc_id][term]
#                     idf = idf_dict.get(term, 0)
#                     numerator = term_freq * (k1 + 1)
#                     denominator = term_freq + k1 * (1 - b + b * (doc_len / avgdl))
#                     doc_score += idf * (numerator / denominator)

#             scores[doc_id] = doc_score

#         # Sort and keep the top 10 documents for the query
#         ranked_docs = sorted(scores, key=scores.get, reverse=True)[:10]
#         ranked_documents_dict[query_id] = ranked_docs

#     return ranked_documents_dict



def get_doc_length(doc):
    # Calculate document length on demand
    return len(doc.split())

def rank_documents_with_bm25_optimized(corpus, train_query, tf_dict, idf_dict, avgdl, k1=1.5, b=0.75, n_jobs=4):
    ranked_documents_dict = {}

    def score_single_query(query_id, query_terms):
        scores = defaultdict(float)
        
        for term in query_terms:
            if term not in idf_dict:
                continue  # Skip terms not in the corpus IDF dictionary
            idf = idf_dict[term]
            
            # Use only relevant documents containing the term
            for doc_id in tf_dict.get(term, {}):
                term_freq = tf_dict[doc_id].get(term, 0)
                doc_len = get_doc_length(corpus.loc[doc_id, "preprocessed_text"])

                # BM25 term score calculation
                numerator = term_freq * (k1 + 1)
                denominator = term_freq + k1 * (1 - b + b * (doc_len / avgdl))
                scores[doc_id] += idf * (numerator / denominator)
        
        # Return top 10 documents sorted by score
        ranked_docs = sorted(scores, key=scores.get, reverse=True)[:10]
        return query_id, ranked_docs

    # Process each query without creating additional copies of large objects
    for _, row in tqdm(train_query.iterrows(), desc="Ranking documents", total=len(train_query)):
        query_id = row["query_id"]
        query_terms = row["preprocessed_query"].split()
        query_result = score_single_query(query_id, query_terms)
        ranked_documents_dict[query_result[0]] = query_result[1]

    return ranked_documents_dict


# Load and preprocess queries
def load_and_preprocess_queries(path_to_train_query):
    train_query = pd.read_csv(path_to_train_query)
    train_query["preprocessed_query"] = train_query["query"].apply(preprocess)
    return train_query


# Execution flow
print("Loading and preprocessing corpus...")

# Load the preprocessed corpus
with open(path_to_save_df + "preprocessed_text_df.pkl", "rb") as f:
    corpus = pickle.load(f)

print("Computing TF, DF, and avgdl...")
if os.path.exists(path_to_saved_file + "tf_dict.pkl"):
    with open(path_to_saved_file + "tf_dict.pkl", "rb") as f:
        tf_dict = pickle.load(f)
    with open(path_to_saved_file + "df_dict.pkl", "rb") as f:
        df_dict = pickle.load(f)
    with open(path_to_saved_file + "avgdl.pkl", "rb") as f:
        avgdl = pickle.load(f)
    with open(path_to_saved_file + "num_docs.pkl", "rb") as f:
        num_docs = pickle.load(f)
else:
    tf_dict, df_dict, avgdl, num_docs = compute_tf_df_and_avgdl(corpus)

print("Computing IDF scores...")
idf_dict = compute_idf(df_dict, num_docs)

print("Loading and preprocessing queries...")

print("Sample tf_dict:", list(tf_dict.items())[:5])
print("Sample idf_dict:", list(idf_dict.items())[:5])
print("Sample corpus text:", corpus.loc[0, "preprocessed_text"])

exit()

if os.path.exists(path_to_save_df + "preprocessed_train_query.pkl"):
    train_query = pd.read_pickle(path_to_save_df + "preprocessed_train_query.pkl")
else:
    # Load the train queries as a DataFrame
    train_query = pd.read_csv(path_to_train_query)
    batch_size = 1000
    num_batches = len(train_query) // batch_size + 1
    for i in range(num_batches):
        batch_df = train_query.iloc[i * batch_size : (i + 1) * batch_size]
        process_batch(batch_df, path_to_save_df, i)

    preprocessed_train_query = []
    for i in range(num_batches):
        batch_file_path = os.path.join(
            path_to_save_df, f"preprocessed_train_query{i}.pkl"
        )
        preprocessed_train_query.append(pd.read_pickle(batch_file_path))
    preprocessed_train_query = pd.concat(preprocessed_train_query, ignore_index=True)
    preprocessed_train_query.to_pickle(
        os.path.join(path_to_save_df, "preprocessed_train_query.pkl")
    )

print("Ranking documents for each query using BM25...")
ranked_documents_dict = rank_documents_with_bm25_optimized(
    corpus, train_query, tf_dict, idf_dict, avgdl
)

# Save ranked documents as CSV
print(ranked_documents_dict)
exit()
ranked_documents_df = pd.DataFrame(ranked_documents_dict).T
ranked_documents_df.columns = [f"doc_{i}" for i in range(1, 11)]
ranked_documents_df.index.name = "query_id"
ranked_documents_df.to_csv(os.path.join(path_to_save_df, "ranked_documents_bm25.csv"))

print(
    "Ranking completed. The ranked documents are saved as 'ranked_documents_bm25.csv'."
)
