import pandas as pd
import numpy as np
import os
import string
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json
from scipy.sparse import lil_matrix, save_npz, load_npz


# Paths
path_to_save_df = "./data/pd_df/"
path_to_train_query = "./data/train.csv"
corpus_file_path = "./data/corpus.json/corpus.json"
tokenized_text_path = os.path.join(path_to_save_df, "tokenized_text.pkl")
tokenized_query_path = os.path.join(path_to_save_df, "tokenized_query.pkl")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# HELPER FUNCTIONS
def preprocess(text):
    # Lowercasing, punctuation removal, stopword removal, and stemming
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

def process_batch(batch_df, path_to_save_df, batch_index):
    preprocessed_data = Parallel(n_jobs=-1)(
        delayed(lambda row: (row["docid"], preprocess(row["text"])))(row)
        for _, row in tqdm(batch_df.iterrows(), total=len(batch_df))
    )
    preprocessed_text_df = pd.DataFrame(preprocessed_data, columns=["doc_id", "preprocessed_text"])
    batch_file_path = os.path.join(path_to_save_df, f"preprocessed_text_df_batch_{batch_index}.pkl")
    preprocessed_text_df.to_pickle(batch_file_path)

def compute_tf_idf(corpus=None):
    # Term Frequency (TF)
    tf_dict = defaultdict(lambda: defaultdict(int))
    df_dict = defaultdict(int)
    idf_dict = defaultdict(float)
    total_documents = 268022
    
    # STEP 1: Compute TF and DF
    if os.path.exists(os.path.join(path_to_save_df, "tf_dict.json")) and os.path.exists(os.path.join(path_to_save_df, "df_dict.json")):
        print("Loading the precomputed TF dictionary...")
        with open(os.path.join(path_to_save_df, "tf_dict.json"), "r") as f:
            tf_dict = json.load(f)
        # print("Loading the precomputed DF dictionary...")
        # with open(os.path.join(path_to_save_df, "df_dict.json"), "r") as f:
        #     df_dict = json.load(f)
    else:
        print("Computing TF and DF...")
        for doc_id, doc in tqdm(corpus.items(), desc="Processing documents"):
            words = doc.split()
            word_count = len(words)
            unique_words = set(words)
            for word in words:
                tf_dict[doc_id][word] += 1 / word_count
            for word in unique_words:
                df_dict[word] += 1
        # Save STEP 1 results
        with open(os.path.join(path_to_save_df, "tf_dict.json"), "w") as f:
            json.dump(tf_dict, f)
        with open(os.path.join(path_to_save_df, "df_dict.json"), "w") as f:
            json.dump(df_dict, f)
    

    
    
    # STEP 2: Inverse Document Frequency (IDF)
    if os.path.exists(os.path.join(path_to_save_df, "idf_dict.json")):
        print("Loading the precomputed IDF dictionary...")
        with open(os.path.join(path_to_save_df, "idf_dict.json"), "r") as f:
            idf_dict = json.load(f)
    else:
        print("Computing IDF...")
    
        for word, count in tqdm(df_dict.items(), desc="Calculating IDF"):
            idf_dict[word] = np.log((total_documents + 1) / (count + 1)) + 1

        # Save the IDF dictionary
        with open(os.path.join(path_to_save_df, "idf_dict.json"), "w") as f:
            json.dump(idf_dict, f)
    
    
    # STEP 3: TF-IDF
    tf_idf_dict = defaultdict(lambda: defaultdict(float))
    for doc_id, word_tf in tqdm(tf_dict.items(), desc="Calculating TF-IDF"):
        for word, tf in word_tf.items():
            tf_idf_dict[doc_id][word] = tf * idf_dict[word]
    
    return tf_idf_dict

def vectorize_query(query, tf_idf_dict, vocab):
    query_vector = np.zeros(len(vocab))
    for word in query.split():
        if word in vocab:
            tf = query.count(word) / len(query.split())
            print(f"TF for word '{word}': {tf}")
            idf = np.log(len(tf_idf_dict) / (1 + sum(1 for doc in tf_idf_dict.values() if word in doc)))
            print(f"IDF for word '{word}': {idf}")
            query_vector[vocab[word]] = tf * idf
        else:
            print(f"Warning: Word '{word}' not found in vocabulary.")
    return query_vector

def logistic_regression_fit(X, y, epochs=1000, lr=0.01):
    # Logistic Regression: Initialize weights and use gradient descent
    n, d = X.shape
    weights = np.zeros(d)
    bias = 0

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        # Linear combination: z = Xw + b
        linear_model = np.dot(X, weights) + bias
        # Sigmoid: p = 1 / (1 + exp(-z))
        probabilities = 1 / (1 + np.exp(-linear_model))
        
        # Gradient descent: update weights and bias
        gradient_weights = np.dot(X.T, (probabilities - y)) / n
        gradient_bias = np.sum(probabilities - y) / n
        
        weights -= lr * gradient_weights
        bias -= lr * gradient_bias
    
    return weights, bias

def predict(X, weights, bias):
    # Linear combination
    linear_model = np.dot(X, weights) + bias
    # Apply sigmoid to get probability
    probabilities = 1 / (1 + np.exp(-linear_model))
    return probabilities

def rank_documents(query_vector, doc_vectors):
    # Compute dot product (cosine similarity) between query vector and document vectors
    similarities = np.dot(doc_vectors, query_vector)
    top_doc_indices = np.argsort(similarities)[::-1][:10]
    return top_doc_indices

## LOAD THE CORPUS
if os.path.exists(os.path.join(path_to_save_df, "corpus_df.pkl")):
    print("Loading the preprocessed corpus DataFrame...")
    # corpus_df = pd.read_pickle(os.path.join(path_to_save_df, "corpus_df.pkl"))
else:
    corpus_df = pd.read_json(corpus_file_path)
    corpus_df.to_pickle(os.path.join(path_to_save_df, "corpus_df.pkl"))

## PREPROCESS THE CORPUS
if os.path.exists(os.path.join(path_to_save_df, "preprocessed_text_df.pkl")):
    print("Loading the preprocessed text DataFrame...")
    # preprocessed_text_df = pd.read_pickle(os.path.join(path_to_save_df, "preprocessed_text_df.pkl"))
else:
    batch_size = 10000
    num_batches = len(corpus_df) // batch_size + 1
    for i in range(num_batches):
        batch_df = corpus_df.iloc[i * batch_size: (i + 1) * batch_size]
        process_batch(batch_df, path_to_save_df, i)

    preprocessed_text_dfs = []
    for i in range(num_batches):
        batch_file_path = os.path.join(path_to_save_df, f"preprocessed_text_df_batch_{i}.pkl")
        preprocessed_text_dfs.append(pd.read_pickle(batch_file_path))
    
    preprocessed_text_df = pd.concat(preprocessed_text_dfs, ignore_index=True)
    preprocessed_text_df.to_pickle(os.path.join(path_to_save_df, "preprocessed_text_df.pkl"))

## CREATE VOCABULARY
print("Creating the vocabulary...")

# Load the preprocessed text DataFrame
if os.path.exists(os.path.join(path_to_save_df, "corpus.json")):
    print("Loading the corpus...")
    with open(os.path.join(path_to_save_df, "corpus.json"), "r") as f:
        corpus = json.load(f)
else:
    print("Creating the corpus...")
    corpus = {row["doc_id"]: row["preprocessed_text"] for _, row in tqdm(preprocessed_text_df.iterrows(), desc="Processing corpus")}
    # Save the corpus as a JSON file
    with open(os.path.join(path_to_save_df, "corpus.json"), "w") as f:
        json.dump(corpus, f)

# Load the TF-IDF dictionary
if os.path.exists(os.path.join(path_to_save_df, "tf_idf_dict.json")):
    print("Loading the TF-IDF dictionary...")
    with open(os.path.join(path_to_save_df, "tf_idf_dict.json"), "r") as f:
        tf_idf_dict = json.load(f)
else:
    print("Computing the TF-IDF dictionary...")
    tf_idf_dict = compute_tf_idf(corpus)
    # Save the TF-IDF dictionary
    with open(os.path.join(path_to_save_df, "tf_idf_dict.json"), "w") as f:
        json.dump(tf_idf_dict, f)
        
# Load the IDF dictionary
if os.path.exists(os.path.join(path_to_save_df, "idf_dict.json")):
    print("Loading the precomputed IDF dictionary...")
    with open(os.path.join(path_to_save_df, "idf_dict.json"), "r") as f:
        idf_dict = json.load(f)
else:
    print("Computing IDF...")
    idf_dict = {}
    total_documents = len(corpus)
    df_dict = defaultdict(int)
    for doc in corpus.values():
        for word in set(doc.split()):
            df_dict[word] += 1
    for word, count in tqdm(df_dict.items(), desc="Calculating IDF"):
        idf_dict[word] = np.log((total_documents + 1) / (count + 1)) + 1
    # Save the IDF dictionary
    with open(os.path.join(path_to_save_df, "idf_dict.json"), "w") as f:
        json.dump(idf_dict, f)

# Load the vocabulary
if os.path.exists(os.path.join(path_to_save_df, "vocab.pkl")):
    print("Loading the vocabulary...")
    vocab = pd.read_pickle(os.path.join(path_to_save_df, "vocab.pkl"))
else:
    print("Building the vocabulary...")
    all_words = set()
    for tf_idf in tqdm(tf_idf_dict.values(), desc="Collecting words from TF-IDF"):
        for word in tf_idf.keys():
            all_words.add(word)
    # Create the vocabulary with a progress bar
    vocab = {word: idx for idx, word in enumerate(tqdm(all_words, desc="Building vocabulary"))}
    # Save the vocabulary
    vocab_df = pd.DataFrame(list(vocab.items()), columns=["word", "index"])
    vocab_df.to_pickle(os.path.join(path_to_save_df, "vocab.pkl"))


## CONVERT CORPUS TO TF-IDF MATRIX
# Create a mapping from document IDs to integer indices
print("Creating the doc_id to index mapping...")
doc_id_to_index = {doc_id: idx for idx, doc_id in tqdm(enumerate(corpus.keys()), total=len(corpus), desc="Mapping doc IDs")}

# Convert corpus to TF-IDF matrix
if os.path.exists(os.path.join(path_to_save_df, "doc_vectors.npz")):
    print("Loading the precomputed doc_vectors...")
    doc_vectors = load_npz(os.path.join(path_to_save_df, "doc_vectors.npz"))
else:
    print("Converting the corpus to TF-IDF matrix...")
    doc_vectors = lil_matrix((len(doc_id_to_index), len(vocab)))
    for doc_id, tf_idf in tqdm(tf_idf_dict.items(), desc="Processing TF-IDF"):
        for word, value in tf_idf.items():
            if word in vocab:
                doc_vectors[doc_id_to_index[doc_id], vocab[word]] = value
            else:
                print(f"Warning: Word '{word}' not found in vocabulary.")
    # Save the doc_vectors as a sparse matrix
    save_npz(os.path.join(path_to_save_df, "doc_vectors.npz"), doc_vectors.tocsr())
# Load queries and preprocess them
print("Loading and preprocessing the queries...")
train_query = pd.read_csv(path_to_train_query)
train_query["preprocessed_query"] = train_query["query"].apply(preprocess)

# Prepare training data for Logistic Regression
print("Preparing training data for logistic regression...")
def process_row(row, vocab, idf_dict):
    query_vector = vectorize_query(row["preprocessed_query"], vocab, idf_dict)
    query_vectors = [query_vector]
    labels = [1]  # Positive label for the positive document
    negative_docs = json.loads(row["negative_docs"])  # Use json.loads instead of eval
    for _ in negative_docs:
        query_vectors.append(query_vector)
        labels.append(0)  # Negative label for the negative documents
    return query_vectors, labels

results = Parallel(n_jobs=-1)(delayed(process_row)(row, vocab, idf_dict) for _, row in tqdm(train_query.iterrows(), desc="Processing training queries", total=len(train_query)))

# Combine results
query_vectors = np.vstack([item for sublist in results for item in sublist[0]])
labels = np.hstack([item for sublist in results for item in sublist[1]])

# Train Logistic Regression
print("Training the logistic regression model...")
weights, bias = logistic_regression_fit(query_vectors, labels)

# Save the weights and bias
np.save(os.path.join(path_to_save_df, "weights.npy"), weights)
np.save(os.path.join(path_to_save_df, "bias.npy"), bias)



# Rank Documents for Each Query
print("Ranking documents for each query...")
ranked_documents_dict = {}
for query_id, query in tqdm(zip(train_query["query_id"], train_query["preprocessed_query"]), desc="Processing queries", total=len(train_query["query_id"])):
    query_vector = vectorize_query(query, tf_idf_dict, vocab)
    top_doc_indices = rank_documents(query_vector, doc_vectors)
    ranked_documents_dict[query_id] = top_doc_indices

# Save the ranked documents as a CSV with query_id and top 10 document ids
ranked_documents_df = pd.DataFrame(ranked_documents_dict).T
ranked_documents_df.columns = [f"doc_{i}" for i in range(1, 11)]
ranked_documents_df.index.name = "query_id"
ranked_documents_df.to_csv(os.path.join(path_to_save_df, "ranked_documents.csv"))

print("Ranking completed. The ranked documents are saved as 'ranked_documents.csv'.")
