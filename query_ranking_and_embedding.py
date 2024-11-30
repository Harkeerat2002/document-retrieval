import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pickle
import os
from scipy.sparse import lil_matrix, vstack, save_npz, load_npz, csr_matrix
import gc
from collections import Counter

from joblib import Parallel, delayed
from scipy.sparse.linalg import norm
from nltk.corpus import stopwords
from ko_ww_stopwords.stop_words import ko_ww_stop_words
from nltk.stem import PorterStemmer
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
from collections import defaultdict
import pickle
import os
import json
from tqdm import tqdm
import nltk
from konlpy.tag import Okt
import joblib
import string
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.util import ngrams
import gc
from multiprocessing import Pool, cpu_count

# Paths
path_to_save_df = "./data/pd_df/"
path_to_saved_file = "./data/saved_files/"
path_to_train_query = "./data/train.csv"
corpus_file_path = "./data/corpus.json/corpus.json"
path_save_emb = "./data/saved_files/"

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

okt = Okt()
lemmatizer = WordNetLemmatizer()
stemmer_dict = {
    'fr': SnowballStemmer('french'),
    'de': SnowballStemmer('german'),
    'es': SnowballStemmer('spanish'),
    'it': SnowballStemmer('italian'),
    'en': SnowballStemmer('english')
}

def load_stopwords(languages=['english', 'french', 'german', 'spanish', 'italian']):
    stop_words = set()
    for lang in languages:
        stop_words.update(nltk.corpus.stopwords.words(lang))
    return stop_words

stop_words = load_stopwords(['english', 'french', 'german', 'spanish', 'italian'])

# Create path_to_save_df if it does not exist
if not os.path.exists(path_to_save_df):
    os.makedirs(path_to_save_df)


def preprocess(text, lang):
    if not isinstance(text, str):
        text = ""
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    if lang in ['en', 'fr', 'de', 'es', 'it']:
        tokens = nltk.word_tokenize(text)
    elif lang == 'ko':
        tokens = okt.morphs(text)
    else:
        tokens = text.split()
    
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    if lang == 'en':
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    elif lang in ['fr', 'de', 'es', 'it']:
        stemmer = stemmer_dict.get(lang, None)
        if stemmer:
            tokens = [stemmer.stem(word) for word in tokens]
    
    if lang in ['fr', 'de', 'es', 'it'] and len(tokens) >= 2:
        n_grams = ['_'.join(gram) for gram in ngrams(tokens, 2)]
        tokens = tokens + n_grams
    
    cleaned_text = ' '.join(tokens)
    return cleaned_text

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


# Manually create TF-IDF embeddings
def create_tfidf_embedding(corpus, tf_dict, idf_dict, term_index):
    # Structure of tfidf embeddings: [num_documents, num_terms]

    num_docs = len(corpus)
    num_terms = len(term_index)

    # Initialize data structures for sparse matrix
    data = []
    rows = []
    cols = []
    doc_ids = []
    embeddings = []

    for doc_id, text in tqdm(
        corpus[["docid", "preprocessed_text"]].values, desc="Creating TF-IDF embeddings"
    ):
        terms = text.split()
        term_freqs = Counter(terms)  # Count term frequencies once
        embedding = lil_matrix(
            (1, num_terms), dtype=np.float32
        )  # Initialize sparse matrix

        for term, freq in term_freqs.items():
            if term in term_index:
                term_tf = freq  # Term frequency
                term_idf = idf_dict.get(term, 0)  # Inverse document frequency
                tfidf_score = term_tf * term_idf  # TF-IDF score
                embedding[0, term_index[term]] = (
                    tfidf_score  # Update the sparse matrix. 0 is the row index
                )

                # # Append data for sparse matrix
                # data.append(tfidf_score)
                # rows.append(len(doc_ids))  # Current document index
                # cols.append(term_index[term])
        embeddings.append(embedding)
        doc_ids.append(doc_id)

    return vstack(embeddings), doc_ids


def save_embeddings_to_mmap(embeddings, mmap_file):
    save_npz(mmap_file, embeddings)


def load_embeddings_from_mmap(mmap_file):
    return load_npz(mmap_file).tocsr()


def generate_query_embedding(query_text, tf_dict, idf_dict, term_index):
    query_embedding = lil_matrix((1, len(term_index)), dtype=np.float32)
    for term in query_text.split():
        if term in term_index:
            query_embedding[0, term_index[term]] = idf_dict.get(term, 0)
    return query_embedding

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
    # Structure of df_dict: {term: df -> number of documents containing the term}
    idf_dict = {
        term: np.log((num_docs - df + 0.5) / (df + 0.5))
        for term, df in tqdm(df_dict.items(), desc="Computing IDF scores")
    }
    return idf_dict






# Function to preprocess and rank using cosine similarity
# BM25 Scoring
def bm25_score(query_terms, doc_id, tf_dict, idf_dict, avgdl, k1=1.5, b=0.75):
    score = 0
    doc_length = sum(tf_dict.get(term, {}).get(doc_id, 0) for term in query_terms)
    for term in query_terms:
        if term in tf_dict:
            term_tf = tf_dict[term].get(doc_id, 0)
            term_idf = idf_dict.get(term, 0)
            numerator = term_tf * (k1 + 1)
            denominator = term_tf + k1 * (1 - b + b * (doc_length / avgdl))
            score += term_idf * (numerator / denominator)
    return score

# Rank documents with cosine similarity, then BM25
def rank_documents_with_cosine_similarity_and_bm25(corpus, train_query, tf_dict, idf_dict, avgdl, batch_size=400):
    term_index = {term: idx for idx, term in enumerate(tf_dict.keys())}

    if os.path.exists(path_to_saved_file + "embeddings.npz"):
        embeddings = load_embeddings_from_mmap(path_to_saved_file + "embeddings.npz")
        with open(path_to_saved_file + "doc_ids.pkl", "rb") as f:
            doc_ids = pickle.load(f)
    else:
        embeddings, doc_ids = create_tfidf_embedding(corpus, tf_dict, idf_dict, term_index)
        save_embeddings_to_mmap(embeddings, path_to_saved_file + "embeddings.npz")
        with open(path_to_saved_file + "doc_ids.pkl", "wb") as f:
            pickle.dump(doc_ids, f)

    doc_norms = norm(embeddings, axis=1).reshape(-1, 1)
    normalized_embeddings = embeddings.multiply(1 / doc_norms)

    ranked_documents_dict = {}
    num_queries = len(train_query)

    for start in tqdm(range(0, num_queries, batch_size), desc="Ranking queries"):
        end = min(start + batch_size, num_queries)
        batch_queries = train_query[["id", "preprocessed_query"]].values[start:end]
        batch_query_embeddings = []
        for _, query_text in batch_queries:
            query_embedding = generate_query_embedding(query_text, tf_dict, idf_dict, term_index)
            query_norm = norm(query_embedding)
            batch_query_embeddings.append(query_embedding.multiply(1 / query_norm))

        batch_query_embeddings = csr_matrix(vstack(batch_query_embeddings))
        cosine_similarities = normalized_embeddings.dot(batch_query_embeddings.T).toarray()
        
        for i, (query_id, query_text) in enumerate(batch_queries):
            top_indices = np.argsort(cosine_similarities[:, i])[::-1][:200]
            query_terms = query_text.split()

            top_100_docs = [(doc_ids[idx], bm25_score(query_terms, doc_ids[idx], tf_dict, idf_dict, avgdl))
                            for idx in top_indices]
            top_10_bm25 = sorted(top_100_docs, key=lambda x: x[1], reverse=True)[:10]

            ranked_documents_dict[query_id] = [doc[0] for doc in top_10_bm25]

    return ranked_documents_dict


def load_and_preprocess_queries(path_to_train_query):
    train_query = pd.read_csv(path_to_train_query)
    for query, lang in tqdm(
        train_query[["query", "lang"]].values, desc="Preprocessing queries"
    ):
        train_query.loc[train_query["query"] == query, "preprocessed_query"] = (
            preprocess(query, lang)
        )
    return train_query


# Execution Flow
if __name__ == "__main__":
    
    # with open(path_to_save_df + "corpus_lang.pkl", "rb") as f:
    #         corpus_lang = pickle.load(f)
            
    # print(corpus_lang)
    # exit()
    BATCH_SIZE = 1000 
    # Load data and preprocess as in your original code
    if not os.path.exists(path_to_save_df):
        os.makedirs(path_to_save_df)

    if os.path.exists(path_to_save_df + "preprocessed_corpus.pkl"):
        print("Loading preprocessed corpus...")
        corpus_df = pd.read_pickle(path_to_save_df + "preprocessed_corpus.pkl")
    else:
        print("Loading and preprocessing corpus in batches...")
        corpus_df = pd.read_json(corpus_file_path)
        corpus_df = preprocess_corpus_in_batches(corpus_df, batch_size=BATCH_SIZE, n_jobs=cpu_count())
        corpus_df.to_pickle(path_to_save_df + "preprocessed_corpus.pkl")
        
        

    # Load preprocessed queries
    if os.path.exists(path_to_save_df + "preprocessed_train_query.pkl"):
        print("Loading preprocessed queries...")
        # train_query_df = pd.read_pickle(
        #     path_to_save_df + "preprocessed_train_query.pkl"
        # )
    else:
        print("Loading and preprocessing queries...")
        #train_query_df = load_and_preprocess_queries(path_to_train_query)

    # Load or compute TF, DF, avgdl, and IDF
    if os.path.exists(path_to_saved_file + "tf_dict.pkl") and os.path.exists(
        path_to_saved_file + "df_dict.pkl"
    ):
        print("Loading TF and DF dictionaries...")
        with open(path_to_saved_file + "tf_dict.pkl", "rb") as f:
            tf_dict = pickle.load(f)
        with open(path_to_saved_file + "df_dict.pkl", "rb") as f:
            df_dict = pickle.load(f)
        with open(path_to_saved_file + "avgdl.pkl", "rb") as f:
            avgdl = pickle.load(f)
    else:
        print("Computing TF and DF dictionaries...")
        tf_dict, df_dict, avgdl, num_docs = compute_tf_df_and_avgdl(
            corpus_df, path_to_saved_file
        )

    # Compute IDF scores
    if os.path.exists(path_to_saved_file + "idf_dict.pkl"):
        print("Loading IDF scores...")
        with open(path_to_saved_file + "idf_dict.pkl", "rb") as f:
            idf_dict = pickle.load(f)
    else:
        print("Computing IDF scores...")
        idf_dict = compute_idf(df_dict, len(corpus_df))
        # Save the idf_dict
        with open(path_to_saved_file + "idf_dict.pkl", "wb") as f:
            pickle.dump(idf_dict, f)

    # Compute the term index
    term_index = {term: idx for idx, term in enumerate(tf_dict.keys())}
    # freq = 50
    # term_index = {term: idx for term, idx in term_index.items() if df_dict[term] > freq}
    # term_index = {term: idx for idx, term in enumerate(term_index.keys())}

    del df_dict
    gc.collect()

    # Rank documents using FAISS
    print("Ranking documents using Cosine Similarity...")
    # ranked_docs = rank_documents_with_cosine_similarity(corpus_df, train_query_df, tf_dict, idf_dict)

    # Make a CSV with query_id and all the doc_ids in an array
    # ranked_docs_df = pd.DataFrame(ranked_docs.items(), columns=["id", "doc_ids"])
    # ranked_docs_df.to_csv(path_to_save_df + "ranked_docs.csv", index=False)

    path_to_test_query = "./data/dev.csv"

    # Preprocess the test queries
    test_query_df = load_and_preprocess_queries(path_to_test_query)
    

    ranked_docs = rank_documents_with_cosine_similarity_and_bm25(corpus_df, test_query_df, tf_dict, idf_dict, avgdl)
    
    

    # Make a CSV with query_id and all the doc_ids in an array
    ranked_docs_df = pd.DataFrame(ranked_docs.items(), columns=["id", "docids"])
    
    possitive_docs = list(test_query_df["positive_docs"])
    
    pos_do = 0
    
    for i in range(len(ranked_docs_df)):
        if possitive_docs[i] in ranked_docs_df["docids"][i]:
            pos_do += 1
            
            
    per = pos_do / len(ranked_docs_df)        
    print(f"Number of positive docs found: {pos_do} / {len(ranked_docs_df) }, Percentage: {per}")
    

    # # for the id remove everything except the number ID format q-en-0000
    # ranked_docs_df["id"] = ranked_docs_df["id"].str.extract(r"(\d+)")

    ranked_docs_df.to_csv(path_to_save_df + "submission.csv", index=False)

    print("Ranking complete!")
