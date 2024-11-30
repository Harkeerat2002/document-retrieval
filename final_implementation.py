import pandas as pd
import numpy as np
import math
from collections import defaultdict
import pickle
import os
import gc
import nltk
from konlpy.tag import Okt
import joblib
from nltk.stem import SnowballStemmer
import tqdm
import string
from nltk.util import ngrams
from joblib import dump
import time
import multiprocessing as mp
from joblib import dump, load
import concurrent.futures
import multiprocessing
import gzip

# nltk.download("punkt", quiet=True)
# nltk.download("stopwords", quiet=True)
# nltk.download("wordnet", quiet=True)

# stemmers and tokenizers


stemmer_dict = {
    "fr": SnowballStemmer("french"),
    "de": SnowballStemmer("german"),
    "es": SnowballStemmer("spanish"),
    "it": SnowballStemmer("italian"),
    "en": SnowballStemmer("english"),
}


# Load stopwords
def load_stopwords(languages=["english", "french", "german", "spanish", "italian"]):
    stop_words = set()
    for lang in languages:
        try:
            stop_words.update(nltk.corpus.stopwords.words(lang))
        except:
            pass
    return stop_words


# print("Loading stopwords...")



def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def preprocess_text(text, lang):
    okt = Okt()
    lemmatizer = nltk.WordNetLemmatizer()
    stop_words = load_stopwords()
    if not isinstance(text, str):
        text = ""
    text = text.translate(str.maketrans("", "", string.punctuation))

    if lang in ["en", "fr", "de", "es", "it"]:
        tokens = nltk.word_tokenize(text)
    elif lang == "ko":
        tokens = okt.morphs(text)
    else:
        tokens = text.split()

    tokens = [word for word in tokens if word.lower() not in stop_words]

    if lang == "en":
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    elif lang in ["fr", "de", "es", "it"]:
        stemmer = stemmer_dict.get(lang, None)
        if stemmer:
            tokens = [stemmer.stem(word) for word in tokens]

    if lang in ["fr", "de", "es", "it"] and len(tokens) >= 2:
        n_grams = ["_".join(gram) for gram in ngrams(tokens, 2)]
        tokens = tokens + n_grams

    cleaned_text = " ".join(tokens)
    return cleaned_text


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0.0
        self.avgdl = 0.0
        self.df = defaultdict(int)
        self.idf = {}
        self.inverted_index = defaultdict(list)
        self.term_freqs = []
        #self.build(tokenized_corpus)
        self.doc_lengths = 0
        self.precomputed_idf = 0

    def build(self, tokenized_corpus, lang):
        for doc_id, document in enumerate(tokenized_corpus):
            freq = defaultdict(int)
            for word in document:
                freq[word] += 1
            self.term_freqs.append(freq)
            for word in freq.keys():
                self.df[word] += 1
                self.inverted_index[word].append(doc_id)

        for word, freq in self.df.items():
            self.idf[word] = math.log(
                1 + (self.corpus_size - freq + 0.5) / (freq + 0.5)
            )


    def precompute_doc_lengths(self):
        return {doc_id: sum(self.term_freqs[doc_id].values()) for doc_id in tqdm.tqdm(range(self.corpus_size))}

    def precompute_idf(self):
        return {word: self.idf[word] for word in tqdm.tqdm(self.idf.keys())}

    def calculate_scores(self, query):
        scores = np.zeros(self.corpus_size)
        unique_query_terms = set(query)
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl

        for word in unique_query_terms:
            if word not in self.precomputed_idf:
                continue
            idf = self.precomputed_idf[word]
            doc_ids = self.inverted_index[word]
            for doc_id in doc_ids:
                tf = self.term_freqs[doc_id][word]
                dl = self.doc_lengths[doc_id]
                score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + dl / avgdl)))
                scores[doc_id] += score

        return scores

    def retrieve_top_n(self, query, n=10):
        scores = self.calculate_scores(query)
        if n >= len(scores):
            top_n_indices = np.argsort(scores)[::-1]
        else:
            top_n_indices = np.argpartition(scores, -n)[-n:]
            top_n_indices = top_n_indices[np.argsort(scores[top_n_indices])[::-1]]
        return top_n_indices


class HybridSearch:
    def __init__(self):
        pass

    def load_preprocessed_queries(self):
        queries_path = f"preprocessed_data/preprocessed_test_queries.pkl"
        langs_path = f"preprocessed_data/test_query_langs.pkl"
        test_path = "data/test.csv"

        preprocessed_q = []

        test_cv = pd.read_csv(test_path)
        for query, lang in tqdm.tqdm(zip(test_cv["query"], test_cv["lang"])):
            preprocessed_q.append(preprocess_text(query, lang))

        print(len(preprocessed_q))

        query_langs = load_pickle(langs_path)

        return preprocessed_q, query_langs


def retrieve_top_n_batch(args):
    bm25_model, tokenized_query_batch, k = args
    return [bm25_model.retrieve_top_n(query, n=k) for query in tqdm.tqdm(tokenized_query_batch)]

def retrieve_top_n_batch_shared(args):
    bm25_model, tokenized_query_batch, k, shared_doc_ids = args
    return [[shared_doc_ids[idx] for idx in bm25_model.retrieve_top_n(query, n=k)] for query in tokenized_query_batch]

def save_in_batches_idf(data, path, lang, batch_size=1000000):
    os.makedirs(path, exist_ok=True)
    keys = list(data.keys())

    for i in tqdm.tqdm(range(0, len(keys), batch_size)):
        batch_keys = keys[i:i + batch_size]
        batch = {key: data[key] for key in batch_keys}
        with open(os.path.join(path, f'batch_idf{i // batch_size}_{lang}.pkl'), 'wb') as f:
            pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)
            
def save_in_batches_id(data, path, lang, batch_size=100000):
    os.makedirs(path, exist_ok=True)
    keys = list(data.keys())
    
    if lang == "en":
        batch = 10000

    for i in tqdm.tqdm(range(0, len(keys), batch_size)):
        batch_keys = keys[i:i + batch_size]
        batch = {key: data[key] for key in batch_keys}
        with open(os.path.join(path, f'batch_inverse{i // batch_size}_{lang}.pkl'), 'wb') as f:
            pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            

def save_in_batches(data, path, lang, batch_size=1000):
    os.makedirs(path, exist_ok=True)
    if lang == "en":
        batch_size = 1000
    tf = data.term_freqs
    for i in tqdm.tqdm(range(0, len(tf), batch_size)):
        batch = tf[i:i + batch_size]
        with open(os.path.join(path, f'batch_{i // batch_size}_{lang}.pkl'), 'wb') as f:
            pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)
        

def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_model_picklebatches(path, lang, batch_size=10):
    batch_per_lang = {"ar": 9, "de": 11, "en": 208, "es": 12, "fr": 11, "it": 12, "ko": 8}
    
    if lang == "en":
        batch_size = 10
    else:
        batch_size = 5
    
    term_freqs = []
    file_paths = [f"{path}/batch_{i}_{lang}.pkl" for i in range(batch_per_lang[lang])]
    
    # Split file paths into smaller batches
    file_batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
    
    for file_batch in tqdm.tqdm(file_batches):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(load_pickle_file, file_batch))
        
        for result in results:
            term_freqs.extend(result)
    
    return term_freqs

def load_model_picklebatches_idf(path, lang, batch_size=20):
    batch_per_lang = {"ar": 2, "de": 19, "en": 6, "es": 14, "fr": 16, "it": 17, "ko": 1}
    
    
    idf = {}
    file_paths = [f"{path}/batch_idf{i}_{lang}.pkl" for i in range(batch_per_lang[lang])]
    
    # Split file paths into smaller batches
    file_batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
    
    for file_batch in tqdm.tqdm(file_batches, desc=f"Loading {lang} IDF"):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(load_pickle_file, file_batch))
        
        for result in results:
            idf.update(result)
            
    return idf


def load_model_picklebatches_id(path, lang, batch_size=20):
    batch_per_lang = {"ar": 2, "de": 19, "en": 6, "es": 14, "fr": 16, "it": 17, "ko": 1}
    
    inverted_index = defaultdict(list)
    file_paths = [f"{path}/batch_inverse{i}_{lang}.pkl" for i in range(batch_per_lang[lang])]
    
    # Split file paths into smaller batches
    file_batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
    
    for file_batch in tqdm.tqdm(file_batches, desc=f"Loading {lang} inverted index"):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(load_pickle_file, file_batch))
        
        for result in results:
            for key, value in result.items():
                inverted_index[key].extend(value)
                
    return inverted_index


def retrieve_test_queries_optimized(preprocessed_queries, query_langs, k=10):
    retrieved_docs = [None] * len(preprocessed_queries)
    queries_df = pd.DataFrame(
        {
            "query": preprocessed_queries,
            "lang": query_langs,
            "original_idx": range(len(preprocessed_queries)),
        }
    )
    
    avgdl_dict = {
        "ar": 4418.960584437648,
        "de": 5575.8404294032025,
        "en": 1339.3939950714448,
        "es": 6174.31717941737,
        "fr": 6834.264518546272,
        "it": 6483.073170731707,
        "ko": 4380.481946028126
    }
    
    corpus_size_dict = {
        "ar": 8829,
        "de": 10992,
        "en": 207363,
        "es": 11019,
        "fr": 10676,
        "it": 11250,
        "ko": 7893
    }
    
    grouped = queries_df.groupby("lang")

    for lang, group in tqdm.tqdm(grouped):
        lang_queries = group["query"].tolist()
        lang_original_indices = group["original_idx"].tolist()
        lang_doc_ids_path = f"preprocessed_data/doc_ids_{lang}.pkl"
        lang_bm25_model_path = f"./bm25_model_{lang}.pkl"
        
        lang_bm25_model_path_joblib = "./batches/"
            
        

        # Load model with memory mapping
        print(f"Loading {lang_bm25_model_path}")
        start_time = time.time()
        term_freqs = load_model_picklebatches(lang_bm25_model_path_joblib, lang)
        
        # with open(lang_bm25_model_path, "rb") as f:
        #     bm25_model = pickle.load(f)
            
    
        
        # IDF = bm25_model.idf
        # inverted_index = bm25_model.inverted_index
        # print(f"IDF: {len(IDF)}")
        # print(f"Inverted Index: {len(inverted_index)}")
        
        start_time = time.time()
        # Load the IDF and inverted index
        with open(f"IDF_{lang}.pkl", "rb") as f:
            idfc = pickle.load(f)
        end_time = time.time() - start_time
        
        print(f"IDF load time uncompressed: {end_time}")
        
        # # save idf as a gzip pickle
        # start_time = time.time()
        # with gzip.open(f"IDF_{lang}.pkl.gz", "rb") as f:
        #     idf = pickle.load(f)
        # end_time = time.time() - start_time
        
        # print(f"IDF load time compressed: {end_time}")
        
        
        start_time = time.time()
        with open(f"Inverted_index_{lang}.pkl", "rb") as f:
            inverted_indexc = pickle.load(f)
        end_time = time.time() - start_time
        print(f"Inverted Index load time uncompressed: {end_time}")
        
        start_time = time.time()
            
        # start_time = time.time()
        # idf = load_model_picklebatches_idf("batches_idf", lang)
        # end_time = time.time() - start_time
        # print(f"IDF load time: {end_time}")
        
        # start_time = time.time()
        # inverted_index = load_model_picklebatches_id("batches_inverse", lang)
        # end_time = time.time() - start_time
        # print(f"Inverted Index load time: {end_time}")
        
        save_in_batches_idf(idfc, "batches_idf", lang)
        save_in_batches_id(inverted_indexc, "batches_inverse", lang)
        
        # if idf == idfc:
        #     print("IDF is same")
        # else:
        #     print("IDF is different")
        #     break
            
        # if inverted_index == inverted_indexc:
        #     print("Inverted Index is same")
        # else:
        #     print("Inverted Index is different")
        #     break
            
        # start_time = time.time()
        # with gzip.open(f"Inverted_index_{lang}.pkl.gz", "rb") as f:
        #     inverted_index = pickle.load(f)
        # end_time = time.time() - start_time
        # print(f"Inverted Index load time compressed: {end_time}")
        
        # # save_in_batches_id(inverted_index, "batches_inverse/", lang)
        # print("Loading IDF Values")
        # idf = load_model_picklebatches_idf("batches_idf/", lang)
        # print("Loading Inverted Index")
        # inverted_index = load_model_picklebatches_id("batches_inverse/", lang)
        
        
        # if idf == bm25_model.idf:
        #     print("IDF is same")
        # else:
        #     print("IDF is different")
            
        # if inverted_index == bm25_model.inverted_index:
        #     print("Inverted Index is same")
        # else:
        #     print("Inverted Index is different")
        
        # # exit()
        # # # # Confirm that the bm25_model.term_freqs is same as the loaded term_freqs
        # # assert bm25_model.term_freqs == term_freqs
            
        # # start_time = time.time()
        # # model = load_model_mmap(lang_bm25_model_path_joblib)
        # # end_time = time.time() - start_time
        # # print(f"Joblib load time: {end_time}")
        # # exit()
            
        # # save_in_batches(bm25_model, lang_bm25_model_path_joblib, lang)
            
            
        # # print("Langugae: ", lang)
        # # print("Term-Freqs: ", len(bm25_model.term_freqs))
        # # print("corpus_size: ", bm25_model.corpus_size)
        # # print("avgdl: ", bm25_model.avgdl)
        # # exit()
        
        # # start_time = time.time()
        # # # Resave the bm25 model as memory map with compression
        # # save_model_compressed(bm25_model, lang_bm25_model_path_joblib)
        # # end_time = time.time() - start_time
        # # print(f"Joblib dump time: {end_time}")

        # # start_time = time.time()
        # # # Read the model from memory map
        # # bm25_model = load_model_mmap(lang_bm25_model_path_joblib)
        # # end_time = time.time() - start_time
        # # print(f"Joblib load time: {end_time}")
        
        # bm25_model = BM25()
        # bm25_model.term_freqs = term_freqs
        # bm25_model.corpus_size = corpus_size_dict[lang]
        # bm25_model.avgdl = avgdl_dict[lang]
        # bm25_model.idf = idf
        # bm25_model.inverted_index = inverted_index
        # bm25_model.doc_lengths = bm25_model.precompute_doc_lengths()
        # bm25_model.precomputed_idf = bm25_model.precompute_idf()
        # end_time = time.time() - start_time
        # print(f"BM25 model load time: {end_time}")

        # # bm25_model.doc_lengths = bm25_model.precompute_doc_lengths()
        # # bm25_model.precomputed_idf = bm25_model.precompute_idf()
        # # pickle_load_time = time.time() - start_time
        # # print(f"Joblib load time: {pickle_load_time}")

        
        
        # # bm25_model.doc_lengths = bm25_model.precompute_doc_lengths()
        # # bm25_model.precomputed_idf = bm25_model.precompute_idf()
        # # pickle_load_time = time.time() - start_time

        # # Load document IDs with memory mapping
        # with open(lang_doc_ids_path, "rb") as f:
        #     doc_ids = pickle.load(f)

        # batch_size = 30
        # if lang == "en":
        #     batch_size = 80
        # tokenized_queries = [query.split() for query in lang_queries]
        # num_batches = math.ceil(len(tokenized_queries) / batch_size)
        # batches = [tokenized_queries[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
        # original_idx_batches = [lang_original_indices[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

        # # Retrieve top-k documents for each batch of queries
        # print(f"Retrieving top-{k} documents for {lang} queries...")
        # start_time = time.time()

        # # Process batches sequentially
        # results = []
        # for batch in batches:
        #     result = retrieve_top_n_batch((bm25_model, batch, k))
        #     results.append(result)

        # retrieval_time = time.time() - start_time
        # print(f"Retrieval time: {retrieval_time}")
        
        # # Assign retrieved documents to original indices
        # for batch_results, original_indices in zip(results, original_idx_batches):
        #     for i, result in enumerate(batch_results):
        #         retrieved_docs[original_indices[i]] = [doc_ids[idx] for idx in result]

        # del bm25_model, doc_ids, lang_queries, tokenized_queries
        # gc.collect()
        
    return retrieved_docs


def main():
    # Load test data and preprocessed queries
    preprocessed_test_queries, test_query_langs = (
        HybridSearch().load_preprocessed_queries()
    )
    
    print(preprocessed_test_queries[:5])
    print(test_query_langs[:5])

    # Perform retrieval on test set
    retrieved_docs_test = retrieve_test_queries_optimized(
        preprocessed_queries=preprocessed_test_queries,
        query_langs=test_query_langs,
        k=10,
    )
    
    print(retrieved_docs_test[:5])

    submission_df = pd.DataFrame(
        {"id": np.arange(len(retrieved_docs_test)), "docids": retrieved_docs_test}
    )
    submission_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
