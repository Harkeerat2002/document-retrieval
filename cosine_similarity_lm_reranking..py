import json
import pandas as pd
import numpy as np
import nltk
import string
import os
# Disable parallelism in tokenizers to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from collections import defaultdict, Counter
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
import heapq
import math

# Additional imports for modifications
from fast_langdetect import detect, detect_multilingual
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the data
with open('corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

percent = 1
corpus = corpus[:int(len(corpus) * percent)]

# Load train and dev data
dev_queries = pd.read_csv('dev.csv')

# Initialize device for torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize stop words and stemmer cache
stop_words_cache = {}
stemmer_cache = {}

def split_into_sentences(text):
    return text.split('.')
    

def preprocess_text(text):
    try:
        language = detect(text)["lang"]
    except:
        language = 'en'  # default to English if detection fails
    # possible languages :  English, French, German, Spanish, Italian, Arabic, and Korean
    if language not in ['en', 'fr', 'de', 'es', 'it', 'ar', 'ko']:
        language = 'en'
    # Load stop words for the detected language
    if language not in stop_words_cache:
        if language in stopwords.fileids():
            stop_words_cache[language] = set(stopwords.words(language))
        else:
            stop_words_cache[language] = set()
    stop_words = stop_words_cache[language]

    if language in ['en', 'fr', 'de', 'es', 'it']:
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words]
        return tokens

    # Tokenize using NLTK's word_tokenize with the appropriate language
    tokens = nltk.tokenize.wordpunct_tokenize(text)

    # Remove stop words and punctuation
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]

    return tokens

# Build inverted index
def build_inverted_index(docs):
    local_docid_to_text = {}
    local_inverted_index = defaultdict(dict)
    local_doc_lengths = {}
    for doc in docs:
        docid = doc['docid']
        text = doc['text']
        
        sentences = split_into_sentences(text)
        for idx, sentence in enumerate(sentences):
            sentence_id = f"{docid}_{idx}"
            local_docid_to_text[sentence_id] = sentence
            tokens = preprocess_text(sentence)
            if not tokens:
                continue  # Skip empty sentences
            local_doc_lengths[sentence_id] = len(tokens)
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                local_inverted_index[term][sentence_id] = count
    return local_inverted_index, local_doc_lengths, local_docid_to_text

# Determine the number of CPUs to use
n_cpus = min(mp.cpu_count() - 1, 64)
print(f"Using {n_cpus} cores")

# Split the corpus into batches
batches = np.array_split(corpus, n_cpus)

print("Building inverted index...")
results = process_map(build_inverted_index, batches, max_workers=n_cpus, chunksize=1, tqdm_class=tqdm)

inverted_index = defaultdict(dict)
doc_id_to_text = {}
doc_lengths = {}
for local_inverted_index, local_doc_lengths, local_docid_to_text in results:
    doc_id_to_text.update(local_docid_to_text)
    for term, doc_dict in local_inverted_index.items():
        if term not in inverted_index:
            inverted_index[term] = doc_dict
        else:
            inverted_index[term].update(doc_dict)
    doc_lengths.update(local_doc_lengths)

N = len(doc_lengths)
avg_doc_length = sum(doc_lengths.values()) / N

def get_original_docid(sentence_id):
    return sentence_id.split('_')[0]


print(f"Total documents: {N}")

def preprocess_query(query):
    # For queries, use the same preprocessing
    return preprocess_text(query)

print("Retrieving candidate documents for queries...")

# Combine train and dev queries for building embeddings
all_queries =  dev_queries # pd.concat([train_queries, dev_queries], ignore_index=True)

# Build a set of unique queries to avoid redundant computations
unique_queries = all_queries[['query_id', 'query', 'lang']].drop_duplicates()

# Preprocess all queries and store them
query_id_to_tokens = {}
for idx, row in tqdm(unique_queries.iterrows(), total=unique_queries.shape[0]):
    query_id = row['query_id']
    query_text = row['query']
    query_tokens = preprocess_query(query_text)
    query_id_to_tokens[query_id] = query_tokens

# Build candidate documents for dev queries
candidate_docs = {}
for idx, row in tqdm(dev_queries.iterrows(), total=dev_queries.shape[0]):
    query_id = row['query_id']
    query_tokens = query_id_to_tokens[query_id]
    # Get the set of documents that contain any of the query terms
    doc_sets = [set(inverted_index.get(token, {})) for token in query_tokens]
    # Union of all document sets
    candidate_docids = set().union(*doc_sets)
    candidate_docs[query_id] = (query_tokens, candidate_docids)

# Optionally limit the number of candidate documents per query
MAX_CANDIDATES = 1000  # Increase candidate pool size
for query_id in candidate_docs:
    query_tokens, candidate_docids = candidate_docs[query_id]
    if len(candidate_docids) > MAX_CANDIDATES:
        candidate_docids = set(list(candidate_docids)[:MAX_CANDIDATES])
    candidate_docs[query_id] = (query_tokens, candidate_docids)

args = [(query_id, query_tokens) for query_id, (query_tokens, _) in candidate_docs.items()]

def score_documents_for_query(args):
    query_id, query_tokens = args
    scores = defaultdict(float)
    doc_freqs = {}
    for term in query_tokens:
        if term in inverted_index:
            doc_freqs[term] = len(inverted_index[term])
        else:
            doc_freqs[term] = 0

    for term in query_tokens:
        if term in inverted_index:
            postings = inverted_index[term]
            df = doc_freqs[term]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            for docid, tf in postings.items():
                # BM25 parameters
                k1 = 1.5
                b = 0.75
                dl = doc_lengths[docid]
                score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_doc_length)))
                scores[docid] += score
    if scores:
        top_docs = heapq.nlargest(100, scores, key=scores.get)
    else:
        top_docs = []
    return query_id, top_docs

print("Scoring documents...")
results_list = process_map(score_documents_for_query, args, max_workers=n_cpus, chunksize=1, tqdm_class=tqdm)

# Collect initial retrieval results
initial_results = {query_id: top_docs for query_id, top_docs in results_list}

# Now, re-rank using the multilingual embeddings
print("Re-ranking with multilingual embeddings...")

# Choose a multilingual model
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean Pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().squeeze()

# Build embeddings for candidate documents
print("Building embeddings for sentence chunks...")
sentence_embeddings = {}
all_candidate_sentence_ids = set(sentence_id for sentence_id in doc_id_to_text.keys())
batch_size = 32
sentence_id_list = list(all_candidate_sentence_ids)

for i in tqdm(range(0, len(sentence_id_list), batch_size)):
    batch_sentence_ids = sentence_id_list[i:i+batch_size]
    texts = [doc_id_to_text[sentence_id] for sentence_id in batch_sentence_ids]
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
    for sentence_id, emb in zip(batch_sentence_ids, embeddings):
        sentence_embeddings[sentence_id] = emb

# Build embeddings for queries
print("Building embeddings for queries...")
query_embeddings = {}
query_ids = dev_queries['query_id'].unique()
query_texts = []
query_languages = []

for query_id in query_ids:
    query_row = dev_queries[dev_queries['query_id'] == query_id].iloc[0]
    query_texts.append(query_row['query'])
    query_languages.append(query_row['lang'])

for i in tqdm(range(0, len(query_ids), batch_size)):
    batch_query_ids = query_ids[i:i+batch_size]
    batch_texts = query_texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
    for query_id, emb in zip(batch_query_ids, embeddings):
        query_embeddings[query_id] = emb

print("Re-ranking with sentence embeddings...")
final_results = {}
k_values = [1, 5, 10]
mrr_scores = {k: [] for k in k_values}
recall_scores = {k: [] for k in k_values}

# Normalize sentence embeddings
for sentence_id in sentence_embeddings:
    sentence_embeddings[sentence_id] = sentence_embeddings[sentence_id] / (sentence_embeddings[sentence_id].norm() + 1e-10)

# For each query
for query_id in tqdm(dev_queries['query_id'].unique()):
    query_embedding = query_embeddings[query_id]
    query_embedding = query_embedding / (query_embedding.norm() + 1e-10)  # Normalize
    candidate_sentence_ids = list(sentence_embeddings.keys())

    # Compute similarities
    sentence_embs = torch.stack([sentence_embeddings[sentence_id] for sentence_id in candidate_sentence_ids])
    similarities = torch.matmul(sentence_embs, query_embedding)
    top_k_indices = torch.topk(similarities, k=100).indices  # Retrieve top 100 sentences
    top_sentences = [candidate_sentence_ids[i] for i in top_k_indices]
    
    # Map sentences back to documents
    sorted_docs = []
    seen_docs = set()
    for sentence_id in top_sentences:
        docid = get_original_docid(sentence_id)
        if docid not in seen_docs:
            sorted_docs.append(docid)
            seen_docs.add(docid)
        if len(sorted_docs) >= 10:
            break  # Limit to top 10 documents
    final_results[query_id] = sorted_docs

    # Evaluation
    relevant_doc = dev_queries[dev_queries['query_id'] == query_id]['positive_docs'].values[0]
    # Handle cases where relevant_doc might be a list or string
    if isinstance(relevant_doc, str):
        relevant_docs = [relevant_doc]
    elif isinstance(relevant_doc, list):
        relevant_docs = relevant_doc
    else:
        relevant_docs = []

    # Compute MRR and Recall for different k values
    for k in k_values:
        retrieved_docs = sorted_docs[:k]
        # MRR
        try:
            rank = next(i + 1 for i, docid in enumerate(retrieved_docs) if docid in relevant_docs)
            mrr_scores[k].append(1 / rank)
        except StopIteration:
            mrr_scores[k].append(0)
        # Recall
        recall = len(set(relevant_docs) & set(retrieved_docs)) / len(relevant_docs)
        recall_scores[k].append(recall)

# Compute average MRR and Recall
for k in k_values:
    avg_mrr = np.mean(mrr_scores[k])
    avg_recall = np.mean(recall_scores[k])
    print(f"Sentence-based LM MRR@{k}: {avg_mrr:.4f}")
    print(f"Sentence-based LM Recall@{k}: {avg_recall:.4f}")