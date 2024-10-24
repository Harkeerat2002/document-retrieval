import pandas as pd
import numpy as np
import os
from nltk.corpus import stopwords
from ko_ww_stopwords.stop_words import ko_ww_stop_words
import string
from nltk.stem import PorterStemmer
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import nltk
import spacy
import faiss
from multiprocessing import Pool, cpu_count

# Paths
path_to_save_df = "./data/pd_df/"
path_to_train_query = "./data/train.csv"
corpus_file_path = "./data/corpus.json/corpus.json"
tokenized_text_path = os.path.join(path_to_save_df, "tokenized_text.pkl")
tokenized_query_path = os.path.join(path_to_save_df, "tokenized_query.pkl")


nltk.download("punkt_tab")
nltk.download("stopwords")

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])


# Create the directory if it does not exist
if not os.path.exists(path_to_save_df):
    os.makedirs(path_to_save_df)

# Download a pre-trained FastText model
# fasttext_model_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'
# fasttext_model_gz_path = 'cc.en.300.bin.gz'
# fasttext_model_bin_path = 'cc.en.300.bin'

# if not os.path.exists(fasttext_model_gz_path):
#     print("Downloading FastText model...")
#     os.system(f'wget {fasttext_model_url} -O {fasttext_model_gz_path}')
#     print("FastText model downloaded.")

# # Decompress the .gz file
# if not os.path.exists(fasttext_model_bin_path):
#     print("Decompressing FastText model...")
#     with gzip.open(fasttext_model_gz_path, 'rb') as f_in:
#         with open(fasttext_model_bin_path, 'wb') as f_out:
#             shutil.copyfileobj(f_in, f_out)
#     print("FastText model decompressed.")

# # Load the FastText model
# fasttext_model = fasttext.load_model(fasttext_model_bin_path)

# Stop-Words
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


# HELPER FUNCTIONS
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
        delayed(lambda row: (row["docid"], preprocess(row["text"], row["lang"], row["docid"]))) (row)
        for _, row in tqdm(batch_df.iterrows(), total=len(batch_df))
    )
    preprocessed_text_df = pd.DataFrame(
        preprocessed_data, columns=["doc_id", "preprocessed_text"]
    )
    
    batch_file_path = os.path.join(
        path_to_save_df, f"preprocessed_text_df_batch_{batch_index}.pkl"
    )
    preprocessed_text_df.to_pickle(batch_file_path)
    print(f"Batch {batch_index} saved.")
    
def compute_fasttext_embeddings(texts, model):
    embeddings = []
    for text in tqdm(texts):
        words = text.split()
        word_embeddings = [model.get_word_vector(word) for word in words]
        # Use the mean of the word embeddings as the sentence embedding
        sentence_embedding = np.mean(word_embeddings, axis=0)
        embeddings.append(sentence_embedding)
    return embeddings

def batch_tokenize(texts, ids, tokenizer, device, batch_size=32):
    tokenized_batches = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_docids = ids[i:i + batch_size]
        batch = [{"id": docid, "text": text} for docid, text in zip(batch_docids, batch_texts)]
        tokenized_batch = tokenizer([item["text"] for item in batch], padding=True, truncation=True, return_tensors="pt")
        tokenized_batch = {key: value.to(device) for key, value in tokenized_batch.items()}
        tokenized_batches.append({"batch": batch, "tokenized": tokenized_batch})
    return tokenized_batches


## LOAD THE DATASET

# Check if the Corpus DataFrame is already saved
if os.path.exists(os.path.join(path_to_save_df, "corpus_df.pkl")):
    print("Corpus DataFrame found. Loading it now.")

    #corpus_df = pd.read_pickle(os.path.join(path_to_save_df, "corpus_df.pkl"))
    # print(corpus_df.head())
else:
    print("Corpus DataFrame not found. Creating it now.")
    corpus_df = pd.read_json(corpus_file_path)

    pickle_file_path = os.path.join(path_to_save_df, "corpus_df.pkl")
    corpus_df.to_pickle(pickle_file_path)
    print("Corpus DataFrame saved.")

    # print(corpus_df.head())


## PREPROCESS THE CORPUS

# Check if the preprocessed corpus is already saved
if os.path.exists(os.path.join(path_to_save_df, "preprocessed_text_df.pkl")):
    print("Preprocessed Text found. Loading it now.")
    preprocessed_text_df = pd.read_pickle(
        os.path.join(path_to_save_df, "preprocessed_text_df.pkl")
    )
else:
    print("Preprocessed Text not found. Creating it now.")
    batch_size = 10000  # Define your batch size
    num_batches = len(corpus_df) // batch_size + 1

    for i in range(num_batches):
        batch_df = corpus_df.iloc[i * batch_size : (i + 1) * batch_size]
        process_batch(batch_df, path_to_save_df, i)

    # Combine all batch files into a single DataFrame
    preprocessed_text_dfs = []
    for i in range(num_batches):
        batch_file_path = os.path.join(
            path_to_save_df, f"preprocessed_text_df_batch_{i}.pkl"
        )
        preprocessed_text_dfs.append(pd.read_pickle(batch_file_path))

    preprocessed_text_df = pd.concat(preprocessed_text_dfs, ignore_index=True)
    preprocessed_text_df.to_pickle(
        os.path.join(path_to_save_df, "preprocessed_text_df.pkl")
    )
    print("Preprocessed Text saved.")

## PREPROCESS THE QUERY

if os.path.exists(os.path.join(path_to_save_df, "train_query_preprocessed.pkl")):
    print("Preprocessed query found. Loading it now.")
    # preprocessed_query_df = pd.read_pickle(
    #     os.path.join(path_to_save_df, "train_query_preprocessed.pkl")
    # )
else:
    print("Preprocessed query not found. Creating it now.")
    train_query = pd.read_csv(path_to_train_query)
    batch_size = 10000  # Define your batch size 
    num_batches = len(train_query) // batch_size + 1
    
    for i in range(num_batches):
        batch_df = train_query.iloc[i * batch_size : (i + 1) * batch_size]
        preprocessed_data = Parallel(n_jobs=-1)(
            delayed(lambda row: (row["query"], preprocess(row["query"], row["lang"], row["query_id"]))) (row)
            for _, row in tqdm(batch_df.iterrows(), total=len(batch_df))
        )
        preprocessed_query_df = pd.DataFrame(
            preprocessed_data, columns=["query_id", "preprocessed_query"]
        )
        
        batch_file_path = os.path.join(
            path_to_save_df, f"train_query_preprocessed_batch_{i}.pkl"
        )
        preprocessed_query_df.to_pickle(batch_file_path)
        print(f"Batch {i} saved.")

    # Combine all batch files into a single DataFrame
    preprocessed_query_dfs = []
    for i in range(num_batches):
        batch_file_path = os.path.join(
            path_to_save_df, f"train_query_preprocessed_batch_{i}.pkl"
        )
        preprocessed_query_dfs.append(pd.read_pickle(batch_file_path))
    
    preprocessed_query_df = pd.concat(preprocessed_query_dfs, ignore_index=True)
    preprocessed_query_df.to_pickle(
        os.path.join(path_to_save_df, "train_query_preprocessed.pkl")
    )
    print("Preprocessed query saved.")
        
    
## COMPUTE THE TOKEN 

# Loading the Model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)



# Load the preprocessed text and query
# preprocessed_text_df = pd.read_pickle(os.path.join(path_to_save_df, "preprocessed_text_df.pkl"))
# preprocessed_query_df = pd.read_pickle(os.path.join(path_to_save_df, "train_query_preprocessed.pkl"))

# Tokenize the text and query
# tokenized_text = tokenizer(preprocessed_text_df["preprocessed_text"].tolist(), padding=True, truncation=True, return_tensors="pt").to(device)

# tokenized_query = tokenizer(preprocessed_query_df["preprocessed_query"].tolist(), padding=True, truncation=True, return_tensors="pt").to(device)

# Save the tokenized text and query


if os.path.exists(os.path.join(path_to_save_df, "tokenized_text.pkl")):
    print("Tokenized text and query found. Loading them now.")
    # tokenized_text_batches = torch.load(tokenized_text_path)
    # tokenized_query_batches = torch.load(tokenized_query_path)

    
else:
    print("Tokenized text and query not found. Tokenizing them now.")
    tokenized_text_batches = batch_tokenize(preprocessed_text_df["preprocessed_text"].tolist(), preprocessed_text_df["doc_id"].tolist(), tokenizer, device)
    tokenized_query_batches = batch_tokenize(preprocessed_query_df["preprocessed_query"].tolist(), preprocessed_query_df["query_id"].tolist(), tokenizer, device)



    torch.save(tokenized_text_batches, tokenized_text_path)
    torch.save(tokenized_query_batches, tokenized_query_path)


## COMPUTE THE EMBEDDINGS
if os.path.exists(os.path.join(path_to_save_df, "preprocessed_text_embeddings.pkl")):
    preprocessed_text_embeddings_path = os.path.join(path_to_save_df, "preprocessed_text_embeddings.pkl")
    preprocessed_query_embeddings_path = os.path.join(path_to_save_df, "preprocessed_query_embeddings.pkl")
    print("Text embeddings found. Loading them now.")
    # text_embeddings_dict = pd.read_pickle(os.path.join(path_to_save_df, "preprocessed_text_embeddings.pkl"))
else:
    
    text_embeddings_dict = {}
    query_embeddings_dict = {}

    # Process text batches
    for batch in tqdm(tokenized_text_batches, desc="Processing text batches"):
        with torch.no_grad():
            outputs = model(**batch["tokenized"])
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            for docid, embedding in zip([item["id"] for item in batch["batch"]], batch_embeddings):
                text_embeddings_dict[docid] = embedding

    # Save the text embeddings dictionary
    preprocessed_text_embeddings_path = os.path.join(path_to_save_df, "preprocessed_text_embeddings.pkl")
    pd.to_pickle(text_embeddings_dict, preprocessed_text_embeddings_path)


    # Process query batches
    for batch in tqdm(tokenized_query_batches, desc="Processing query batches"):
        with torch.no_grad():
            outputs = model(**batch["tokenized"])
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            for docid, embedding in zip([item["id"] for item in batch["batch"]], batch_embeddings):
                query_embeddings_dict[docid] = embedding

    # Save the query embeddings dictionary
    preprocessed_query_embeddings_path = os.path.join(path_to_save_df, "preprocessed_query_embeddings.pkl")
    pd.to_pickle(query_embeddings_dict, preprocessed_query_embeddings_path)
    
## COMPUTE ANN SEARCH

# Load the embeddings
text_embeddings_dict = pd.read_pickle(preprocessed_text_embeddings_path)
query_embeddings_dict = pd.read_pickle(preprocessed_query_embeddings_path)

# Convert embeddings to numpy arrays
text_embeddings_array = np.array(list(text_embeddings_dict.values())).astype('float32')
query_embeddings_array = np.array(list(query_embeddings_dict.values())).astype('float32')

# Build the FAISS index and move it to GPU
res = faiss.StandardGpuResources()
index_flat = faiss.IndexFlatL2(text_embeddings_array.shape[1])
gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
gpu_index.add(text_embeddings_array)

# Function to perform ANN search for a single query embedding
def ann_search(query_embedding):
    distances, indices = gpu_index.search(query_embedding.reshape(1, -1), k)
    return distances[0], indices[0]

# Perform ANN search for each query embedding using multiprocessing
k = 10  # Number of nearest neighbors to retrieve
ranked_documents_dict = {}
query_ids = list(query_embeddings_dict.keys())
doc_ids = list(text_embeddings_dict.keys())

with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(ann_search, query_embeddings_array), total=len(query_embeddings_array), desc="Performing ANN search"))

for i, (distances, indices) in enumerate(results):
    ranked_documents_dict[query_ids[i]] = {doc_ids[idx]: distances[j] for j, idx in enumerate(indices)}

# Save the ranked documents dictionary
ranked_documents_path = os.path.join(path_to_save_df, "ranked_documents.pkl")
pd.to_pickle(ranked_documents_dict, ranked_documents_path)

# Save the ranked documents as a csv with query_id and top 10 document ids
ranked_documents_df = pd.DataFrame(ranked_documents_dict).T
ranked_documents_df.columns = [f"doc_{i}" for i in range(1, 11)]
ranked_documents_df.index.name = "query_id"
ranked_documents_df.to_csv(os.path.join(path_to_save_df, "ranked_documents.csv"))

print("Ranking completed. The ranked documents are saved as 'ranked_documents.csv'.")