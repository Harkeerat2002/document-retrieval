# DIS-Project-1: Document Retrieval

**Authors:** Harkeerat Singh Sawhney, Jiayi Li, Alexandre Sallinen

# Overview
This repo contain three different implementations of document retrieval system. If you want to get our **best** results please run the first implementation. The other two implementations are us trying to implement different strategies. The implementations are as follows:
### 1. BM25 Ranking (Best Implementation)
- Filename: `br.ipynb`
- Score: 0.77599
### 2. Cosine Similarity + BM25 Ranking
- Filename: `cosine_similarity_bm25_reraanking.py`
- Score: 0.68688
### 3. Cosine Similarity
- Filename: `cosine_similarity.py`
- Score: 0.27647
# Implementations:
## 1. BM25 Ranking
This implementation gave us the best result. It is implemented in the notebook `bm25_ranking.ipynb`. This gave us the score of $0.77599$ on Kaggle. The main functions of this notebook include:
- **Loading and preprocessing the corpus and datasets.**
This section loads the corpus and the training and test datasets. It preprocesses the corpus and the datasets by using different text processing techniques such as tokenization, lemmatization, and stopword removal.
- **Building BM25 models for each language.**
This section builds BM25 models for each language using the preprocessed corpus.
- **Evaluating Recall@10 on the validation set.**
This section evaluates the Recall@10 metric on the validation set for each language.
- **Retrieving documents for test queries.**
This section retrieves documents for test queries using the BM25 models.
- **Generating a submission file.**
This section generates a submission file containing the retrieval results for test queries.

### How to run the code:
- Place the following three files in the same directory as this notebook: `corpus.json`, `train.csv`, `test.csv`
- Run the notebook `BM25_Ranking.ipynb`, it will generate the required files and submission.csv file.

### Requirements
- `Python 3.x`
- *Required Libraries:* `pandas, numpy, scikit-learn, nltk, transformers, tqdm, joblib, konlpy`

### Generated Files:
- **Preprocessed Corpus:**
    - *Filename:* `preprocessed_corpus.pkl`
    - *Content:* Pickle file containing the preprocessed corpus texts and BM25 Models and Document IDs:

- **BM25 Model Files:**
    - `bm25_model_{lang}.joblib` (e.g., bm25_model_en.joblib)
- **Document ID Files:**
    - `doc_ids_{lang}.pkl` (e.g., doc_ids_en.pkl)
- **Validation Queries:**
    - `preprocessed_val_queries.pkl`
    - `val_query_langs.pkl`
- **Test Queries:**
    - `preprocessed_test_queries.pkl`
    - `test_query_langs.pkl`
- **Submission File:**
    - *Filename:* `submission.csv`
    - *Content:* CSV file containing the retrieval results for test queries.

Your do not need to generate all these files again once you complete execution.

## Cosine Similarity with BM25 Re-ranking
This script implements a document retrieval system that combines cosine similarity with BM25 re-ranking. The system preprocesses text data, computes term frequency (TF) and document frequency (DF) dictionaries, calculates inverse document frequency (IDF) scores, and ranks documents using cosine similarity and BM25.

### Requirements
- Python 3.x
- Required Libraries:
- pandas
- numpy
- nltk
- torch
- transformers
- tqdm

### Directory Setup
The whole dataset should be under the folder named `data` in the same directory as the script. The `data` folder should contain the following files:
- `./corpus.json/corpus.json`: JSON file containing the corpus data.
- `./train.csv`: CSV file containing the training data.
- `./test.csv`: CSV file containing the test data.
- `./dev.csv`: CSV file containing the validation data.

### Functions
- `compute_tf_df_and_avgdl(corpus_df, path_to_saved_file)`
    - Computes the term frequency (TF), document frequency (DF), average document length (avgdl), and number of documents.

- `compute_idf(df_dict, num_docs)`
    - Computes the inverse document frequency (IDF) scores.
- `load_and_preprocess_queries(path_to_test_query)`
    - Loads and preprocesses the test queries.
- `rank_documents_with_cosine_similarity_and_bm25(corpus_df, test_query_df, tf_dict, idf_dict, avgdl)`

### Output Ranked Documents CSV:
- **Filename:** submission.csv
- **Content:** CSV file containing the query IDs and the ranked document IDs.

## 3. Document Retrieval with Cosine Similarity and Language Embedding
This project implements a document retrieval system using cosine similarity and language embeddings. The system preprocesses text data, builds an inverted index, retrieves candidate documents, and re-ranks them using multilingual embeddings.

### Required Libraries:
- pandas
- numpy
- nltk
- torch
- transformers
- tqdm
- fast_langdetect

### Functions
#### 1. split_into_sentences(text)
- Splits the given text into sentences.

#### 2. preprocess_text(text)
- Preprocesses the text by detecting the language, removing stop words, and tokenizing.

#### 3. build_inverted_index(docs)
- Builds an inverted index from the given documents.

#### 4. get_original_docid(sentence_id)
- Extracts the original document ID from a sentence ID.

#### 5. preprocess_query(query)
 - Preprocesses a query using the same preprocessing steps as for documents.

#### 6. score_documents_for_query(args)
- Scores documents for a given query using the BM25 algorithm.

#### 7. get_embedding(text)
- Generates embeddings for the given text using a pre-trained multilingual model.









