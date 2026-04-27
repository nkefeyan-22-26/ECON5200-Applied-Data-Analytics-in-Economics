FedSpeak 2.0 — NLP Pipeline for Central Bank Communications
Course: ECON 5200: Causal Machine Learning & Applied Analytics — Lab 23

Objective
Diagnose, repair, and extend a broken NLP pipeline for analyzing Federal Reserve meeting minutes, with applications to sentiment scoring and document clustering.

Methodology

Pipeline Diagnosis: Identified three planted errors in a broken NLP pipeline — a naive whitespace tokenizer, a domain-inappropriate sentiment dictionary (Harvard GI), and misconfigured TF-IDF parameters (min_df=1, max_df=1.0)
Preprocessing: Corrected tokenization using nltk.word_tokenize() with regex-based punctuation stripping and lemmatization via WordNetLemmatizer
Sentiment Analysis: Replaced the Harvard General Inquirer dictionary with the Loughran-McDonald (LM) financial dictionary to eliminate false-positive negativity signals on neutral financial terms
TF-IDF Vectorization: Fixed feature engineering with min_df=5, max_df=0.85, and bigram support (ngram_range=(1,2)) to remove both noise terms and ubiquitous background words
Sentence-Transformer Embeddings: Encoded FOMC documents using all-MiniLM-L6-v2 (384-dimensional dense vectors) and compared clustering quality against TF-IDF via silhouette score
Predictive Evaluation: Evaluated both representations on a tightening/easing regime classification task using logistic regression with TimeSeriesSplit cross-validation
Module: Packaged the corrected pipeline into src/fomc_sentiment.py with three reusable functions: preprocess_fomc(), compute_lm_sentiment(), and build_tfidf_matrix()
