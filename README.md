# IMDB Sentiment Classification with Sequence Models

PyTorch comparison of RNN, LSTM, and Bidirectional LSTM architectures for binary sentiment classification on the [Stanford IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

The project covers the full pipeline: text preprocessing, vocabulary construction, multiple embedding strategies, controlled training, evaluation, and a FastAPI inference service for the best-performing model.

## Problem

The goal is to classify a movie review as **positive** or **negative**.

Sequence models are well suited to this task because sentiment can depend on word order and context, particularly in cases involving negation. The main experimental question is which architecture and input representation generalize best when the training protocol, vocabulary, data splits, and random seed are kept consistent.


### Architectures

- Vanilla RNN
- LSTM
- Bidirectional LSTM

### Input Representations

Five representation settings are evaluated for each architecture:

| Representation | Variant |
|---|---|
| TF-IDF | Precomputed document vectors (sequence length 1) |
| GloVe | Frozen pretrained embeddings |
| GloVe | Fine-tuned embeddings |
| Word2Vec | Frozen embeddings trained on the IMDB training set |
| Word2Vec | Fine-tuned embeddings |

TF-IDF is included as a bag-of-words baseline. Since it represents the entire document as a single vector, the recurrent layer cannot recover the original word sequence, but it provides a useful baseline for comparing lexical and sequential representations.

### Text Preprocessing

The same preprocessing pipeline is used during training and inference:

- Lowercasing and contraction expansion
- URL and HTML removal
- Punctuation stripping
- Stopword removal
- Negation-sensitive filtering, preserving words such as *not* that can reverse sentiment

### Evaluation

Models are evaluated on the held-out test set using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Classification reports

## Results

Test-set metrics are shown below. Values are rounded, with full result tables available in `results/`.

| Model | Embedding | Variant | Accuracy | F1 |
|---|---|---|---:|---:|
| **BiLSTM** | **Word2Vec** | **Frozen** | **0.897** | **0.896** |
| LSTM | Word2Vec | Frozen | 0.893 | 0.894 |
| RNN | TF-IDF | — | 0.883 | 0.885 |
| LSTM | TF-IDF | — | 0.882 | 0.879 |
| BiLSTM | GloVe | Frozen | 0.882 | 0.880 |
| BiLSTM | TF-IDF | — | 0.880 | 0.877 |
| LSTM | GloVe | Frozen | 0.876 | 0.873 |
| LSTM | Word2Vec | Fine-tuned | 0.872 | 0.876 |
| LSTM | GloVe | Fine-tuned | 0.866 | 0.872 |
| BiLSTM | GloVe | Fine-tuned | 0.867 | 0.858 |
| BiLSTM | Word2Vec | Fine-tuned | 0.863 | 0.853 |
| RNN | Word2Vec | Fine-tuned | 0.829 | 0.840 |
| RNN | Word2Vec | Frozen | 0.820 | 0.810 |
| RNN | GloVe | Fine-tuned | 0.796 | 0.803 |
| RNN | GloVe | Frozen | 0.747 | 0.711 |

### Selected Model

**BiLSTM + Frozen Word2Vec**

- Accuracy: `0.90`
- Precision: `0.91`
- Recall: `0.88`
- F1-score: `0.90`

The metrics reported by the API correspond to the selected model checkpoint.

## Key Observations

- **BiLSTM with domain-trained Word2Vec** achieved the strongest overall performance.
- Frozen Word2Vec slightly outperformed frozen GloVe, suggesting that embeddings trained directly on IMDB reviews captured useful domain-specific language.
- Fine-tuning the embedding layers substantially increased the number of trainable parameters and reduced test F1 in this setup, indicating increased overfitting.
- Vanilla RNNs performed worse with sequential embeddings, consistent with their limitations in capturing long-range dependencies.
- TF-IDF remained a competitive baseline despite discarding word order.
- Packed padded sequences are used to prevent padding tokens from affecting hidden-state updates.

## Repository Structure

```text
.
├── api/
│   └── main.py                          # FastAPI serving the selected BiLSTM model
│
├── notebooks/
│   ├── save_csv.ipynb                   # IMDB files → train/test CSVs
│   ├── preprocessing.ipynb              # Text cleaning pipeline
│   ├── vocab.ipynb                      # Vocabulary, encoding, and padding
│   │
│   └── models/
│       ├── rnn.ipynb
│       ├── lstm.ipynb                   # Word2Vec and TF-IDF artifacts
│       └── bidirectional_lstm.ipynb
│
├── results/                             # Model evaluation results
│
└── data/
    └── shared/                          # Vocabulary, embeddings, and generated splits
```

Model checkpoints are expected under:

```text
notebooks/models/models/
```

For example:

```text
biltsm_word2vec_frozen.pt
```

The vocabulary is stored at:

```text
data/shared/vocabulary.pkl
```

Large dataset artifacts such as `data/aclImdb/` are excluded from Git.

## Model Serving

The FastAPI service loads the selected frozen Word2Vec BiLSTM checkpoint and applies the same preprocessing and encoding pipeline used during training.

Predictions are generated using a sigmoid over the model output to obtain positive and negative sentiment scores.

### Install Dependencies

```bash
pip install torch fastapi uvicorn pydantic
```

### Start the API

```bash
uvicorn api.main:app --reload
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check model loading status |
| `/model_info` | GET | View architecture, embedding type, and test metrics |
| `/list_models` | GET | List available `.pt` checkpoints |
| `/predict` | POST | Predict sentiment for a movie review |

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This movie was excellent!\"}"
```

### Example Response

```json
{
  "text": "This movie was excellent!",
  "prediction": "positive",
  "confidence": 0.97,
  "scores": {
    "negative": 0.03,
    "positive": 0.97
  },
  "model": "biltsm_word2vec_frozen"
}
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```


## Tech Stack

- Python
- PyTorch
- scikit-learn
- Gensim / Word2Vec
- GloVe
- TF-IDF
- NLTK
- NumPy
- pandas
- Matplotlib
- FastAPI


## Dataset

This project uses the **Stanford Large Movie Review Dataset (IMDB)** introduced by Maas et al. (ACL 2011).
