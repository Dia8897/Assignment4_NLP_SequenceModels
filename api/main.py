"""FastAPI deployment for the best sentiment model in this project."""

import pickle
import re #to apply cleaning with regular expressions
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "notebooks" / "models" / "models"
MODEL_PATH = CHECKPOINT_DIR / "lstm_word2vec_frozen.pt"
VOCAB_PATH = ROOT / "data" / "shared" / "vocabulary.pkl"

MODEL_NAME = "lstm_word2vec_frozen"
METRICS = {
    "accuracy": 0.8891576952542236,
    "precision": 0.8678357245881728,
    "recall": 0.9189710610932476,
    "f1": 0.8926716901573419,
}


class ReviewRequest(BaseModel):
    text: str = Field(min_length=1, examples=["This movie was excellent!"])
    # the field called text must be a string
    # predict(review) -> use review.text


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_weights: torch.Tensor, padding_idx: int):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights, freeze=True, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            input_size=embedding_weights.shape[1],
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(128, 1)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor):
        embedded = self.embedding(sequences)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.output(self.dropout(hidden[-1])).squeeze(1)


# The training preprocessing kept negation words because they affect sentiment
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "hers",
    "him", "his", "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "my", "myself", "of", "on", "or", "our", "ours", "ourselves",
    "she", "so", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why",
    "with", "you", "your", "yours", "yourself", "yourselves",
}

CONTRACTIONS = {
    "can't": "cannot", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "isn't": "is not",
    "mustn't": "must not", "needn't": "need not", "shan't": "shall not",
    "shouldn't": "should not", "wasn't": "was not", "weren't": "were not",
    "won't": "will not", "wouldn't": "would not", "i'm": "i am",
    "i've": "i have", "i'll": "i will", "it's": "it is", "that's": "that is",
    "they're": "they are", "we're": "we are", "you're": "you are",
}


def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = " ".join(CONTRACTIONS.get(word, word) for word in text.split())
    text = re.sub(r"https?://\S+|www\.\S+|<.*?>|&amp;", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [word for word in text.split() if word not in STOPWORDS]


def load_model():
    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        return None, None, None, "Model checkpoint or vocabulary is missing"

    try:
        with VOCAB_PATH.open("rb") as file:
            vocabulary = pickle.load(file)

        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        embedding_weights = state["embedding.weight"]
        # we load the embedding weights from the trained model checkpoint, not from the original Word2Vec model
        model = LSTMClassifier(
            embedding_weights=embedding_weights,
            padding_idx=vocabulary["word_to_idx"]["<PAD>"],
        )
        

        model.load_state_dict(state)
        model.eval()
        return model, vocabulary["word_to_idx"], vocabulary["MAX_LENGTH"], None
    except Exception as error:
        return None, None, None, str(error)


model, word_to_idx, max_length, load_error = load_model()
app = FastAPI(title="IMDB Sentiment API", version="1.0.0")


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "error": load_error,
    }


@app.get("/model_info")
def model_info():
    return {
        "name": MODEL_NAME,
        "architecture": "LSTM",
        "embedding": "Word2Vec",
        "embedding_trainable": False,
        "task": "binary IMDB sentiment classification",
        "labels": ["negative", "positive"],
        "metrics": METRICS,
    }


@app.get("/list_models")
def list_models():
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt")) if CHECKPOINT_DIR.exists() else []
    return {
        "models": [
            {"name": path.stem, "selected": path.stem == MODEL_NAME}
            for path in checkpoints
        ]
    }


@app.post("/predict")
def predict(review: ReviewRequest):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model is unavailable: {load_error}")

    tokens = preprocess(review.text)
    if not tokens:
        raise HTTPException(status_code=400, detail="Text has no usable words")

    unknown = word_to_idx.get("<UNK>", 1)
    pad = word_to_idx["<PAD>"]
    token_ids = [word_to_idx.get(token, unknown) for token in tokens][:max_length]
    length = len(token_ids)
    token_ids += [pad] * (max_length - length)

    sequence = torch.tensor([token_ids], dtype=torch.long)
    lengths = torch.tensor([length], dtype=torch.long)
    with torch.inference_mode(): #don't compute gradients, don't prepare for training, just perform a forward pass
        positive_score = torch.sigmoid(model(sequence, lengths)).item() #sigmoid makes the logit between 0 and 1

    scores = {
        "negative": round(1.0 - positive_score, 6),
        "positive": round(positive_score, 6),
    }
    return {
        "text": review.text,
        "prediction": max(scores, key=scores.get),
        "confidence": round(max(scores.values()), 6),
        "scores": scores,
        "model": MODEL_NAME,
    }
