# 😏 Sarcasm Detection using Bidirectional LSTM + GloVe

## Overview
An NLP model that detects sarcasm in news headlines using a Bidirectional LSTM neural network with pretrained GloVe word embeddings. Achieves **85.7% accuracy on the test set** - a strong result for this notoriously difficult NLP task.

## Problem Statement
Sarcasm is one of the hardest aspects of natural language for machines to understand. Can a deep learning model learn to detect it from news headlines?

## Approach

| Step | Detail |
|------|--------|
| Dataset | News headlines dataset (binary: sarcastic / not sarcastic) |
| Text Preprocessing | Tokenisation, integer sequences, padding |
| Word Embeddings | GloVe (Global Vectors for Word Representation) |
| Embedding Layer | Pretrained weights, frozen during training |
| Model | Bidirectional LSTM |
| Regularisation | 10% Dropout |
| Activation | ReLU (hidden), Sigmoid (output) |
| Loss Function | Binary Crossentropy |
| Optimiser | Adam |
| Training | 20 epochs, 80/20 train-test split |

## Key Concepts
- **GloVe Embeddings**: Pretrained word vectors capturing semantic relationships between words
- **Bidirectional LSTM**: Processes text in both forward and backward directions - critical for understanding tone and irony
- **Frozen Embedding Layer**: Pretrained GloVe weights kept fixed so the model uses existing semantic knowledge

## Results

| Metric | Score |
|--------|-------|
| Training Accuracy | ~95% |
| **Test Accuracy** | **85.7%** |

## Tech Stack
```
Python | TensorFlow | Keras | NumPy | Pandas | scikit-learn | Seaborn
```

## How to Run
1. Clone the repo
2. Install dependencies: `pip install tensorflow numpy pandas scikit-learn seaborn`
3. Download GloVe embeddings (`glove.6B.100d.txt`) from [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/) and update the path
4. Open in Jupyter or Google Colab and run all cells

---
*Completed as part of the Post-Graduate Program in AI & Machine Learning - University of Texas at Austin (2021)*
