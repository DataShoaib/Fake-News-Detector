# Fake News Detection using LSTM Deep Learning

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Project Overview](#project-overview)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [Word Embeddings — GloVe](#word-embeddings--glove)
7. [Tech Stack](#tech-stack)
8. [Getting Started](#getting-started)
9. [Pipeline Walkthrough](#pipeline-walkthrough)
10. [Results](#results)
11. [Requirements](#requirements)
12. [Contributing](#contributing)
13. [License](#license)
14. [Acknowledgements](#acknowledgements)

---

## Problem Statement

The proliferation of fake news and misinformation has emerged as one of the most critical challenges of the modern digital era. With billions of articles, posts, and news pieces published online every day, it has become increasingly difficult for readers to distinguish credible information from fabricated or misleading content.

Fake news carries far-reaching consequences — it can sway political elections, fuel social unrest, damage individual reputations, and even endanger lives during public health crises. Traditional manual fact-checking methods are slow, resource-intensive, and cannot scale to match the velocity at which misinformation spreads across platforms.

**There is a pressing need for an automated, scalable, and accurate system capable of detecting potentially unreliable news articles in real time.**

This project addresses that need by developing a deep learning-based text classification model that learns linguistic and semantic patterns from thousands of labeled news articles. The model uses a **Long Short-Term Memory (LSTM)** network — well-suited for understanding the sequential and contextual nature of natural language — combined with **pre-trained GloVe word embeddings** that represent words as dense semantic vectors. The system classifies any given news article as either **Reliable (0)** or **Potentially Unreliable / Fake (1)**, achieving up to **95% accuracy** when trained sufficiently.

---

## Project Overview

| Attribute         | Details                                              |
|-------------------|------------------------------------------------------|
| Task              | Binary Text Classification (Fake vs Real News)       |
| Model             | LSTM (Long Short-Term Memory) Neural Network         |
| Embeddings        | GloVe 6B — 100-dimensional pre-trained vectors       |
| Framework         | TensorFlow / Keras                                   |
| Best Accuracy     | ~95% (50+ epochs)                                    |
| Input             | News article title + body text                       |
| Output            | `0` = Reliable &nbsp;&nbsp; `1` = Unreliable (Fake)  |

---

## Project Structure

```
fake-news-detection/
│
├── data/
│   ├── train.csv                  # Labeled training dataset
│   ├── test.csv                   # Unlabeled test dataset
│         
│
├── notebooks/
│   └── fake_news_lstm.ipynb       # End-to-end Jupyter notebook
│
├── models/
│   └── lstm_model.h5              # Saved Keras model after training
│
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Dataset

The dataset contains real-world news articles labeled for credibility. It was originally published as part of a Kaggle competition on fake news detection.

| Column   | Type    | Description                                                        |
|----------|---------|--------------------------------------------------------------------|
| `id`     | Integer | Unique identifier for each article                                 |
| `title`  | String  | Headline of the news article                                       |
| `author` | String  | Author of the article (may be missing for some entries)            |
| `text`   | String  | Full body text of the article (may be incomplete)                  |
| `label`  | Integer | `1` = Unreliable (Fake) &nbsp; / &nbsp; `0` = Reliable (Real)     |

> **Setup:** Download `train.csv` and `test.csv` from the Kaggle Fake News competition page and place both files inside the `data/` directory before running the notebook.

---

## Model Architecture

The model is a sequential LSTM-based deep learning architecture designed specifically for binary text classification.

```
┌─────────────────────────────────────────────┐
│              Input: Raw News Text            │
└──────────────────────┬──────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   Text Preprocessing    │
          │  (Clean, Tokenize, Pad) │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │    Embedding Layer      │
          │  GloVe 100d (frozen)    │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │      LSTM Layer         │
          │      (128 units)        │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │    Dropout Layer        │
          │       (rate = 0.5)      │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │    Dense Layer          │
          │   (64 units, ReLU)      │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │    Output Layer         │
          │   (1 unit, Sigmoid)     │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  Fake (1) / Real (0)    │
          └─────────────────────────┘
```

### Why LSTM?

Standard feedforward networks treat text as a flat bag of words, losing all positional and sequential information. LSTM networks solve this by maintaining a hidden state that carries contextual information across each time step in the sequence.

| Capability                    | Explanation                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| Sequential understanding      | Preserves word order and sentence structure — critical signals for credibility detection |
| Long-range dependencies       | Connects context from the beginning of an article to conclusions drawn at the end |
| Gated memory                  | Forget, input, and output gates allow selective retention of relevant information |
| Gradient stability            | Mitigates the vanishing gradient problem present in vanilla RNNs            |

---

## Word Embeddings — GloVe

The embedding layer is initialized with **pre-trained GloVe (Global Vectors for Word Representation)** vectors — specifically the `glove.6B.100d.txt` file, where each word is represented as a 100-dimensional dense float vector.

GloVe embeddings were trained on 6 billion tokens sourced from Wikipedia and Gigaword corpora, encoding rich semantic and syntactic relationships between words in the vector space. Words with similar meanings are positioned close together geometrically.

**Why use pre-trained embeddings instead of training from scratch?**

| Approach                   | Benefit                                                          |
|----------------------------|------------------------------------------------------------------|
| Pre-trained GloVe (used)   | Immediately captures world knowledge from 6 billion tokens       |
| Training from scratch      | Requires significantly more labeled data and compute to match    |
| Transfer learning benefit  | Better generalization even with limited training examples        |

The embedding layer weights are **frozen (non-trainable)** during model training to preserve the semantic quality of the GloVe representations and prevent overfitting on the smaller news dataset.

> **Setup:** Download `glove.6B.100d.txt` and place it inside the `data/` directory before running the notebook.

---

## Tech Stack

| Category          | Library / Tool                          |
|-------------------|-----------------------------------------|
| Data Handling     | `pandas`, `numpy`                       |
| Text Processing   | `nltk` (stopwords, stemming)            |
| Visualization     | `matplotlib`                            |
| Deep Learning     | `tensorflow`, `keras`                   |
| Model Evaluation  | `scikit-learn`                          |
| Embeddings        | GloVe 6B 100d (Stanford NLP Group)      |
| Environment       | Jupyter Notebook                        |

---

## Getting Started

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download Required Files

Place the following files inside the `data/` directory:

- `train.csv` — Labeled news dataset (from Kaggle)
- `test.csv` — Unlabeled test set (from Kaggle)
- `glove.6B.100d.txt` — Pre-trained GloVe vectors (100-dimensional)

### Step 4 — Run the Notebook

```bash
jupyter notebook notebooks/fake_news_lstm.ipynb
```

Execute all cells sequentially from top to bottom. Training for **50+ epochs** is recommended to reach peak accuracy.

---

## Pipeline Walkthrough

The notebook implements the following end-to-end pipeline:

**Step 1 — Load Data**
Read `train.csv` and `test.csv` using pandas. Inspect shape, null value counts, and class balance (fake vs real distribution).

**Step 2 — Text Preprocessing**
Combine `title`, `author`, and `text` columns into a single unified content field. Remove punctuation, special characters, and numeric tokens. Convert all text to lowercase. Remove English stopwords using NLTK's stopword corpus. Apply stemming where needed.

**Step 3 — Tokenization and Sequence Padding**
Fit a Keras `Tokenizer` on the training corpus to build a vocabulary. Convert each article to a sequence of integer token IDs. Pad all sequences to a uniform maximum length using `pad_sequences`.

**Step 4 — Build GloVe Embedding Matrix**
Parse `glove.6B.100d.txt` into a word-to-vector dictionary. Construct an embedding matrix of shape `(vocab_size, 100)`. Only words present in both the tokenizer vocabulary and GloVe are embedded; the rest are initialized to zero.

**Step 5 — Build the LSTM Model**
Define the model using Keras Sequential API: Embedding layer (GloVe-initialized, frozen) → LSTM (128 units) → Dropout (0.5) → Dense (64, ReLU) → Output (1, Sigmoid). Compile with Adam optimizer and Binary Crossentropy loss.

**Step 6 — Train the Model**
Fit the model on training data with a validation split. Monitor training and validation accuracy/loss per epoch. Train for 50+ epochs for optimal convergence.

**Step 7 — Evaluate the Model**
Generate a full classification report (precision, recall, F1-score per class). Plot the confusion matrix. Visualize training and validation accuracy/loss curves across epochs.

**Step 8 — Generate Predictions**
Run the trained model on `test.csv`. 

---

## Results

| Metric              | Value                 |
|---------------------|-----------------------|
| Training Accuracy   | ~95%                  |
| Validation Accuracy | ~93–95%               |
| Loss Function       | Binary Crossentropy   |
| Optimizer           | Adam                  |
| Recommended Epochs  | 50+                   |

> Training accuracy approaches **95%** after 50+ epochs. Early stopping and learning rate scheduling callbacks can be added to further stabilize and speed up convergence.

---

## Requirements

```
tensorflow>=2.6.0
keras>=2.6.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
nltk>=3.6.0
```

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## Contributing

Contributions, improvements, and bug fixes are welcome.

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## Acknowledgements

- Kaggle for providing the Fake News labeled dataset
- Stanford NLP Group for the GloVe pre-trained word vectors
- The open-source community behind TensorFlow, Keras, and NLTK