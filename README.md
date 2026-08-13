# Temporal Social Recommendation on Epinions

A study of **rating prediction on the Epinions dataset** using user-item interactions, social trust relationships, and temporal information.

## Overview

This project compares several recommendation models under a **chronological evaluation setting**, where earlier interactions are used to predict future ratings.

The models implemented include:

- Bias-Only Baseline
- Static GraphRec
- Temporal Matrix Factorization
- Temporal GraphRec with attention

The Epinions dataset contains approximately **922K ratings**, **300K trust relationships**, **22K users**, and **296K items** across **11 temporal bins**. :contentReference[oaicite:0]{index=0}

## Temporal Split

The data is split chronologically:

| Split | Time Bins | Samples |
|---|---|---:|
| Train | 1–7 | 727,400 |
| Validation | 8–9 | 121,524 |
| Test | 10–11 | 73,343 |

This setup evaluates the models on future interactions rather than randomly sampled ratings. :contentReference[oaicite:1]{index=1}

## Models

### Bias-Only

$$ \hat r_{ui} = \mu + b_u + b_i $$

A simple baseline using global, user, and item biases.

### Static GraphRec

Uses:

- User/item embeddings
- Rating embeddings
- User rating history
- Social trust history
- Neural history aggregation

### Temporal Matrix Factorization

Extends matrix factorization with:

- Time-dependent item biases
- User temporal drift
- User/item latent factors

### Temporal GraphRec

Combines:

- User-item interaction history
- Social trust history
- Temporal embeddings
- Attention-based history aggregation
- User/item biases

## Results

| Model | Test RMSE | Test MAE |
|---|---:|---:|
| Bias-Only | 1.0443 | **0.7988** |
| Static GraphRec | 1.0423 | 0.8174 |
| Temporal Matrix Factorization | 1.0570 | 0.8159 |
| **Temporal GraphRec** | **1.0387** | 0.8013 |

The Temporal GraphRec model achieves the best reported **RMSE of 1.0387**. :contentReference[oaicite:2]{index=2}

## Tech Stack

- Python
- PyTorch
- NumPy
- SciPy
- Pandas
- scikit-learn
- NetworkX
- Matplotlib
- Seaborn

## Project Pipeline

```text
Epinions Dataset
       │
       ├── Ratings ──────┐
       │                 │
       └── Trust Graph ──┤
                         ▼
                 Temporal Split
                         │
                         ▼
                Model Training
                         │
                         ▼
                  Rating Prediction
                         │
                         ▼
                   RMSE / MAE
````

## Future Work

* Strict time-aware history construction
* Dynamic social graph modeling
* Temporal/relative-time attention
* Transformer-based history aggregation
* Model ablation studies
* Hyperparameter optimization
* Multi-seed evaluation

## Dataset

The project uses the Epinions rating and trust data with timestamps. The dataset files expected by the current experiments are:


