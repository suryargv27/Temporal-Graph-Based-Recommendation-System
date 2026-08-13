# Temporal Social Recommendation on Epinions

A study of **temporal rating prediction** using user-item interactions, social trust relationships, and temporal dynamics on the Epinions dataset.

## Overview

The project compares four recommendation models under a **chronological evaluation setting**:

- Bias-Only Baseline
- Static GraphRec
- Temporal Matrix Factorization
- Temporal GraphRec with Attention

The dataset contains **922,267 ratings**, **300,548 trust relationships**, **22,164 users**, **296,277 items**, and **11 temporal bins**.

## Temporal Split

| Split | Time Bins | Samples |
|---|---|---:|
| Train | 1–7 | 727,400 |
| Validation | 8–9 | 121,524 |
| Test | 10–11 | 73,343 |

Models are trained on past interactions and evaluated on future interactions.

## Models

### Bias-Only

$$
\hat r_{ui} = \mu + b_u + b_i
$$

Global, user, and item biases provide the baseline.

### Static GraphRec

Uses user/item/rating embeddings, rating history, and social trust history:

$$
h_u = f(H_u^{ratings}, H_u^{social})
$$

### Temporal Matrix Factorization

Models latent factors and temporal drift:

$$
\hat r_{uit} =
\mu+b_u+b_i+b_{i,t}
+\alpha_u dev(u,t)
+P_u^TQ_i
$$

with latent dimension $k=32$.

### Temporal GraphRec

Combines rating history, social history, temporal embeddings, attention, and user/item biases:

$$
\hat r_{uit} =
f(h_u,e_i)+\mu+b_u+b_i
$$

Historical interactions are aggregated using attention over item/rating/time representations.

## Results

| Model | RMSE | MAE |
|---|---:|---:|
| Bias-Only | 1.0443 | **0.7988** |
| Static GraphRec | 1.0423 | 0.8174 |
| Temporal MF | 1.0570 | 0.8159 |
| **Temporal GraphRec** | **1.0387** | 0.8013 |

**Best RMSE:** Temporal GraphRec — **1.0387**

## Pipeline

```text
Epinions
   │
   ├── Ratings ──► User-Item History ──┐
   │                                    │
   └── Trust ───► Social Graph ─────────┤
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


