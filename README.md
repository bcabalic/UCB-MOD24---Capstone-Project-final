# Music Genre Classification via Spotify Audio Features

Automatic music genre classification using Spotify's pre-computed audio features and machine learning. This project performs exploratory data analysis (EDA) and establishes a K-Nearest Neighbors baseline model across 114 genre labels.

---

## Research Question

> *Can songs be automatically classified into musical genres based on extracted audio features such as danceability, energy, loudness, tempo, and acousticness — and which features carry the most discriminative signal?*

---

## Potential Business Applications

### Music Platforms & Streaming
- **API licensing** — license the genre classification model to streaming platforms (Spotify, Apple Music, YouTube Music) for smarter auto-tagging and playlist curation
- **White-label recommendation engine** — serve smaller streaming platforms that lack in-house ML teams

### Music Licensing & Sync
- **Music supervisor search tool** — enable TV, film, and ad teams to search by genre, mood, and vibe with precision; manual tagging is a major industry pain point
- **Stock music marketplace search** — improve discovery and metadata quality for licensing marketplaces

### Rights & Publishing
- **PRO & distributor partnerships** — assist ASCAP, BMI, DistroKid, and TuneCore with automated genre classification for royalty routing

### User-Facing Products
- **Artist analysis app** — let artists upload tracks and receive instant genre positioning feedback (e.g. "73% Hyperpop, 20% PC Music")
- **Deep genre ID app** — a Shazam-style tool that identifies songs and returns full genre breakdowns with similar artist suggestions

### Data & Research
- **A&R trend analytics** — sell genre shift data to record labels for predicting emerging micro-genres before mainstream breakout
- **Market research licensing** — license aggregated genre trend data to music journalists, market researchers, and investment firms

---
## Dataset

**[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)** via Kaggle

- 114,000 track-genre records
- 114 genre labels (1,000 tracks each — perfectly balanced)
- 13 numerical audio descriptors per track

---

## Project Structure

```
├── spotify_genre_classification_module24_final.ipynb   # Main notebook
├── spotify_dataset.csv                                  # Dataset (download from Kaggle)
└── README.md
```

---

## Notebook Overview

| Section | Description |
|---|---|
| 1. Import Libraries | Dependencies and plot configuration |
| 2. Load & Inspect | Dataset structure, dtypes, genre counts |
| 3. Data Cleaning | Null handling, duplicates, outlier inspection |
| 4. Feature Engineering | 6 composite audio features |
| 5. EDA | Distributions, correlations, discriminability |
| 6. Prepare for Modeling | Stratified sampling, encoding, scaling |
| 7. Baseline Model — KNN | Hyperparameter tuning, evaluation |
| 8. Random Forest | Randomized search tuning, feature importance, evaluation |
| 9. Multi-Layer Perceptron | Neural network tuning and evaluation |
| 10. Model Comparison | Leaderboard, bar chart, per-genre F1 overlay |
| 11. Summary & Business Implications | Key findings, generative AI use cases |

---

## Feature Engineering

Six composite features were created to improve genre separation:

| Feature | Formula | Rationale |
|---|---|---|
| `energy_valence_ratio` | `energy / (valence + 0.01)` | Separates high-energy dark (metal) vs. happy (pop) |
| `acoustic_energy_diff` | `acousticness - energy` | Classical/folk vs. electronic |
| `dance_tempo_product` | `danceability × tempo` | Uptempo groove vs. slow dance |
| `speech_instrumental_diff` | `speechiness - instrumentalness` | Hip-hop vs. ambient/classical |
| `duration_min` | `duration_ms / 60000` | Human-readable duration |
| `loudness_norm` | `(loudness + 60) / 60` | Re-scales loudness to [0, 1] |

---

## Key EDA Findings

- **Top discriminative features:** `acousticness`, `energy`, `instrumentalness`, `speechiness`, `loudness_norm`
- **Classical** stands apart with high acousticness and instrumentalness — easiest to classify
- **Metal / Heavy-Metal / Black-Metal** share nearly identical energy profiles — hardest cluster to separate
- **Hip-hop / R&B** overlap in danceability and speechiness; boundary is cultural rather than acoustic
- **16,299 tracks** appear under 2+ genre labels, reflecting real-world multi-label music categorization

---

## Baseline Model Results

**Algorithm:** K-Nearest Neighbors (KNN) with Euclidean distance  
**Tuning:** 5-fold stratified cross-validation over K = 1–20  
**Best K:** 18

| Metric | Score |
|---|---|
| Test Accuracy | See notebook output |
| Macro F1 Score | See notebook output |
| Random Chance Baseline | 0.88% (1/114) |

---

## Module 24 — Multi-Model Comparison Results

Three models were trained and evaluated on the same 34,200-row stratified dataset:

| Model | Test Accuracy | Macro F1 | Lift vs Random |
|---|---|---|---|
| KNN (Baseline) | See notebook output | See notebook output | See notebook output |
| Random Forest | See notebook output | See notebook output | See notebook output |
| MLP Neural Network | See notebook output | See notebook output | See notebook output |

**Key findings:**
- Random Forest outperformed both KNN and the MLP neural network
- The same four features — `acousticness`, `instrumentalness`, `energy`, `speechiness` — ranked as most important across EDA, Random Forest feature importance, and model comparison
- Ambient, sleep, and study genres were the hardest cluster across all models — an acoustic overlap that reflects intentional design, not a modeling failure
- All three models beat random chance by a significant margin, confirming that Spotify's audio feature vectors carry genuine discriminative signal

---

