# Speaker Recognition with Gaussian Mixture Models (GMM)

This project implements a speaker recognition system using Gaussian Mixture Models (GMMs) as part of the Voice Technology MSc. 2024–2025 Speech Recognition 1 course.

---

## Original Toolkit

The project is based on the sklearn.mixture.GaussianMixture implementation for Part A, while Part B includes custom implementations of univariate and multivariate GMMs without external libraries.

---

## Project Description

The goal of this project is to train and evaluate GMM-based models to recognize speakers from audio recordings. The project consists of:

### Part A

Implements speaker_rec_GMM(audio_dir, test_dir) which:

- Creates and trains one GMM per speaker using the audio files in audio_dir.
- Predicts the speaker label for each test audio file in test_dir.
- Returns a dictionary mapping test filenames to predicted speaker labels.

### Part B

Implements custom GMM functions:

- custom_GMM_uni(data, K_components, epsilon, seed) for univariate data.
- custom_GMM_multi(data, K_components, epsilon, seed) for multivariate data.
These functions estimate mixture weights, means, and variances from scratch.
A version of speaker_rec_GMM using custom_GMM_multi is also included.

---

## What This Repository Contains

This folder only includes the modified or created files for the project:

- s6028608_A.py – Contains the Part A implementation using sklearn.mixture.GaussianMixture.
- s6028608_B.py – Contains the custom GMM implementations and a speaker_rec_GMM function that uses them.

---

## Install Required Libraries

```bash
pip install numpy scipy librosa scikit-learn tqdm
```

---

## Language

The project is implemented in Python and tested with English audio prompts.

