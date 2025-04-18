# POS Tagging for Hindi using Hidden Markov Models (HMM)

Part-of-Speech (POS) tagging for Indian languages, especially Hindi, remains an active area of research due to the complexity and flexibility of the language structure. This project presents a POS tagger based on the Hidden Markov Model (HMM) framework, offering both supervised and unsupervised learning approaches.

## Overview

- **Supervised POS Tagging**: Utilizes a manually annotated Hindi corpus to train the HMM. The model learns word-tag associations and transition probabilities based on real language data. Accuracy improves with larger and more diverse training datasets.
  
- **Unsupervised POS Tagging**: Eliminates the need for pre-tagged data. It uses clustering and statistical inference to automatically assign tags to words based on their context. This approach is useful in low-resource settings or when labeled corpora are unavailable.

## Features

- Supports UTF-8 encoded Devanagari Hindi input.
- Computes emission and transition matrices.
- Uses Viterbi decoding for most probable tag sequence.
- Outputs tagged text and evaluation metrics.

## Example Usage

```bash
# Supervised Mode
python supervised.py 0 ./data/hindi_testing.txt

# Unsupervised Mode
python unsupervised.py 0 ./data/hindi_testing.txt
