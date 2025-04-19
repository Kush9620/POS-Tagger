# POS Tagging for Hindi using Hidden Markov Models (HMM)

Part-of-Speech (POS) tagging for Indian languages, especially Hindi, remains an active area of research due to the complexity and flexibility of the language structure. This project presents a POS tagger based on the Hidden Markov Model (HMM) framework, offering both supervised and unsupervised learning approaches.

## 🌐 Overview

### What is POS Tagging?

Part-of-Speech (POS) tagging is the process of assigning a grammatical category (such as noun, verb, adjective, etc.) to each word in a sentence. In the context of Natural Language Processing (NLP), POS tagging is essential for tasks such as syntactic parsing, machine translation, information extraction, and more.

For example, in the sentence:  
**"राम ने आम खाया।"**  
The tags might be:
- `राम/NNP` (Proper Noun)
- `ने/PSP` (Postposition)
- `आम/NN` (Common Noun)
- `खाया/VB` (Verb)

---

## 🤖 How POS Tagging Works Using HMM

### Hidden Markov Model (HMM) Basics

A Hidden Markov Model is a statistical model that assumes the system being modeled is a Markov process with hidden states. In the case of POS tagging:

- **Observed States**: The words in a sentence.
- **Hidden States**: The corresponding POS tags.

HMM is defined by:
1. **Transition Probabilities (A)**: The probability of a tag following another tag.
   - Example: P(VB | NN) — the probability of a verb following a noun.
2. **Emission Probabilities (B)**: The probability of a word being associated with a tag.
   - Example: P(खाया | VB) — the probability that the word "खाया" is a verb.
3. **Initial Probabilities (π)**: The probability of a tag starting a sentence.


---

## ✅ Supervised POS Tagging

This approach uses a manually annotated Hindi corpus to train the HMM. The model learns:
- **Word-Tag Emission Probabilities**: How likely a word is to be tagged with a particular POS.
- **Tag Transition Probabilities**: How likely one tag follows another.

### Pros:
- High accuracy when trained on a large and diverse dataset.
- Learns language-specific syntax and semantics effectively.

### Cons:
- Requires a large, manually tagged dataset (costly to build).

---

## 🚫 Unsupervised POS Tagging

In low-resource settings where tagged corpora are unavailable, unsupervised POS tagging offers an alternative. It uses clustering and statistical inference to deduce the most probable tag sequences based on context and co-occurrence patterns.

### Techniques:
- Expectation Maximization (EM) for parameter estimation.
- Clustering words into syntactic categories based on usage.
- Bayesian inference methods (e.g., Gibbs Sampling).

### Pros:
- No need for labeled data.
- Useful for under-resourced languages or domains.

### Cons:
- Generally lower accuracy than supervised models.
- Requires careful tuning and large untagged corpora.

---

## 📈 Applications

- Machine Translation
- Question Answering Systems
- Syntactic Parsing
- Spell Correction
- Named Entity Recognition (NER)

---


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
