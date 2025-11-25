---
layout: page
title: YouTube Views Prediction Using Neural Networks
permalink: /projects/youtube-predictor/
---

# YouTube Views Prediction Using Neural Networks

## Quick Summary

A deep learning system that predicts YouTube video view counts using natural language processing and multi-input neural networks. The model analyzes video titles, descriptions, channel metadata, and temporal features to forecast video performance, achieving meaningful predictions across diverse video categories.

**Tech Stack:** Python, TensorFlow/Keras, YouTube Data API, NLTK, Scikit-learn  
**Status:** ✅ Complete Research Project (Fall 2024)  
**GitHub:** [View Source Code](https://github.com/yourusername/youtube-view-predictor)

---

## The Challenge

YouTube's algorithm and viewer behavior create a complex ecosystem where seemingly similar videos can receive wildly different view counts. Understanding what drives video performance is valuable for content creators, but the factors are numerous and interrelated:

- **Textual signals:** Title wording, description content, channel branding
- **Temporal factors:** Publication date, video age, trending cycles
- **Channel authority:** Subscriber count, historical performance
- **Content categorization:** Video category, topic clustering

**The Goal:** Build a neural network that can predict view counts by learning patterns from these multi-modal features, providing insights into what makes videos successful.

---

## System Architecture

### Data Collection Pipeline

Built a flexible data collection system using the YouTube Data API v3:

**Collection Strategies:**
- `getRandomChannelsVideos()` - Fetches random videos, then collects from their channels for distribution
- `getPopularChannelsVideos()` - Analyzes trending videos and their channels for high-performance patterns
- `getChannelsVideos()` - Targets specific channels for controlled datasets
- `combineDatasets()` - Merges multiple collection runs while removing duplicates

**Data Points Collected:**
```python
{
    'videoId': 'dQw4w9WgXcQ',
    'title': 'Video Title Here',
    'description': 'Full video description...',
    'channelTitle': 'Channel Name',
    'channelId': 'UCxxxxxxx',
    'subscriberCount': 1000000,
    'categoryId': '24',
    'publishedAt': '2023-01-01T00:00:00Z',
    'views': 5000000
}
```

**Filtering Logic:**
- Minimum 1,000 views (ensures real engagement)
- English language only (defaultAudioLanguage check)
- Valid statistics (handles missing data gracefully)

---

## Machine Learning Pipeline

### 1. Natural Language Processing

**Text Preprocessing (`preprocessText`):**
```python
def preprocessText(text):
    text = text.lower()                          # Normalize case
    text = remove_punctuation(text)               # Clean symbols
    tokens = word_tokenize(text)                  # NLTK tokenization
    tokens = remove_stopwords(tokens)             # Filter common words
    return ' '.join(tokens)
```

**Techniques Applied:**
- **Tokenization:** NLTK word tokenizer breaks text into meaningful units
- **Stop Word Removal:** Removes "the", "is", "and" etc. using NLTK + sklearn stopwords
- **Normalization:** Lowercase conversion for consistency
- **Sequence Padding:** Ensures uniform input length for neural network

### 2. Feature Engineering

**Numerical Features:**
- **Subscriber Count:** Scaled using StandardScaler to prevent magnitude dominance
- **Days Since Publication:** Calculated as `(today - publishedAt).days`, then scaled
- Both features normalized to mean=0, std=1 for uniform contribution

**Textual Features:**
- **Title:** Tokenized → Sequences → Padded (max length determined from dataset)
- **Description:** Same process as title, separate tokenizer
- **Channel Title:** Captures brand recognition patterns
- **Category ID:** Converted to sequences (e.g., "24" → Entertainment)

### 3. Neural Network Architecture

Built using Keras Functional API for multi-input processing:
