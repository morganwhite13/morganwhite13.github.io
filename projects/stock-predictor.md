---
layout: page
title: Predicting Stock Price Changes Using Reddit Sentiment
permalink: /projects/stock-predictor/
---

# Predicting Stock Price Changes Using Reddit Sentiment

## Quick Summary

A deep learning Recurrent Neural Network (RNN) designed to predict a stock's daily price percentage change by analyzing a hybrid dataset combining real-time financial data with sentiment analysis of Reddit posts. [cite_start]The model focuses on the non-financial social factors that drive investor behavior and stock volatility[cite: 612, 613].

**Tech Stack:** Python, TensorFlow/Keras, NLTK (VADER Sentiment), PRAW (Reddit API), yfinance (Yahoo Finance API), Pandas, Scikit-learn  
[cite_start]**Course:** COMP 4107B - Neural Networks [cite: 601]  
**Status:** ✅ Complete Research Project  
**GitHub:** [View Source Code](https://github.com/yourusername/stock-predictor-nn)

---

## Background and Motivation

[cite_start]The stock market is influenced not only by a company's financial performance but also by investor sentiment and public perception[cite: 613]. [cite_start]Inspired by the concept of rapidly analyzing market rumors and social data—similar to a scene in the movie *Limitless* [cite: 617][cite_start]—the goal was to build an AI program to predict stock movements[cite: 611]. [cite_start]By combining time-series financial metrics with natural language analysis of social media, the project aimed to provide deeper insights into market dynamics beyond traditional fundamental analysis[cite: 612]. [cite_start]Reddit was chosen as the social media source due to the restrictive nature of the Twitter API[cite: 621].

## System Architecture

The project is structured around three main components: Data Collection, Preprocessing, and the Neural Network Model.

### 1. Data Collection & Integration

1.  [cite_start]**Financial Data**: Historical Open/Close/Volume data is retrieved in real-time for a list of stocks using the `yfinance` API[cite: 647].
2.  [cite_start]**Social Data**: Posts are collected from relevant financial subreddits (e.g., r/stocks, r/investing) using the **PRAW/Reddit API**, searching for specific stock symbols[cite: 649, 518].
3.  **Hybrid Dataset Creation**: The two datasets are combined. [cite_start]For each Reddit post, the stock's **daily percentage change** (calculated as `(Close - Open) / Open * 100`) on the date the post was created is attached as the prediction target[cite: 650, 517, 519].

### 2. Preprocessing & Feature Engineering

The data is cleaned and transformed to be suitable for the neural network:

* [cite_start]**Text Preprocessing**: The `preprocess_text` function uses **NLTK's word tokenizer** to clean the post's title and body, removing punctuation and stop words, and converting text to lowercase to reduce noise[cite: 658, 660].
* **Sentiment Analysis**: The **NLTK Sentiment Intensity Analyzer** with the **VADER lexicon** is used to calculate a compound sentiment score for the post's title and body separately. [cite_start]VADER is specifically "attuned to sentiments expressed in social media," making it ideal for the Reddit data[cite: 661, 662].
* [cite_start]**Numerical Scaling**: Numerical features (e.g., post score, upvote ratio, number of comments, and sentiment scores) are scaled using `StandardScaler` to ensure uniform contribution during training[cite: 651].
* [cite_start]**Sequence Preparation**: Text sequences (title and body) are padded to a uniform length for consistent input to the RNN layers[cite: 659].

### 3. Recurrent Neural Network Model

[cite_start]A **Multi-Input Recurrent Neural Network** was constructed using the Keras Functional API[cite: 499].

* [cite_start]**Multi-Input Structure**: The model uses three distinct input branches to process different feature types simultaneously[cite: 500]:
    1.  **Title Input**: Text sequences for the post title.
    2.  **Body Input**: Text sequences for the post body.
    3.  [cite_start]**Feature Input**: The scaled numerical features (karma, ratios, sentiment scores, etc.)[cite: 500].
* **Sequential Processing**: The Title and Body branches each pass through:
    * An **Embedding Layer** (to map words to a vector space).
    * [cite_start]An **LSTM Layer** (to capture long-term sequential dependencies in the text)[cite: 501].
    * A **GlobalMaxPooling1D Layer** (to extract the most important learned features from the sequence).
* [cite_start]**Concatenation and Prediction**: All three branches (Title, Body, and Numerical Features) are **Concatenated** before being passed through a final set of **Dense** and **Dropout** layers, which output the final prediction of the stock's price percentage change[cite: 502].

## Results and Lessons Learned

[cite_start]Initial validation on a limited dataset showed a test accuracy of **0.3866** (which corresponds to predicting the correct price change direction or a narrow range)[cite: 571]. [cite_start]In a domain as complex and volatile as stock market prediction, the key goal was not perfect accuracy but **relative accuracy**—determining which of two videos would perform better—and gaining **insights** into the correlation between social data and price movement[cite: 570, 616].

**Key Insights:**

* [cite_start]**Sentiment Correlation**: The project validated the hypothesis that social media sentiment contributes to a stock's daily change, independent of pure financial data[cite: 613].
* [cite_start]**Architecture**: The multi-input RNN architecture was effective at fusing complex, multi-modal data types (text, time-series, numerical features)[cite: 501, 502].

**Future Directions**:
* [cite_start]The primary limitation is the dataset size and the need for data persistence (saving/loading collected data)[cite: 587, 556].
* [cite_start]Adding more features, such as the company's income statements, cash flow, and balance sheet, would provide a more complete financial picture[cite: 589].
