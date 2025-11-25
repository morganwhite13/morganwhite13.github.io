---
layout: post
title: "Project Deep Dive: Predicting Stock Changes with Reddit Sentiment"
date: 2025-11-25 10:00:00 -0500
categories: machine-learning nlp finance deep-learning
---

[cite_start]I've just finished a major project for my **COMP 4107B - Neural Networks** course [cite: 601][cite_start], focused on the challenging task of **stock price prediction**[cite: 599]. [cite_start]The project, "Predicting Price Changes Using Reddit Posts," explores the idea that market movements are driven as much by **investor sentiment** as they are by fundamental financial data[cite: 613].

## The Hypothesis: Social Sentiment Matters

[cite_start]The core of this project was to test whether a deep learning model could find a correlation between the sentiment expressed on social media and a stock's daily price change[cite: 612].

I built a system that:
1.  [cite_start]**Collects real-time financial data** using the `yfinance` API[cite: 647].
2.  [cite_start]**Scrapes stock-related posts** from subreddits using the PRAW API[cite: 649, 518].
3.  [cite_start]**Analyzes the text** using the social-media-attuned **VADER lexicon** in NLTK to assign sentiment scores[cite: 661, 662].
4.  [cite_start]**Trains a Multi-Input Recurrent Neural Network (RNN)** on this hybrid dataset to predict the stock's percentage change for that day[cite: 502, 654].

[cite_start]The resulting model uses a complex architecture with separate input branches for the post title, post body, and numerical features, all connected by **LSTM layers** for sequence processing[cite: 500, 501].

## Check out the Full Details

This was a fascinating journey into fusing Natural Language Processing (NLP) with time-series financial modeling. For a complete breakdown of the multi-input neural network architecture, data preparation pipeline, and key insights learned, check out the dedicated project page:

**[View Full Project Documentation](/projects/stock-predictor/)**

**Tools Used:** Python, TensorFlow/Keras, NLTK, PRAW, yfinance
