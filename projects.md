---
layout: page
title: Projects
permalink: /projects/
---

# My Projects

Here are some of the projects I've built. Each one taught me something new and pushed my skills further.

---

## 🤖 Autonomous Robot Jar Collection System

**Tech Stack:** Webots R2025a, Java, Computer Vision, Robotics, Sensor Fusion

Built a fully autonomous mobile robot system that navigates a 20×20 meter warehouse, detects honey jars using color-based computer vision, manipulates objects with a gripper, and places them in designated storage areas. The robot combines multiple sensor modalities (camera, compass, distance sensors, touch sensor) with real-time path planning to autonomously complete complex manipulation tasks.

**Key Achievements:**
- 100% success rate collecting 4 objects without human intervention
- ±3° heading accuracy and ±10cm position accuracy in 20m arena
- Multi-layer object detection with fallback logic for robust operation
- Real-time sensor fusion combining vision, compass, and distance data
- 800+ lines of optimized Java controller code

[View Detailed Project](/projects/robot-project/) | [View on GitHub](https://github.com/yourusername/robot-collection-system)

---

## 🎬 YouTube Views Prediction Using Neural Networks

**Tech Stack:** Python, TensorFlow/Keras, YouTube Data API, NLTK, NLP, Deep Learning

Developed a multi-input deep learning system that predicts YouTube video view counts by analyzing textual content (titles, descriptions, channel names), channel authority metrics, temporal features, and content categories. Built six progressive model variations to empirically determine feature importance, culminating in a bidirectional LSTM architecture that processes multiple data streams simultaneously.

**Key Achievements:**
- Multi-input neural network with 5 text embedding branches and numerical features
- Bidirectional LSTM layers capture forward/backward context in video metadata
- Integrated YouTube Data API with intelligent caching and rate limit management
- Comprehensive NLP pipeline with NLTK tokenization and stopword removal
- Tested...

---

## 📈 Predicting Stock Price Changes Using Reddit Sentiment

**Tech Stack:** Python, TensorFlow/Keras, NLTK, NLP, Deep Learning, Financial Modeling

Developed a multi-input Recurrent Neural Network (RNN) to predict a stock's daily price percentage change. [cite_start]The model fuses time-series financial data (Yahoo Finance) with real-time sentiment analysis of stock-related posts scraped from Reddit[cite: 611, 649]. [cite_start]By applying VADER sentiment analysis to post titles and bodies, the model learns the correlation between investor sentiment and market volatility[cite: 661, 662].

**Key Achievements:**
- [cite_start]Multi-input RNN architecture with three distinct branches (Title, Body, and Numerical Features)[cite: 500].
- [cite_start]Integration of three major real-time APIs (yfinance, PRAW, NLTK) for cohesive data collection[cite: 518].
- [cite_start]Sentiment-enhanced feature engineering that explicitly targets social factors in financial prediction[cite: 612, 652].
- [cite_start]Used LSTM layers to process the sequential nature of both text and time-series data[cite: 501].

[View Detailed Project](/projects/stock-predictor/) | [View on GitHub](https://github.com/yourusername/stock-predictor-nn)

---

## 📱 Morgan's Reviews - iOS App

**Tech Stack:** Swift, UIKit, MVC Pattern, Mobile Development

A native iOS application for tracking and reviewing entertainment media. Features a custom-built interactive star rating system, table view data management, and an admin authentication flow for content management. Designed to provide a spoiler-free environment for sharing honest reviews on movies and games.

**Key Achievements:**
- Developed a custom `@IBDesignable` UI control for star ratings with accessibility support.
- Implemented the Model-View-Controller (MVC) architectural pattern.
- Built a secure navigation flow with admin authentication logic.
- Integrated `UIImagePickerController` for handling user-generated media.

[View Detailed Project](/projects/morgans-reviews/) | [View on GitHub](https://github.com/yourusername/morgans-reviews)

