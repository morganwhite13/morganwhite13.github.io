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

[View Detailed Project](/projects/robot-project/)

---

## 🎬 YouTube Views Prediction Using Neural Networks

**Tech Stack:** Python, TensorFlow/Keras, YouTube Data API, NLTK, NLP, Transformer Architecture

Developed an advanced deep learning system that predicts YouTube video view counts using Transformer architecture with multi-head attention. The model analyzes textual content (titles, descriptions, channel names), channel authority, temporal features, and content categories. Enhanced with GloVe pretrained embeddings and sophisticated feature interaction layers that learn how different features amplify or modulate each other.

**Key Achievements:**
- Transformer encoder with multi-head attention and bidirectional context processing
- GloVe 300D pretrained embeddings for semantic understanding (6B token corpus)
- 5 learned feature interaction types (title×description, text×subscribers, category×channel, time×subscribers)
- Log-scale prediction handling 5+ orders of magnitude in view counts (1K to 100M+)
- Multi-layered regularization (L2, Dropout, BatchNorm, Early Stopping, LR scheduling)
- 1,500+ lines of production-grade Python code

[View Detailed Project](/projects/youtube-predictor/)

---

## 📈 Predicting Stock Price Changes Using Reddit Sentiment

**Tech Stack:** Python, TensorFlow/Keras, NLTK, NLP, Deep Learning, Financial Modeling

Developed a multi-input Recurrent Neural Network (RNN) to predict a stock's daily price percentage change. The model fuses time-series financial data (Yahoo Finance) with real-time sentiment analysis of stock-related posts scraped from Reddit. By applying VADER sentiment analysis to post titles and bodies, the model learns the correlation between investor sentiment and market volatility.

**Key Achievements:**
- Multi-input RNN architecture with three distinct branches (Title, Body, and Numerical Features)
- Integration of three major real-time APIs (yfinance, PRAW, NLTK) for cohesive data collection
- Sentiment-enhanced feature engineering that explicitly targets social factors in financial prediction
- Used LSTM layers to process the sequential nature of both text and time-series data

[View Detailed Project](/projects/stock-predictor/)

---

## 📱 Morgan's Reviews - iOS App

**Tech Stack:** Swift, UIKit, MVC Pattern, Mobile Development

A native iOS application for tracking and reviewing entertainment media. Features a custom-built interactive star rating system, table view data management, and an admin authentication flow for content management. Designed to provide a spoiler-free environment for sharing honest reviews on movies and games.

**Key Achievements:**
- Developed a custom `@IBDesignable` UI control for star ratings with accessibility support
- Implemented the Model-View-Controller (MVC) architectural pattern
- Built a secure navigation flow with admin authentication logic
- Integrated `UIImagePickerController` for handling user-generated media

[View Detailed Project](/projects/morgans-reviews/)
