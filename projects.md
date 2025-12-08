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

## 🧠 Agent Elimination in Multi-Agent Reinforcement Learning

**Tech Stack:** Unity ML-Agents, C#, Python, PPO, SAC, POCA

Honours research project investigating whether agent elimination mechanisms can improve cooperation in Multi-Agent Reinforcement Learning environments. Built a custom Unity environment with configurable elimination rules (1-4 agents to eliminate, freeze vs. permanent), testing how agents balance self-interest against collective benefit in resource-sharing dilemmas. Deployed three RL algorithms (PPO, SAC, POCA) across 500K training steps to analyze emergent cooperative behavior.

**Key Achievements:**
- Custom Unity ML-Agents environment with 2,000+ lines of C# code
- Configurable elimination mechanics (freeze, permanent, group consensus)
- Advanced observation space with 20+ normalized features tracking agent states
- Hit tracking system using distributed HashSet for unique shooter detection
- Density-based food respawn logic inspired by DeepMind's Harvest environment
- Three RL algorithms tested: PPO (baseline), SAC (best performance), POCA (cooperation-focused)
- Published honours thesis with 10+ experimental graphs analyzing cooperation dynamics

[View Detailed Project](/projects/marl-elimination/)

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

**Tech Stack:** Python, TensorFlow/Keras, PRAW (Reddit API), yfinance, NLTK, VADER Sentiment

Developed a multi-input Recurrent Neural Network that predicts daily stock price percentage changes by fusing real-time financial data with sentiment analysis of Reddit posts. The model scrapes stock discussions from financial subreddits (r/stocks, r/options, r/investing), applies VADER sentiment analysis to capture crowd psychology, and combines this social signal with Yahoo Finance data to understand how investor sentiment drives market movements.

**Key Achievements:**
- Multi-input RNN architecture with three distinct branches (Title, Body, and Numerical Features)
- VADER sentiment analysis optimized for social media text (handles slang, emojis, negations)
- Date-synchronized data fusion correlating Reddit posts with same-day price movements
- Integration of two real-time APIs (PRAW, yfinance) for cohesive data collection
- LSTM layers process sequential nature of both social media text and financial time series
- Dual sentiment scoring (title + body separately) to capture nuanced investor psychology

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
