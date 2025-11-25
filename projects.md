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
- Tested across 400+ videos with multiple dataset collection strategies
- Demonstrated clear impact of title patterns, channel size, and temporal factors on performance

[View Detailed Project](/projects/youtube-predictor/) | [View on GitHub](https://github.com/yourusername/youtube-view-predictor)

---

## 🚀 Real-Time Collaborative Code Editor

**Tech Stack:** React, Node.js, Socket.io, Monaco Editor, PostgreSQL

A web-based code editor that allows multiple users to write and execute code together in real-time. Features include syntax highlighting for 10+ languages, live cursors, and integrated chat.

**Key Features:**
- Real-time synchronization using WebSockets
- User authentication and session management
- Code execution in sandboxed environment
- Syntax highlighting and auto-completion

[View on GitHub](https://github.com/yourusername/collab-editor) | [Live Demo](https://collab-editor.demo.com)

---

## 🤖 Movie Recommendation System

**Tech Stack:** Python, Scikit-learn, Flask, Pandas, React

Built a content-based and collaborative filtering recommendation engine that suggests movies based on user preferences and viewing history. Achieved 87% accuracy on test dataset.

**Key Features:**
- Hybrid recommendation algorithm
- RESTful API for predictions
- Interactive frontend for exploring recommendations
- Visualization of recommendation patterns

[View on GitHub](https://github.com/yourusername/movie-recommender) | [Read More](/blog/2024/11/building-recommendation-system)

---

## 📊 Personal Finance Tracker

**Tech Stack:** Django, PostgreSQL, Chart.js, Bootstrap

A full-stack web application for tracking income, expenses, and financial goals. Includes data visualization, budget alerts, and export functionality.

**Key Features:**
- Secure user authentication
- Interactive charts and analytics
- Category-based expense tracking
- CSV import/export functionality
- Monthly budget planning tools

[View on GitHub](https://github.com/yourusername/finance-tracker)

---

## 🎮 Pathfinding Visualizer

**Tech Stack:** JavaScript, HTML5 Canvas, CSS

An interactive visualization tool for graph traversal and pathfinding algorithms including Dijkstra's, A*, BFS, and DFS. Users can draw obstacles and watch algorithms find the shortest path.

**Key Features:**
- Real-time algorithm visualization
- Multiple algorithm implementations
- Interactive grid with obstacles
- Speed control and step-through mode

[View on GitHub](https://github.com/yourusername/pathfinding-viz) | [Live Demo](https://yourusername.github.io/pathfinding-viz)

---

## 🌐 Weather Dashboard

**Tech Stack:** React, OpenWeather API, Tailwind CSS

A responsive weather dashboard that displays current conditions and forecasts for multiple cities. Features location detection and favorite cities management.

**Key Features:**
- Real-time weather data
- 5-day forecast
- Location-based detection
- Clean, responsive UI

[View on GitHub](https://github.com/yourusername/weather-dashboard) | [Live Demo](https://weather.demo.com)

---

## Open Source Contributions

I also contribute to open-source projects. Some of my contributions include:
- **Project XYZ:** Added feature for bulk data processing (PR #456)
- **Library ABC:** Fixed critical bug in authentication flow (PR #789)
- **Framework DEF:** Improved documentation with examples (PR #123)
