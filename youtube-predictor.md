layout: page
title: YouTube Views Prediction Using Neural Networks
permalink: /projects/youtube-predictor/
YouTube Views Prediction Using Neural Networks
Quick Summary
A deep learning system that predicts YouTube video view counts using natural language processing and multi-input neural networks. The model analyzes video titles, descriptions, channel metadata, and temporal features to forecast video performance, achieving meaningful predictions across diverse video categories.
Tech Stack: Python, TensorFlow/Keras, YouTube Data API, NLTK, Scikit-learn
Status: ✅ Complete Research Project (Fall 2024)
GitHub: View Source Code

The Challenge
YouTube's algorithm and viewer behavior create a complex ecosystem where seemingly similar videos can receive wildly different view counts. Understanding what drives video performance is valuable for content creators, but the factors are numerous and interrelated:

Textual signals: Title wording, description content, channel branding
Temporal factors: Publication date, video age, trending cycles
Channel authority: Subscriber count, historical performance
Content categorization: Video category, topic clustering

The Goal: Build a neural network that can predict view counts by learning patterns from these multi-modal features, providing insights into what makes videos successful.

System Architecture
Data Collection Pipeline
Built a flexible data collection system using the YouTube Data API v3:
Collection Strategies:

getRandomChannelsVideos() - Fetches random videos, then collects from their channels for distribution
getPopularChannelsVideos() - Analyzes trending videos and their channels for high-performance patterns
getChannelsVideos() - Targets specific channels for controlled datasets
combineDatasets() - Merges multiple collection runs while removing duplicates

Data Points Collected:
python{
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
Filtering Logic:

Minimum 1,000 views (ensures real engagement)
English language only (defaultAudioLanguage check)
Valid statistics (handles missing data gracefully)


Machine Learning Pipeline
1. Natural Language Processing
Text Preprocessing (preprocessText):
pythondef preprocessText(text):
    text = text.lower()                          # Normalize case
    text = remove_punctuation(text)               # Clean symbols
    tokens = word_tokenize(text)                  # NLTK tokenization
    tokens = remove_stopwords(tokens)             # Filter common words
    return ' '.join(tokens)
Techniques Applied:

Tokenization: NLTK word tokenizer breaks text into meaningful units
Stop Word Removal: Removes "the", "is", "and" etc. using NLTK + sklearn stopwords
Normalization: Lowercase conversion for consistency
Sequence Padding: Ensures uniform input length for neural network

2. Feature Engineering
Numerical Features:

Subscriber Count: Scaled using StandardScaler to prevent magnitude dominance
Days Since Publication: Calculated as (today - publishedAt).days, then scaled
Both features normalized to mean=0, std=1 for uniform contribution

Textual Features:

Title: Tokenized → Sequences → Padded (max length determined from dataset)
Description: Same process as title, separate tokenizer
Channel Title: Captures brand recognition patterns
Category ID: Converted to sequences (e.g., "24" → Entertainment)

3. Neural Network Architecture
Built using Keras Functional API for multi-input processing:
Architecture Overview:
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
├──────────┬──────────┬──────────┬──────────┬────────────┤
│  Title   │Description│ Channel │ Category │  Numerical │
│ (text)   │  (text)   │  (text) │  (int)   │ (scaled)   │
└────┬─────┴─────┬─────┴────┬────┴────┬─────┴─────┬──────┘
     │           │          │         │           │
     ▼           ▼          ▼         ▼           │
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│Embedding│ │Embedding│ │Embedding│ │Embedding│  │
│ 300-dim │ │ 300-dim │ │ 300-dim │ │ 300-dim │  │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘  │
     │           │          │         │           │
     ▼           ▼          ▼         ▼           │
┌──────────┐┌──────────┐┌──────────┐┌──────────┐ │
│Bi-LSTM   ││Bi-LSTM   ││Bi-LSTM   ││Bi-LSTM   │ │
│128 units ││128 units ││128 units ││128 units │ │
└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘ │
     │           │          │         │           │
     ▼           ▼          ▼         ▼           │
┌──────────┐┌──────────┐┌──────────┐┌──────────┐ │
│ GlobalMax││ GlobalMax││ GlobalMax││ GlobalMax│ │
│ Pooling  ││ Pooling  ││ Pooling  ││ Pooling  │ │
└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘ │
     │           │          │         │           │
     └───────────┴──────────┴─────────┴───────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Concatenate   │
                 │  All Features │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  Dense (512)  │
                 │   ReLU        │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  Dropout(0.3) │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  Dense (256)  │
                 │   ReLU        │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  Dropout(0.3) │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  Dense (1)    │
                 │   Linear      │
                 └───────────────┘
                         │
                         ▼
                  Predicted Views
Key Components:
Embedding Layers (300-dimensional):

Convert categorical text tokens into dense vector representations
Learns semantic relationships between words
Separate embeddings for title, description, channel, category

Bidirectional LSTM (128 units):

Processes sequences in both forward and backward directions
Captures temporal dependencies in text
return_sequences=True maintains full sequence output

GlobalMaxPooling1D:

Extracts most important features from each sequence
Reduces dimensionality while preserving critical information
One pooling layer per text input branch

Dense Layers with Dropout:

512 → Dropout(0.3) → 256 → Dropout(0.3) → 1
ReLU activation for non-linearity
Dropout prevents overfitting by randomly disabling neurons during training

Loss Function: Mean Squared Error (MSE)

Appropriate for regression tasks
Measures average squared difference between predicted and actual views
Formula: MSE = (1/n) × Σ(predicted - actual)²

Optimizer: Adam

Adaptive learning rate optimization
Combines benefits of RMSprop and momentum
Default learning rate with automatic adjustment


Model Variations & Evolution
I developed five progressive model versions to understand feature importance:
Version 1: Title Only (neuralTitleModel)

Features: Video title (text)
Purpose: Baseline - can title alone predict views?
Architecture: Single embedding → Bi-LSTM → Dense layers

Version 2: Title + Subscribers (neuralTitleSubscriberModel)

Features: Title + channel subscriber count
Insight: Tests if channel size impacts prediction accuracy
Result: Significant improvement - established channels have predictable performance

Version 3: Title + Subscribers + Date (neuralTSDateModel)

Features: Title + subscribers + days since publication
Insight: Temporal dimension matters - older videos accumulate more views
Architecture: Introduced multi-input design with Functional API

Version 4: + Description (neuralTSDDescriptionModel)

Features: All above + video description
Insight: Description quality affects discoverability and click-through
Improvement: Better predictions for educational/tutorial content

Version 5: + Channel Title (neuralTSDDChannelModel)

Features: All above + channel branding
Insight: Channel name recognition impacts view potential
Example: "MrBeast" vs. "Random Channel" - brand matters

Version 6: Complete Model (neuralAllModel) ⭐

Features: Title + Description + Channel + Category + Subscribers + Date
Result: Best overall performance
Training: 10 epochs, batch size 32, 80/20 train-test split


Results & Performance
Quantitative Analysis
Dataset Comparison (MSE on 20% test set):
Dataset TypeVideosMSECharacteristicsRandom Channels400 (20×20)3.3 trillionWide distribution, thousands to millions of viewsPopular Channels400 (20×20)1.6 quadrillionHigh-view videos (millions+), larger error magnitudeSpecific Channels400 (20×20)1.8 quadrillionCurated variety, overfitting on limited data
Understanding the MSE:

Large MSE values reflect the scale of view counts (millions)
MSE scales quadratically with prediction error
Example: 1M view error on 10M video = 1 trillion MSE contribution

Qualitative Insights
What the Model Learned:
✅ Title Patterns That Work:

Numbered lists ("Top 10...")
Comparisons ("$1 vs $1000...")
Time challenges ("100 Days...")
Question formats ("What If...?")

✅ Channel Authority Matters:

Larger subscriber counts → higher predicted views
Brand recognition (channel name) influences predictions
Consistent style across channel improves accuracy

✅ Category Differences:

Gaming (cat 20) vs Entertainment (cat 24) have different view patterns
Category affects baseline expected performance

✅ Temporal Effects:

Older videos accumulate more views
Diminishing returns after certain threshold
Recent uploads need time to reach potential

Example Predictions
Test Case 1:
Title: "I Survived 100 Days in Canada"
Description: "This is an amazing journey!"
Channel: "Random" (1M subscribers)
Category: 24 (Entertainment)
Days Old: 700 days
Predicted Views: ~2.5M
Test Case 2:
Title: "$1 VS $1,000 Water"
Description: "Which one is better? Let's find out!"
Channel: "Money Man" (500K subscribers)
Category: 20 (Gaming)
Days Old: 1000 days
Predicted Views: ~1.8M
Sensitivity Analysis:

90x subscriber increase → ~8x view prediction increase
Title change (generic → viral format) → ~3x increase
Category change → ±30% variation
Date (older video) → ~40% increase


Technical Achievements
API Integration Excellence
✅ Built robust YouTube API wrapper with error handling
✅ Implemented caching system (checks for existing JSON before re-fetching)
✅ Language filtering ensures English-only dataset
✅ Subscriber count enrichment via secondary API calls
✅ Deduplication across multiple data collection runs
NLP Pipeline
✅ Multi-library stopword removal (NLTK + sklearn)
✅ Dynamic sequence length determination
✅ Separate tokenizers for different text types
✅ Efficient preprocessing with functional approach
Neural Network Design
✅ Multi-input architecture using Keras Functional API
✅ Parallel processing of text and numerical features
✅ Bidirectional LSTM for context understanding
✅ Proper regularization (dropout) to prevent overfitting
✅ Feature scaling for numerical inputs
Software Engineering
✅ Modular function design (6 model variants)
✅ Configurable hyperparameters
✅ JSON-based data persistence
✅ Comprehensive inline documentation

Challenges & Solutions
Challenge 1: API Rate Limits
Problem: YouTube Data API has strict quota limits
Impact: Couldn't gather massive datasets (thousands of videos)
Solution:

Implemented caching system to avoid redundant API calls
Created multiple collection strategies to maximize diversity
JSON persistence for reusable datasets

Challenge 2: High MSE Values
Problem: Mean Squared Error in trillions looks alarming
Context: View counts range from 1,000 to 100,000,000+
Insight:

MSE = squared differences → magnifies large-scale predictions
A 2M error on 10M video = 4 trillion contribution alone
Relative error more meaningful than absolute MSE

Solution: Focused on qualitative validation and relative ranking
Challenge 3: Feature Imbalance
Problem: Text features (10,000+ dimensions) vs numerical features (2)
Solution:

Used StandardScaler to normalize numerical inputs
Separate embedding dimensions for each text type
Dense layers for numerical features before concatenation

Challenge 4: Limited Dataset Size
Problem: Only ~400 videos per dataset (API constraints)
Impact: Risk of overfitting, especially on specific channel data
Mitigation:

Dropout layers (0.3) for regularization
20% validation split during training
Multiple dataset collection strategies for diversity


Key Learnings
About Machine Learning
Model Complexity vs. Performance:

Adding features monotonically improved predictions
Diminishing returns after 5-6 features
NLP features (title, description) most impactful

Architecture Decisions:

Bidirectional LSTM crucial for capturing context
GlobalMaxPooling better than flattening for variable-length sequences
Dropout essential given limited training data

Loss Function Reality:

MSE appropriate for regression but intimidating at scale
Consider mean absolute percentage error (MAPE) for future work
Relative ranking accuracy > absolute view prediction

About YouTube
Content Patterns:

Viral formats are recognizable and repeatable
Click-worthy titles follow patterns (lists, comparisons, challenges)
Channel authority (subscribers) strongly correlates with views
Category matters - different niches have different scales

Temporal Dynamics:

View accumulation isn't linear
Long-term videos benefit from search traffic
Recent videos need time to reach equilibrium

About Software Engineering
API Design:

Caching is critical for iterative development
Error handling must account for missing fields
Rate limiting requires thoughtful data strategy

Modular Development:

Starting simple (title-only model) then adding complexity
Each model version isolated in separate function
Easy to compare feature importance through variants


Future Improvements
Short-Term Enhancements
Larger Dataset:

Invest in YouTube API quota increase
Target 10,000+ videos across 100+ channels
Would significantly reduce overfitting

Additional Features:

Thumbnail analysis (computer vision on image)
Video duration (length affects completion rate)
Upload time (day of week, time of day patterns)
Tags/keywords (additional metadata)

Better Evaluation Metrics:

Mean Absolute Percentage Error (MAPE)
R² score for explained variance
Relative ranking accuracy (can model order videos correctly?)

Long-Term Vision
Real-Time Prediction Interface:

Web app where creators input video metadata
Instant view prediction with confidence intervals
A/B testing different titles/descriptions

Transfer Learning:

Pre-trained language models (BERT, GPT embeddings)
Could improve text understanding significantly
Require more computational resources

Time Series Component:

Predict view trajectory over time (not just final count)
LSTM for temporal view patterns
Account for viral spikes vs. steady growth

Multi-Objective Optimization:

Predict views AND engagement (likes, comments)
Multi-output neural network
Understand trade-offs between metrics


Technologies & Tools
Programming & ML:

Python 3.x
TensorFlow 2.x / Keras
NumPy, Pandas

NLP & Text Processing:

NLTK (Natural Language Toolkit)
Scikit-learn (TfidfVectorizer, stopwords)

Data Collection:

Google API Client Library
YouTube Data API v3

Data Processing:

StandardScaler (feature normalization)
Tokenizer (text → sequences)
pad_sequences (uniform input size)

Model Architecture:

Keras Functional API
Bidirectional LSTM
Embedding layers
Dense/Dropout layers


Project Structure
youtube-view-predictor/
├── projectFinal.py           # Main implementation
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/                     # Cached datasets
│   ├── randomChannelsVideos8.json
│   ├── popularChannelsVideos4.json
│   ├── channelsVideos3.json
│   └── combinedVideos3.json
└── models/                   # Saved model weights (optional)

How to Run
1. Install Dependencies:
bashpip install numpy pandas google-api-python-client nltk scikit-learn tensorflow
python -m nltk.downloader punkt stopwords
2. Set Up API Key:
python# In projectFinal.py, replace with your key:
API_KEY = 'your_youtube_api_key_here'
3. Collect Data:
python# Choose a collection method:
df = pd.DataFrame(getRandomChannelsVideos())
# or
df = pd.DataFrame(getPopularChannelsVideos())
4. Train Model:
pythonneuralAllModel(df)  # Complete model with all features
5. Make Predictions:
pythonpredicted_views = predict_view_count(
    title="Amazing Video Title",
    description="Interesting description here",
    channel_title="Your Channel",
    category="24",
    subscriber_count=100000,
    days_since_publication=30
)

Academic Context
Course: COMP 3106 - Introduction to Artificial Intelligence
Institution: Carleton University
Semester: Fall 2024
Project Type: Individual Research Project

Conclusion
This project successfully demonstrates that YouTube view counts can be partially predicted using machine learning, despite the inherent complexity of human behavior and algorithmic recommendations. While absolute accuracy is limited by dataset size and API constraints, the model effectively learns:

Textual patterns that indicate viral potential
Channel authority as a performance multiplier
Category-specific viewing behaviors
Temporal accumulation of views over time

The modular design allows for easy experimentation with feature combinations, and the insights gained provide actionable intelligence for content creators. Most importantly, the project showcases end-to-end ML engineering: from API integration and data collection, through NLP preprocessing and feature engineering, to deep learning model development and evaluation.
Key Takeaway: Even with limited data, careful feature engineering and appropriate model architecture can extract meaningful patterns from complex real-world phenomena.
