---
title: Predicting Stock Price Changes Using Reddit Sentiment
permalink: /projects/stock-predictor/
description: "A multi-input RNN that predicts daily stock price changes by combining Yahoo Finance data with VADER sentiment from Reddit posts."
---

## Quick Summary

A multi-input Recurrent Neural Network that predicts daily stock price percentage changes by fusing real-time financial data with sentiment analysis of Reddit posts. The system scrapes stock discussions from financial subreddits, applies VADER sentiment analysis, and combines this social signal with Yahoo Finance data to test whether investor sentiment helps predict market movements.

**Tech Stack:** Python, TensorFlow/Keras, PRAW (Reddit API), yfinance, NLTK, VADER Sentiment  
**Status:** ✅ Complete and Operational  
**GitHub:** [View Source Code](https://github.com/morganwhite13/Predicting-Stock-Price-Changes-Using-Reddit-Sentiment)

---

## The Challenge

Stock prices are influenced by more than just company fundamentals—investor sentiment, social media buzz, and crowd psychology play massive roles in short-term price movements. The challenge was to build a system that could:

- Capture real-time social sentiment from Reddit's financial communities
- Combine textual data (post titles/bodies) with numerical features (karma, comments)
- Correlate social signals with actual price movements on the same day
- Handle the noisy, unpredictable nature of both social media and stock markets
- Predict percentage changes rather than absolute prices (more actionable for traders)

This required bridging two very different data sources—structured financial time series and unstructured social media text—into a unified prediction framework.

---

## Inspiration: The Limitless Approach

This project was inspired by the movie *Limitless* (2011), where Bradley Cooper's character uses enhanced mental abilities to analyze not just financial data, but rumors, social sentiment, and public perception to turn $10,000 into $2,000,000 in a week through day trading.

**Key Scene:** [Limitless Day Trading Scene](https://www.youtube.com/watch?v=XMQC01n7hHo)

The idea: **What if we could build an AI that thinks like that?** One that doesn't just look at price charts, but understands what people are *saying* about stocks in real-time.

---

## System Architecture

### Dual Data Pipeline

**Financial Data Stream (Yahoo Finance API):**
- Historical stock data: Open, High, Low, Close, Adj Close, Volume
- Calculated metric: Daily percentage change = (Close - Open) / Open × 100
- Real-time updates for any stock symbol
- Date-indexed for correlation with social data

**Social Media Stream (Reddit API via PRAW):**
- Scrapes posts from financial subreddits: r/stocks, r/options, r/investing
- Searches for specific stock symbols (e.g., AAPL, TSLA, NVDA)
- Captures: Title, body text, karma score, upvote ratio, comment count, timestamp
- Filters for posts created on same date as trading day

**Data Fusion:**
```python
for post in subreddit.search(symbol, limit=limit):
    post_date = datetime.fromtimestamp(post.created_utc)
    # Find matching stock data for same date
    percentage_change = (row['Close'] - row['Open']) / row['Open'] * 100
    # Attach to post data
```

The result: **Every Reddit post is paired with the actual price change that occurred on that day.**

---

## Algorithm Highlights

### 1. VADER Sentiment Analysis

**Why VADER?**
- Specifically designed for social media text
- Understands slang, emojis, and informal language
- Handles negations ("not good" vs "good")
- Intensity-aware ("AMAZING!!!" > "good")

**Implementation:**

```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def calculate_sentiment(text):
    return sia.polarity_scores(text)['compound']

# Applied separately to titles and bodies
reddit_data['Title Sentiment'] = reddit_data['title'].apply(calculate_sentiment)
reddit_data['Body Sentiment'] = reddit_data['body'].apply(calculate_sentiment)
```

**Example Outputs:**
- "TSLA to the moon! 🚀🚀🚀" → +0.87 (very positive)
- "Company reports disappointing earnings" → -0.52 (negative)
- "Holding my shares, unsure about future" → +0.12 (neutral-slight positive)

**Why Two Separate Scores?**
- Titles often sensationalized, bodies more detailed
- Model learns which to trust more
- Captures contradiction (clickbait title vs. skeptical body)

### 2. Text Preprocessing Pipeline

**NLTK-Based Cleaning:**

```python
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    words = [word.lower() for word in tokens 
             if word.isalnum() and 
             word.lower() not in stopwords.words('english')]
    return ' '.join(words)
```

**Steps:**
1. Tokenization → Split text into words
2. Lowercase normalization → "Apple" = "apple"
3. Alphanumeric filtering → Remove punctuation
4. Stop word removal → Remove "the", "and", "is", etc.
5. Rejoin → Create clean string

**Before:** "I'm buying $AAPL because the new iPhone is AMAZING!!!"  
**After:** "buying aapl new iphone amazing"

### 3. Multi-Input RNN Architecture

**Three Distinct Input Branches:**

```
Input Branch 1: Post Titles
├── Embedding Layer (vocab_size → 100D)
├── LSTM Layer (64 units, return_sequences=True)
└── GlobalMaxPooling1D

Input Branch 2: Post Bodies
├── Embedding Layer (vocab_size → 100D)
├── LSTM Layer (64 units, return_sequences=True)
└── GlobalMaxPooling1D

Input Branch 3: Numerical Features
└── [score, ratio, num_comments, title_sentiment, 
    body_sentiment, date, symbol_index]

Concatenation Layer
├── Merges all three branches

Dense Layers
├── Dense(32, activation='relu')
├── Dropout(0.2)
└── Dense(1, activation='linear') → Price change prediction
```

**Why This Architecture?**

**Separate Text Branches:**
- Titles and bodies have different vocabulary distributions
- Model learns importance of each independently
- Captures relationship between them

**LSTM Layers:**
- Designed for sequential data (time series)
- Maintains memory of previous words
- Captures context (e.g., "not good" vs "good")

**GlobalMaxPooling1D:**
- Extracts most important feature from sequence
- Reduces dimensionality
- Highlights key words that drive sentiment

**Dropout (0.2):**
- Randomly deactivates 20% of neurons during training
- Prevents overfitting on limited data
- Improves generalization

### 4. Feature Engineering

**Numerical Feature Processing:**

```python
scaler = StandardScaler()
reddit_data[['score', 'ratio', 'num_comments', 
             'Title Sentiment', 'Body Sentiment', 
             'date', 'symbolIndex']] = scaler.fit_transform(...)
```

**Why StandardScaler?**
- Karma scores: range from 1 to 10,000+
- Sentiment: range from -1 to +1
- Without scaling: karma dominates, sentiment ignored
- After scaling: all features contribute equally

**Engineered Features:**
- **Symbol Index** - Categorical encoding of stock tickers
- **Date as Numeric** - Unix timestamp captures temporal patterns
- **Score** - Post karma (community validation)
- **Upvote Ratio** - Agreement level (controversial vs. consensus)
- **Comment Count** - Engagement level (how much discussion)

### 5. Sequence Padding Strategy

**Problem:** Posts vary wildly in length
- Short title: "Buy AAPL?" → 2 tokens
- Long body: 500+ word analysis → 500+ tokens

**Solution:** Dynamic padding

```python
max_len_title = max(len(seq) for seq in X_title)
max_len_body = max(len(seq) for seq in X_body)

X_title_pad = [seq + [0] * (max_len_title - len(seq)) 
               for seq in X_title]
X_body_pad = [seq + [0] * (max_len_body - len(seq)) 
              for seq in X_body]
```

**Result:** All sequences padded to same length with zeros (LSTM ignores padding)

### 6. Training Configuration

**Optimizer:** Adam (learning_rate=0.001)
- Adaptive learning rates for each parameter
- Fast convergence
- Handles sparse gradients well

**Loss Function:** Mean Squared Error (MSE)
- Regression task (predicting continuous percentage)
- Penalizes large errors more heavily
- Standard for financial prediction

**Training Parameters:**
- Epochs: 30
- Batch size: 32
- Train/validation split: 80/20
- Early stopping if validation loss plateaus

---

## Key Technical Achievements

✅ **Dual-API Integration** - Real-time data from Reddit (PRAW) and Yahoo Finance (yfinance)  
✅ **VADER Sentiment Analysis** - Social media-optimized sentiment detection  
✅ **Multi-Input RNN** - Three separate branches for titles, bodies, and numerical features  
✅ **Date-Based Data Fusion** - Correlates social posts with same-day price movements  
✅ **LSTM Sequence Processing** - Captures temporal patterns and context  
✅ **Dynamic Dataset Generation** - Configurable stocks and subreddits  
✅ **Feature Scaling** - StandardScaler for numerical feature normalization  
✅ **Text Tokenization** - Vocabulary-based encoding for neural network input  

---

## Performance Results

### Experiment 1: Multi-Stock Portfolio

**Configuration:**
- Stocks: AAPL, MSFT, GOOGL, AMZN, META, NFLX, TSLA, NVDA, INTC, AMD (10 stocks)
- Posts per stock: 10
- Subreddits: r/stocks, r/options, r/investing
- Training: 30 epochs, batch size 32

**Results:**
- **Training MSE:** 0.3866
- **Validation MSE:** 0.1785

**Analysis:**
- Validation error is lower than training error, which is unusual. Part of this is dropout (active during training, off during validation), but see the caveat below: the split also lets near-duplicate examples into the validation set.

### Experiment 2: Single-Stock Deep Dive

**Configuration:**
- Stock: AAPL only
- Posts: 100
- Subreddits: r/stocks, r/options, r/investing
- Training: 30 epochs, batch size 32

**Results:**
- **Training MSE:** 0.0602
- **Validation MSE:** 0.0024

**Analysis:**
- Validation error is about 25× *lower* than training error. That is not a sign of a good model: it is the classic signature of data leakage between the training and validation sets (explained below).

### Caveat: How Much to Trust These Numbers

Looking back, the evaluation has a flaw worth being upfront about:

- **Leakage through shared targets.** The data is split randomly by post (`train_test_split`, 80/20). Every post about the same stock on the same day has the *same* target (that day's price change), so the validation set ends up with posts whose answers the model has effectively already seen in training. This is strongest in the AAPL-only run, where 100 posts cover a small number of trading days.
- **No baseline.** The MSE values aren't compared to a trivial predictor, such as always predicting 0% change or the training mean, so they don't show whether sentiment adds any signal.
- **Same-day timing.** Posts are matched to the price change of the day they were written, so some posts may be *reacting* to the move rather than predicting it.

A fair test would split by date (train on earlier days, validate on later ones), report the error of a naive baseline alongside the model's, and only use posts written before the market opens.

### Features Used

The model sees each post's title and body text, their VADER sentiment scores, and its karma score, comment count and upvote ratio. I haven't measured how much each feature contributes; an ablation study (retraining without each feature under a date-based split) would be the way to find out.

---

## Challenges & Solutions

### Challenge: API Rate Limits

**Problem:** Reddit API limits requests, Yahoo Finance throttles high-frequency queries

**Solution:** Sequential loading with built-in delays

```python
for symbol in symbols:
    data = yf.download(symbol)  # Respects rate limits
    for subreddit_name in subreddits:
        for post in subreddit.search(symbol, limit=limit):
            # Process posts one at a time
```

**Impact:** Slow data collection (100 posts ≈ 5-10 minutes) but reliable

**Future Improvement:** Implement caching system to save/load historical data

---

### Challenge: Date Mismatches

**Problem:** Reddit posts timestamped in UTC, stock data in market timezone, weekends/holidays have no trading

**Solution:** Convert timestamps and validate trading days

```python
post_date = datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d')

# Find matching stock data
for index, row in stock_data.iterrows():
    if index.strftime('%Y-%m-%d') == post_date:
        percentage_change = ((row['Close'] - row['Open']) / row['Open']) * 100
```

**Result:** Only posts on valid trading days included in dataset

---

### Challenge: Vocabulary Size Explosion

**Problem:** Combined title + body vocabulary → 50,000+ unique words

**Solution:** Separate tokenizers for titles and bodies

```python
tokenizer_title = Tokenizer()
tokenizer_title.fit_on_texts(reddit_data['processed_title'])

tokenizer_body = Tokenizer()
tokenizer_body.fit_on_texts(reddit_data['processed_body'])

# Combined vocab for embedding layer
combined_vocab = dict(tokenizer_title.word_index, 
                      **tokenizer_body.word_index)
vocab_size = len(combined_vocab) + 1
```

**Result:** Efficient vocabulary management, reduced memory usage

---

### Challenge: Noisy Social Data

**Problem:** Reddit posts full of sarcasm, jokes, irrelevant content, spam

**Solution:** Multi-layered filtering
- VADER handles some sarcasm via punctuation/caps
- Karma score filters low-quality posts
- Upvote ratio identifies controversial posts
- Comment count shows genuine engagement
- Model learns to weight features appropriately

**Result:** Noise reduced but not eliminated (inherent challenge in sentiment analysis)

---

### Challenge: Extreme Class Imbalance

**Problem:** Most days have small price changes (±2%), rare days have huge swings (±10%)

**Solution:** MSE loss function
- Naturally emphasizes larger errors
- Model learns to predict magnitude, not just direction
- Continuous output (not binary classification)

**Alternative Explored:** Could use weighted loss to prioritize large movements

---

## Code Architecture

### Main Components

```
StockPredictorFinal.py (600+ lines)
├── API Integration
│   ├── Reddit API (PRAW) - Post scraping
│   └── Yahoo Finance (yfinance) - Stock data
│
├── Data Collection Functions
│   ├── get_reddit_posts() - Multi-subreddit scraping
│   ├── get_stock_data() - Historical price data
│   └── get_stock_data_date() - Date-filtered queries
│
├── Preprocessing Functions
│   ├── preprocess_text() - NLTK text cleaning
│   ├── calculate_sentiment() - VADER analysis
│   └── StandardScaler() - Feature normalization
│
├── Model Architecture
│   ├── Input layers (3 branches)
│   ├── Embedding layers (title, body)
│   ├── LSTM layers (sequence processing)
│   ├── GlobalMaxPooling1D (feature extraction)
│   ├── Concatenate (merge branches)
│   ├── Dense + Dropout (prediction)
│   └── Output (price change percentage)
│
└── Training & Evaluation
    ├── train_test_split (80/20)
    ├── model.fit() with validation
    ├── Prediction on test set
    └── CSV output with results
```

### Design Patterns Used

- **Pipeline Architecture** - Data flows through collection → preprocessing → modeling
- **Multi-Input Model** - Keras Functional API for complex architectures
- **API Abstraction** - Separate functions for each data source
- **Dynamic Configuration** - Adjustable stocks, subreddits, limits
- **Batch Processing** - Sequential data loading with error handling

---

## What I Learned

This project taught me:

**Financial Machine Learning**
- Stock market prediction is extremely difficult
- Sentiment analysis can capture crowd psychology
- Multiple data sources better than single source
- Overfitting is a major challenge with limited financial data

**API Integration**
- Reddit API (PRAW) for social media scraping
- Yahoo Finance API for real-time stock data
- Rate limit management and error handling
- Data synchronization across different sources

**Natural Language Processing**
- VADER sentiment analysis for social media
- Text preprocessing and tokenization
- Stop word removal and normalization
- Embedding layers for word representation

**Recurrent Neural Networks**
- LSTM architecture for sequential data
- Multi-input model design
- GlobalMaxPooling for feature extraction
- Dropout for regularization

**Time Series Analysis**
- Date-based data alignment
- Percentage change calculation
- Temporal feature engineering
- Market day validation (no weekends/holidays)

**Experimental Design**
- Train/validation splits for time series
- Hyperparameter tuning (epochs, batch size, layers)
- Ablation studies (multi-stock vs. single-stock)
- Performance metrics for regression (MSE)

---

## Future Improvements

If I were to extend this project, I would:

1. **Add Twitter/X Data** - Larger social media footprint, real-time sentiment
2. **News Article Scraping** - Financial news as additional text source
3. **Company Fundamentals** - Quarterly reports, earnings calls, balance sheets
4. **Technical Indicators** - Moving averages, RSI, MACD from price data
5. **Attention Mechanisms** - Transformer architecture for better context
6. **Ensemble Models** - Combine multiple models for robust predictions
7. **Data Caching System** - Save historical data, incremental updates
8. **Real-Time Deployment** - Live predictions before market close
9. **Backtesting Framework** - Simulate trading strategies with predictions
10. **Multi-Day Prediction** - Predict next 3-5 days instead of just one
11. **Explainability** - SHAP/LIME to show which posts drove predictions
12. **Web Dashboard** - Interactive UI for non-technical users

---

## Technical Deep Dive: Why This Architecture Works

### The Multi-Input Advantage

**Traditional Approach:** Concatenate everything into one big feature vector
- Problem: Model can't distinguish between text and numbers
- Problem: Different features need different processing
- Problem: Title and body treated as one blob

**Multi-Input Approach:** Separate pipelines for different data types
- **Text branches** use embeddings + LSTMs (understand language)
- **Numerical branch** uses direct input (already numeric)
- **Late fusion** lets each branch learn independently, then combine

**Example:**
```
Title: "AAPL earnings beat expectations! 🚀"
  → Embedding → LSTM → MaxPool → [0.23, 0.87, -0.12, ...]

Body: "Revenue up 15%, EPS $1.50 vs $1.30 expected..."
  → Embedding → LSTM → MaxPool → [0.65, 0.34, 0.91, ...]

Features: [score=245, ratio=0.92, sentiment=0.87, ...]
  → Direct input → [245, 0.92, 0.87, ...]

Concatenate all three → Dense layers → Prediction: +3.2%
```

### Why LSTMs for Text?

**Traditional RNNs:** Vanishing gradient problem
- Can't remember long-term dependencies
- Struggles with sentences >10 words

**LSTMs (Long Short-Term Memory):**
- **Forget gate** - Decides what to discard from memory
- **Input gate** - Decides what new info to store
- **Output gate** - Decides what to output

**Example:**
```
Title: "Despite strong earnings, AAPL stock drops on guidance concerns"

LSTM processing:
1. "Despite" → Flag: contradiction coming
2. "strong earnings" → Store: positive fundamental
3. "stock drops" → Remember: actual price action
4. "guidance concerns" → Store: negative catalyst
5. Output: Weighted representation emphasizing contradiction
```

Regular RNN would forget "Despite" by the end. LSTM remembers!

### Sentiment Analysis: The Social Signal

**Why Sentiment Matters:**
- Stock prices ≠ company fundamentals alone
- Psychology drives short-term movements
- Herd behavior amplifies trends
- Sentiment precedes action (post → trade)

**Example Scenario:**

**Day 1:** Reddit post "TSLA production issues in Germany" (sentiment: -0.65)  
**Day 1 Market:** TSLA closes -2.3%  
**Model learns:** Negative sentiment → Price drop

**Day 2:** Reddit post "TSLA deliveries exceed expectations!" (sentiment: +0.83)  
**Day 2 Market:** TSLA closes +4.7%  
**Model learns:** Positive sentiment → Price surge

**Prediction for Day 3:** New post "TSLA recall announced" (sentiment: -0.52)  
**Model predicts:** -1.8% based on learned sentiment→price relationship

### GlobalMaxPooling: The Feature Extractor

**Problem:** LSTM outputs sequence of vectors, need single vector for concatenation

**Bad Solution:** Take average (loses important peaks)

**GlobalMaxPooling Solution:** Take maximum value across sequence
- Highlights most important word/feature
- One word can drive entire sentiment
- Reduces dimensionality without losing signal

**Example:**
```
LSTM output sequence for "AAPL earnings CRUSHING expectations!":
[0.23, 0.34, 0.91, 0.45, 0.67, 0.29]
         word: CRUSHING ^^^^ (highest activation)

GlobalMaxPooling → 0.91 (captures "CRUSHING")
```

---

## Illustrative Example

*This is a hypothetical walkthrough of how inputs flow through the model, not a recorded prediction.*

### Scenario: NVIDIA (NVDA) Earnings Week

**Reddit Activity (3 days before earnings):**
- 15 posts on r/stocks discussing AI chip demand
- Average title sentiment: +0.72 (very positive)
- Average body sentiment: +0.58 (positive but cautious)
- High engagement: 200+ upvotes per post, 50+ comments

**Model Input:**
```python
posts = [
    {
        'title': "NVDA crushing it in AI! Buy before earnings?",
        'body': "Data center revenue up 200% YoY, but valuation seems high...",
        'score': 234,
        'ratio': 0.89,
        'num_comments': 67,
        'symbol': 'NVDA'
    },
    # ... 14 more posts
]

# Model processes all 15 posts
predictions = model.predict([titles, bodies, features])
average_prediction = np.mean(predictions)  # +2.8%
```

**How the Model Would Process It:**
1. Title sentiment captured bullish tone
2. Body sentiment showed some caution (tempers prediction)
3. High karma/comments showed consensus (increases confidence)
4. Multiple posts about same topic (amplifies signal)
5. LSTM understood "AI chip demand" context
6. GlobalMaxPooling highlighted key phrases like "crushing it"

---

## Files & Resources

**Project Files:**
- `StockPredictorFinal.py` - Main implementation (600+ lines)
- `COMP4107 Project Report.pdf` - Academic documentation
- `predictions3.csv` - Training set predictions output
- `predictions4.csv` - Validation set predictions output

**API Credentials Required:**
- Reddit API: Client ID, Client Secret, User Agent
- Yahoo Finance: No key required (free API)

**Required Libraries:**

```python
praw>=7.7.0           # Reddit API wrapper
yfinance>=0.2.28      # Yahoo Finance data
nltk>=3.8.0           # NLP and sentiment analysis
tensorflow>=2.14.0    # Neural network framework
keras>=2.14.0         # High-level NN API
pandas>=2.0.0         # Data manipulation
numpy>=1.24.0         # Numerical operations
scikit-learn>=1.3.0   # Preprocessing and evaluation
```

**NLTK Data Packages:**
```python
nltk.download('vader_lexicon')  # Sentiment analysis
nltk.download('stopwords')      # Text preprocessing
nltk.download('punkt')          # Tokenization
```

**How to Run:**
1. Install dependencies: `pip install -r requirements.txt`
2. Set up Reddit API credentials (get from reddit.com/prefs/apps)
3. Replace API keys in script
4. Configure stocks and subreddits in script
5. Run: `python StockPredictorFinal.py`
6. Wait for data collection (5-10 minutes for 100 posts)
7. Model trains and outputs predictions to CSV

---

## Research Foundation

### Prior Work This Builds On

**Systematic Review of ML for Stock Prediction:**
- 69 reviewed papers on stock market prediction
- Key finding: RNNs/LSTMs outperform traditional ML for time series
- Sentiment analysis + financial data = better than either alone
- [Source: ScienceDirect Review](https://www.sciencedirect.com/science/article/pii/S2590291124000615)

**Stock Price Prediction Using LSTM:**
- Netflix 3-year prediction using LSTM on price history alone
- A useful reference design, though its error isn't directly comparable to this project's (different target and data)
- [Source: ProjectPro](https://www.projectpro.io/article/stock-price-prediction-using-machine-learning-project/571)

**VADER Sentiment Analysis:**
- Designed specifically for social media text
- Outperforms general-purpose sentiment analyzers on Twitter/Reddit
- Handles emoji, slang, capitalization, punctuation
- [Source: VADER GitHub](https://github.com/cjhutto/vaderSentiment)

### What This Project Adds

- Multi-input architecture separating titles and bodies
- Date-synchronized fusion of Reddit and financial data
- Dual sentiment analysis (title + body separately)
- Training across multiple stocks rather than one
- Live data collection from multiple APIs

---

## Takeaway

This project demonstrates the power of combining multiple data modalities—structured financial time series and unstructured social media text—to predict market movements. By leveraging modern NLP techniques (VADER sentiment, LSTM sequence processing) and sophisticated neural architectures (multi-input RNNs), the system captures the human psychology behind stock price changes.

Stock prediction is notoriously hard (see the efficient market hypothesis), and this project doesn't show that Reddit sentiment beats the market. What it does show is a working pipeline for joining social media text with market data, and the evaluation lessons above: split by time, compare against a baseline, and watch for leakage when many examples share a target.

The system showcases end-to-end machine learning engineering: API integration, data fusion, feature engineering, model architecture design, training, and evaluation—all in service of tackling one of the most challenging prediction problems in finance.
