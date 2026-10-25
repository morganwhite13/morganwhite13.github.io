---
title: YouTube Views Prediction Using Neural Networks
permalink: /projects/youtube-predictor/
description: "A multi-input TensorFlow model with bidirectional LSTMs that predicts YouTube view counts from a video's title, description, channel, category, subscriber count and age."
---

## Quick Summary

A deep learning system that predicts how many views a YouTube video will get from its title, description, channel name, category, subscriber count and age. It collects its own data through the YouTube Data API, preprocesses the text with NLTK, and builds up a multi-input neural network in six stages, adding one feature at a time, with a bidirectional LSTM for each text field.

**Tech Stack:** Python, TensorFlow/Keras, YouTube Data API, NLTK, scikit-learn  
**Status:** ✅ Complete (course project, COMP 3106 Introduction to AI)  
**GitHub:** [View Source Code](https://github.com/morganwhite13/YouTube-Views-Prediction-Using-Neural-Networks)

---

## The Challenge

View counts depend on far more than anything visible in a video's metadata: the recommendation algorithm, timing, thumbnails, and luck. The goal was to see how much a neural network could learn from the information that *is* available:

- Text: title, description and channel name
- Category
- Channel size (subscriber count)
- Video age (days since publication)

Two things make this hard. View counts span several orders of magnitude, from a thousand to hundreds of millions, and every training example has to be collected through a rate-limited API.

---

## Data Collection Pipeline

The project uses the YouTube Data API v3 with several sampling strategies, so the dataset isn't dominated by one kind of channel:

- `getRandomVideos()` and `getRandomChannelsVideos()`: random search results, and a batch of videos from each of several random channels
- `getPopularVideos()` and `getPopularChannelsVideos()`: trending videos and videos from popular channels
- `getCategoryVideos()` and `getChannelsVideos()`: category-specific and channel-specific sampling
- `combineDatasets()`: merges every cached dataset, keeping each video ID once

**Data per video:** title, description, channel name, subscriber count (via `get_channel_subscriber_count()`), category ID, publish date, and view count (the target).

**Filters:** English only (`relevanceLanguage='en'`, plus a check of each video's default language) and more than 1,000 views.

**Caching:** every collection function saves its results to a JSON file and reloads that file on later runs. This keeps experiments within the API's daily quota and makes training runs repeatable.

---

## Text Preprocessing

Each text field goes through the same NLTK pipeline:

```python
def preprocessText(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stopWords2 = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
    tokens = [token for token in tokens if token not in stopWords2]
    return ' '.join(tokens)
```

Then:
- A separate Keras `Tokenizer` per field (title, description, channel name, category) turns words into integer IDs
- Sequences are padded to the longest example in the dataset
- Subscriber count and days since publication are standardized with `StandardScaler`

---

## Model: Six Stages, One Feature at a Time

Rather than starting with everything, I built the model up in stages. Each stage is its own function in `projectFinal.py`, so the effect of each new input can be compared:

| Stage | Function | Inputs |
| :--- | :--- | :--- |
| 1 | `neuralTitleModel` | Title |
| 2 | `neuralTitleSubscriberModel` | + subscriber count |
| 3 | `neuralTSDateModel` | + days since publication |
| 4 | `neuralTSDDescriptionModel` | + description |
| 5 | `neuralTSDDChannelModel` | + channel name |
| 6 | `neuralAllModel` | + category |

### Final Architecture (`neuralAllModel`)

```
Title ─────────► Embedding(300) ─► BiLSTM(128) ─► GlobalMaxPool ─┐
Description ───► Embedding(300) ─► BiLSTM(128) ─► GlobalMaxPool ─┤
Channel name ──► Embedding(300) ─► BiLSTM(128) ─► GlobalMaxPool ─┤
Category ──────► Embedding(300) ─► BiLSTM(128) ─► GlobalMaxPool ─┼─► Concatenate
Subscribers (scaled) ────────────────────────────────────────────┤      │
Days since publication (scaled) ─────────────────────────────────┘      ▼
                                              Dense(512, ReLU) ─► Dropout(0.3)
                                              Dense(256, ReLU) ─► Dropout(0.3)
                                              Dense(1, linear) ─► predicted views
```

Built with the Keras Functional API, so each input gets its own branch before the branches are merged.

```python
title_lstm = Bidirectional(LSTM(units=128, return_sequences=True))(title_embedding)
title_pooling = GlobalMaxPooling1D()(title_lstm)
# ...same for description, channel name and category...
concat = Concatenate()([title_pooling, description_pooling, channel_title_pooling,
                        category_pooling, input_subscriber, input_days_since_publication])
dense1 = Dense(512, activation='relu')(concat)
dropout1 = Dropout(0.3)(dense1)
dense2 = Dense(256, activation='relu')(dropout1)
dropout2 = Dropout(0.3)(dense2)
output = Dense(1, activation='linear')(dropout2)
```

- **Bidirectional LSTMs** read each text field in both directions.
- **Global max pooling** keeps the strongest signal from each position, so a single important word (like "survived" or "$1,000") can drive the output.
- **Dropout** after each dense layer is the only regularization.

---

## Training and Evaluation

- 80/20 train/test split (`train_test_split`, `random_state=42`)
- Adam optimizer, mean squared error loss, 10 epochs, batch size 32
- The script prints the test-set MSE

### What-If Predictions

After training, `neuralAllModel` predicts views for a few made-up videos, then changes one input at a time to see how the prediction moves:

- "I Survived 100 Days in Canada" (1M subscribers), "$1 VS $1,000 Water" (500K subscribers) and "Worlds Craziest Invention" (100K subscribers)
- Then, for one video: a different description, a different title ("$1 VS $100,000 Invention"), a different channel name, 90× the subscribers, a different category, and an older publish date

This is a quick sensitivity check: it shows which inputs the trained model actually responds to.

### An Honest Look at the Results

The test MSE isn't recorded in the repo, and a few design choices limit how far the numbers can be trusted:

- **Raw view counts as the target.** Training on raw counts with MSE means a handful of viral videos dominate the loss. Predicting `log(views)` would let the model learn proportional differences instead.
- **No baseline.** Without a simple comparison model (for example, one that predicts from subscriber count alone), there's no way to tell how much the text adds.
- **A very large model for the data.** Each of the four embedding tables is sized for 100,000 words × 300 dimensions, about 120 million parameters in total, trained on a few hundred videos. Smaller vocabularies or pretrained embeddings would suit this much data far better.
- **The test set doubles as validation data** during training. Since there's no early stopping it doesn't change the weights, but a separate validation split would be cleaner.

---

## Challenges & Solutions

### Limited API Quota

**Problem:** The YouTube Data API allows 10,000 quota units per day, and search calls are expensive.

**Solution:** Every collection function caches its results to JSON and reuses them on later runs, so the model can be retrained many times on the same data without new API calls.

### Text Sequence Length

**Problem:** Titles are short, descriptions can be very long, and channel names are a word or two.

**Solution:** A separate tokenizer and padding length for each field, so short fields aren't swamped with padding sized for descriptions.

### Keeping Experiments Comparable

**Problem:** With many possible inputs, it's hard to tell which ones help.

**Solution:** Six model functions that each add one input, on the same data and split.

---

## Code Architecture

```
projectFinal.py (about 1,400 lines)
├── Data collection
│   ├── getRandomVideos(), getPopularVideos(), getCategoryVideos()
│   ├── getChannelsVideos(), getPopularChannelsVideos(), getRandomChannelsVideos()
│   ├── getAllVideos(), combineDatasets()
│   └── get_channel_subscriber_count()
├── Preprocessing
│   └── preprocessText()
└── Models (each one trains, evaluates and runs example predictions)
    ├── neuralTitleModel()
    ├── neuralTitleSubscriberModel()
    ├── neuralTSDateModel()
    ├── neuralTSDDescriptionModel()
    ├── neuralTSDDChannelModel()
    └── neuralAllModel()
```

---

## What I Learned

**Deep learning**
- Multi-input models with the Keras Functional API
- Embedding layers, bidirectional LSTMs and pooling for text
- Building a model up feature by feature to see what each input contributes

**NLP**
- Tokenization, stop-word removal and padding with NLTK and Keras

**Working with a real API**
- Sampling strategies to avoid a biased dataset
- Caching to stay within quota and keep runs repeatable

**Evaluation**
- Why a heavy-tailed target like view counts needs a log transform
- Why a baseline matters as much as the model
- How easy it is to build a model far larger than the data can support

---

## Future Improvements

1. **Predict `log(views)`** and report error in that space
2. **Add baselines**: subscriber count alone, and a linear model on the same features
3. **Use pretrained word embeddings** (such as GloVe) with a much smaller vocabulary
4. **Collect more data** and keep a separate validation split for early stopping
5. **Try a Transformer encoder or a pretrained language model** for the text fields
6. **Add thumbnails** with a CNN, since they strongly influence clicks

---

## Files & Resources

**Project files:**
- `projectFinal.py`: data collection, preprocessing and all six models
- `*.json`: cached datasets, created on first run

**To run it:**
1. `pip install -r requirements.txt`
2. Replace `INSERTAPIKEY` in `projectFinal.py` with a YouTube Data API v3 key
3. `python projectFinal.py` (it downloads the NLTK data it needs on first run, then runs `neuralAllModel`; uncomment another stage at the bottom of the file to run it instead)

---

## Takeaway

This project covers the full machine learning loop on real, messy data: collecting it through a rate-limited API, cleaning the text, and building a multi-input neural network one feature at a time. Its biggest lesson was about evaluation. With a heavy-tailed target, a small dataset and no baseline, the honest next step isn't a bigger model but a log-scale target, a simple baseline to beat, and a model sized to the data.
