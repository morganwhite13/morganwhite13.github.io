---
layout: post
title: "Predicting YouTube Views with Deep Learning: What I Learned"
date: 2023-12-15 10:00:00 -0500
categories: machine-learning nlp deep-learning
---

Can you predict how many views a YouTube video will get before it's even published? I spent a semester building a neural network to find out. Here's what I discovered.

## The Question That Started It All

As someone who occasionally makes YouTube videos, I've always been fascinated by why some videos explode while others flop—even when they seem similar. Two creators upload videos on the same topic, with comparable production quality, but one gets 10,000 views and the other gets 10 million. Why?

I decided to build a machine learning model to answer this question.

## The Data Challenge

First problem: getting the data. YouTube's API has strict rate limits, so I couldn't just download millions of videos. Instead, I built several collection strategies:

**Random Sampling:** Find random videos, then collect more from their channels  
**Trending Analysis:** Study what's currently popular  
**Targeted Channels:** Examine specific successful creators  

Each approach gave me different insights. Random sampling showed the full spectrum of YouTube, from niche content to viral hits. Trending analysis revealed what works *right now*. Targeted channels let me see patterns within consistent styles.

The result? About 400 videos per dataset—not huge, but enough to learn from.

## Building Up Complexity

I didn't jump straight to a complex model. Instead, I built six versions, each adding more features:

### Version 1: Title Only
Just the video title. Can you predict views from "I Survived 100 Days in Minecraft" alone?

Surprisingly effective! The model learned patterns:
- Numbered formats ("Top 10...")
- Comparisons ("$1 vs $1000...")
- Time challenges ("24 Hour...")
- Questions ("What Happens If...?")

### Version 2: + Subscriber Count
Added the channel's subscriber count. Huge improvement! A video from a 10M subscriber channel performs very differently than one from a 10K channel.

### Version 3: + Publication Date
Older videos have more time to accumulate views. Adding "days since publication" helped the model understand temporal dynamics.

### Version 4: + Description
Video descriptions affect searchability and click-through rates. The model learned that detailed, keyword-rich descriptions correlate with higher views.

### Version 5: + Channel Name
Brand recognition matters. "MrBeast" in the channel name carries weight. The model picked up on this.

### Version 6: + Category
Entertainment videos behave differently than gaming videos or how-to tutorials. Category became the final piece.

## The Neural Network Architecture

The final model uses a multi-input architecture:
Text Inputs → Embedding Layers → Bidirectional LSTM → Pooling
Numerical Inputs → Scaling → Dense Layers
All Branches → Concatenate → Dense Layers → Prediction

**Why Bidirectional LSTM?**  
I needed to capture context in both directions. In "I Survived 100 Days in Minecraft," the words relate to each other forward and backward. LSTM cells remember long-term dependencies, and bidirectional processing doubles this power.

**Why Multiple Embeddings?**  
Titles and descriptions use words differently. A 300-dimensional embedding space lets the model learn these differences independently before combining them.

**Why GlobalMaxPooling?**  
After processing sequences, I need to extract the *most important* features. Max pooling grabs the strongest signals from each sequence.

## The Preprocessing Pipeline

Raw text needs cleaning. My pipeline:

1. **Lowercase everything** - "AMAZING" and "amazing" should be treated the same
2. **Remove punctuation** - "video!" becomes "video"
3. **Tokenize** - Split into words using NLTK
4. **Remove stopwords** - Filter "the", "is", "and" etc.
5. **Convert to sequences** - Words → numbers the model can process
6. **Pad sequences** - Make all inputs the same length

Numerical features (subscribers, days) get scaled using StandardScaler so they don't dominate training.

## What Did It Learn?

The model revealed some fascinating patterns:

**Title Formulas Work:**
- "$1 vs $1000" format consistently predicted high views
- "I Survived X Days" pattern strong across categories
- Question formats engage curiosity

**Channel Size Matters, But Not Linearly:**
- 10x more subscribers ≠ 10x more views
- Diminishing returns after certain thresholds
- Small channels CAN go viral with the right title

**Category Creates Baselines:**
- Gaming videos have different "normal" view counts than tutorials
- Entertainment broad, Education narrow but dedicated
- Music videos unpredictable

**Time Accumulates Views Non-Linearly:**
- Initial spike, then long tail
- Some videos gain views steadily over years
- Recent uploads need time to reach potential

## The Numbers Game

Mean Squared Error ended up in the trillions. Sounds terrible, right?

But context matters. When you're predicting values from 1,000 to 100,000,000, and MSE squares the differences, the numbers explode. A prediction that's off by 2 million views on a 10 million view video contributes 4 trillion to the MSE alone.

What mattered more: **relative accuracy**. Could the model tell which of two videos would perform better? Yes. Could it predict exact view counts? Not reliably—too many factors I couldn't measure (thumbnail quality, actual video content, luck).

## Technical Challenges

**API Rate Limits:**  
YouTube's API quotas meant I couldn't gather massive datasets. Solution: intelligent caching and multiple collection strategies.

**Text Complexity:**  
Titles and descriptions vary wildly in length. Solution: dynamic max length detection and padding.

**Feature Imbalance:**  
Text inputs have thousands of dimensions; subscriber count is one number. Solution: separate processing branches that converge later.

**Overfitting Risk:**  
Small dataset + complex model = danger. Solution: dropout layers (randomly disable 30% of neurons during training).

## Lessons Learned

### About Machine Learning
- **Start simple, add complexity** - My incremental approach revealed which features mattered
- **MSE isn't everything** - For real-world tasks, qualitative validation matters
- **Feature engineering > model complexity** - Good inputs beat fancy architecture

### About YouTube
- **Patterns exist but aren't deterministic** - You can optimize, but virality has randomness
- **Consistency builds expectations** - Channels with clear brands perform predictably
- **Metadata matters more than you think** - Titles, descriptions, tags affect discoverability

### About Software Engineering
- **Modular design enables experimentation** - Six model versions in one codebase
- **Caching saves time and money** - API rate limits are real constraints
- **Documentation while building** - Future me appreciated past me's comments

## What's Next?

If I continue this project, I'd add:

**Thumbnail Analysis:** Computer vision on images—colors, faces, text overlays matter hugely

**Video Duration:** 5-minute videos vs 30-minute videos have different view patterns

**More Data:** 10,000+ videos would reduce overfitting significantly

**Real-Time Interface:** Web app where creators test titles before uploading

**Time-Series Prediction:** Predict view trajectory over time, not just final count

## The Real Insight

The most valuable outcome wasn't the model's accuracy—it was understanding the **why** behind view counts. YouTube success isn't random, but it's also not purely algorithmic. It's a mix of:

- **Optimization** (titles, descriptions, metadata)
- **Authority** (subscriber base, brand recognition)
- **Timing** (publication date, trending topics)
- **Quality** (which my model couldn't measure)
- **Luck** (going viral has unpredictable elements)

Building this model taught me that data science isn't just about predictions—it's about gaining insights into complex systems.

---

**Interested in the details?** Check out the [full project documentation](/projects/youtube-predictor/) or explore the [code on GitHub](https://github.com/yourusername/youtube-view-predictor).

**Tools Used:** Python, TensorFlow/Keras, YouTube Data API, NLTK, Pandas, Scikit-learn  
**Course:** COMP 3106 - Introduction to Artificial Intelligence (Fall 2024)
