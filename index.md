---
layout: home
title: Home
description: "Morgan White, AI developer and Carleton University Computer Science Honours graduate (AI and Machine Learning stream), working on applied ML, NLP and multi-agent reinforcement learning."
---

<section class="hero" markdown="1">

# Hi, I'm Morgan White

I'm an **AI developer** with a Computer Science Honours degree from Carleton University (AI and Machine Learning stream, CGPA 10.92/12). At the **Canadian Food Inspection Agency** I built a machine learning tool that classifies 90,000+ food product records, cutting the share of analysts' time spent on data collection from 80% to 20%. My research is in **multi-agent reinforcement learning**: how agents learn to cooperate, and how they cope when things go wrong.

<p class="hero-links">
  <a class="button" href="{{ '/projects/' | relative_url }}">See my projects</a>
  <a class="button button-secondary" href="{{ '/assets/morgan-white-resume.pdf' | relative_url }}">Resume (PDF)</a>
</p>

</section>

## Featured projects

<div class="project-grid">
{%- for project in site.data.projects -%}
  {%- if project.featured -%}
    {%- include project-card.html project=project -%}
  {%- endif -%}
{%- endfor -%}
</div>

[All projects →](/projects/)

## Experience

**AI Developer**, Canadian Food Inspection Agency · *May 2024 – April 2025*
: Built an NLP classification tool (TF-IDF, lemmatization, scikit-learn) for 90,000+ food product records, with over 80% precision, recall and F1. Trained the team to use it; offered two contract extensions.

**Teaching Assistant**, Carleton University · *Winter 2023, Fall 2024*
: Programming Paradigms (Haskell, Prolog) and Introduction to AI.

**Programmer Analyst Co-op**, Elections Canada · *June – December 2022*
: Moved hard-coded settings in the EREG application into database-driven configuration (C#, SQL).

[More about me →](/about/)

## Skills

**Languages:** Python, Java, C#, C++, Swift, SQL, Haskell, Prolog  
**Machine learning:** PyTorch, TensorFlow/Keras, scikit-learn, Pandas, NumPy, NLTK  
**Reinforcement learning:** Unity ML-Agents, TorchRL, VMAS, PPO, SAC, POCA, MAPPO
{: .skills}

## Get in touch

I'm looking for roles in **AI/ML engineering**, **research and development**, and **robotics and autonomous systems**. I'm based in Hamilton, Ontario, and open to relocating.

[Email](mailto:morgan13white@icloud.com) · [LinkedIn](https://www.linkedin.com/in/morgan-white-95b245237) · [GitHub](https://github.com/morganwhite13) · [Resume (PDF)](/assets/morgan-white-resume.pdf)
