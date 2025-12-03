---
layout: page
title: Blog
permalink: /blog/
---

# Blog ✍️

I write about software development, AI, problem-solving, and things I'm learning. Here are my recent posts:

{% comment %}
This Liquid code block iterates through all posts and lists them.
It uses the `site.posts` variable, which Jekyll automatically generates.
{% end comment %}

<ul class="post-list">
  {% for post in site.posts %}
    <li>
      <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
      <h2>
        <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
      </h2>
      {% if post.excerpt %}
        <p>{{ post.excerpt | strip_html | truncatewords: 30 }}</p>
      {% endif %}
    </li>
  {% endfor %}
</ul>
