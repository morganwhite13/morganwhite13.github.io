---
layout: page
title: Blog
permalink: /blog/
---

# Blog Posts 📝

I write about software development, AI, problem-solving, and things I'm learning. Here are my recent posts:

<ul class="post-list">
  {% for post in site.posts %}
    <li>
      <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
      <h3>
        <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
      </h3>
      {% if site.show_excerpts %}
        {{ post.excerpt }}
      {% endif %}
    </li>
  {% endfor %}
</ul>
