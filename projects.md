---
layout: page
title: Projects
permalink: /projects/
description: "Projects by Morgan White spanning multi-agent reinforcement learning, robotics, deep learning for prediction and iOS development."
---

Research, coursework and personal projects. Each one links to a detailed write-up and its source code.

<div class="project-list">
{%- for project in site.data.projects -%}
  {%- include project-card.html project=project highlights=true heading_level=2 -%}
{%- endfor -%}
</div>
