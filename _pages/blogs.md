---
layout: archive
title: "Tokens for Thoughts"
permalink: /blogs/
author_profile: true
---

Welcome to my blog! Here I share thoughts on AI research, machine learning, and technology trends.

## Featured

<article>
  <h2><a href="/posts/2026/02/central-dogma-of-artificial-intelligence/">The Central Dogma of Artificial Intelligence</a></h2>
  <p><span class="archive__date">📅</span> <strong>Published:</strong> <time>February 7, 2026</time></p>
  <p>Every mature science has its central dogma. Biology has DNA → RNA → Protein. What is ours? Intelligence is the compression of experience into generalization. This essay explores how compression theory explains pre-training, post-training, and alignment — and why understanding compression losses may be the key frontier in AI.</p>
</article>

<article>
  <h2><a href="https://tokens-for-thoughts.notion.site/the-rl-environment-field-guide" target="_blank">The RL Environment Field Guide</a></h2>
  <p><span class="archive__date">📅</span> <strong>Published:</strong> <time>January 26, 2026</time></p>
  <p>A practical guide to RL environments using Pokemon Red as a case study. Covers the agent-environment loop, observation and action spaces, reward design, trajectories, graders, and the credit assignment problem—with real code examples and lessons from training an LLM-based gameplay agent.</p>
</article>

<article>
  <h2><a href="https://tokens-for-thoughts.notion.site/post-training-101" target="_blank">Post-training 101: A Hitchhiker's Guide to LLM Post-training</a></h2>
  <p><span class="archive__date">📅</span> <strong>Published:</strong> <time>September 12, 2025</time></p>
  <p>A comprehensive guide to post-training techniques for large language models, covering supervised fine-tuning, RLHF, reward models, and practical implementation details. This guide walks through the entire journey from pre-training to instruct-tuned models with hands-on examples and best practices.</p>
</article>

---

{% assign years = site.posts | group_by_exp: "post", "post.date | date: '%Y'" %}

{% if years.size > 1 %}
  {% comment %} Only show year headers if there are multiple years {% endcomment %}
  {% capture written_label %}'None'{% endcapture %}
  {% for post in site.posts %}
    {% capture year_label %}{{ post.date | date: "%Y" }}{% endcapture %}
    {% if year_label != written_label %}
      <h2 id="{{ year_label | slugify }}" class="archive__subtitle">{{ year_label }}</h2>
      {% capture written_label %}{{ year_label }}{% endcapture %}
    {% endif %}
    {% include archive-single.html %}
  {% endfor %}
{% else %}
  {% comment %} Just list posts without year headers if only one year {% endcomment %}
  {% for post in site.posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endif %}