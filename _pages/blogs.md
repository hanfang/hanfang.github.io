---
layout: archive
title: "Blog Posts"
permalink: /blogs/
author_profile: true
---

Welcome to my blog! Here I share thoughts on AI research, machine learning, and technology trends.

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