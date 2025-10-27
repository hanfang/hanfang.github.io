---
title: "Generalized Parallel Scaling with Interdependent Generations"
collection: publications
category: conferences
permalink: /publication/2025-10-02-generalized-parallel-scaling
excerpt: 'This paper proposes Bridge, a method that creates interdependent responses in parallel by treating batched hidden states as integrated tensors rather than separate slices.'
date: 2025-10-02
venue: 'NeurIPS 2025'
paperurl: 'https://arxiv.org/abs/2510.01143'
citation: 'Harry Dong, David Brandfonbrener, Eryk Helenowski, Yun He, Mrinal Kumar, Han Fang, Yuejie Chi, Karthik Abinav Sankararaman. (2025). &quot;Generalized Parallel Scaling with Interdependent Generations.&quot; <i>NeurIPS 2025</i>.'
---

This paper addresses a limitation in parallel LLM inference where multiple responses are typically generated independently. The authors propose Bridge, a method that creates interdependent responses in parallel by treating batched hidden states as integrated tensors rather than separate slices. With minimal added parameters (2.8%-5.1%), Bridge achieves up to 50% relative mean accuracy gains from reinforcement learning with verifiable rewards and improves consistency across correct responses. The approach generalizes across different generation widths and works with any post-generation aggregation technique.
