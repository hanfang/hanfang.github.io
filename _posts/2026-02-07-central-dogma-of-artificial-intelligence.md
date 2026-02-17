---
title: 'The Central Dogma of Artificial Intelligence'
date: 2026-02-07
permalink: /posts/2026/02/central-dogma-of-artificial-intelligence/
excerpt: 'Every mature science has its central dogma. Biology has DNA → RNA → Protein. What is ours? Intelligence is the compression of experience into generalization.'
tags:
  - artificial intelligence
  - machine learning
  - deep learning
  - alignment
  - essay
citation: 'Fang, H. (2026). "The Central Dogma of Artificial Intelligence." <i>Han Fang</i>. Available at: https://hanfang.info/posts/2026/02/central-dogma-of-artificial-intelligence/'
---

*Part of the [Tokens for Thoughts](https://tokens-for-thoughts.notion.site) series*

*Every mature science has its central dogma — a foundational claim so deeply embedded that practitioners forget it's even there. Biology has DNA → RNA → Protein. Thermodynamics has entropy. What is ours?*

---

**Intelligence is the compression of experience into generalization.**

Not a definition of intelligence — those are abundant and mostly useless. This is a claim about the *mechanism*. The arrow that connects raw experience to capable behavior. And like Crick's original formulation in molecular biology, it turns out to be both more productive and more subtle than it first appears.

The idea that compression and intelligence are deeply linked has a long pedigree — from Shannon's information theory through Solomonoff's formalization of induction to Hutter's mathematical proof that optimal decision-making reduces to optimal compression. For decades, this was beautiful theory with limited empirical traction. Then came large language models, and suddenly the theory had teeth. But we've mostly treated this as a useful intuition rather than what I think it actually is — a *dogma*, in the scientific sense. Something foundational enough to build on, and constraining enough to be worth taking seriously.

## Data → Weights → Behavior

In biology, the central dogma derives its force from three concrete artifacts — DNA, RNA, Protein — and a claim about the directional flow of information between them. Sequential information flows from nucleic acid to protein, never the reverse.

The AI equivalents:

**Data** exists on disk — terabytes of text, images, interaction logs. Like DNA, it's the raw blueprint, the store of everything the system could potentially learn.

**Weights** exist in the model — billions of parameters shaped by optimization. Like RNA, they're the intermediate representation: a compressed encoding of the data's statistical structure, transcribed into a form that can be acted upon.

**Behavior** is the observable output — the predictions, the generated text, the actions taken. Like protein, it's where the information finally becomes functional. It's what the system *does*.

And the directionality claim holds: **information flows from data into weights into behavior, and is lost at each step.** You cannot fully recover the training data from the weights. You cannot recover the weights from the behavior. Compression is irreversible, and the losses propagate forward.

This describes what most modern AI systems are doing, regardless of architecture or objective. Supervised learning compresses labeled examples into decision boundaries. Pre-training compresses internet-scale text into a geometry of token predictions. RL compresses trajectories of reward into policies. Diffusion models compress the structure of natural images into learned score functions. Different data, different compression, different behavior. Same flow.

The empirical evidence bears this out. Kaplan et al. (2020) and Hoffmann et al. (2022) showed that model performance follows remarkably smooth power laws as you increase compute and data — trends spanning seven orders of magnitude. What are scaling laws, really? They're a quantitative measurement of how compression quality improves with resources. The smoothness of these curves is itself strong evidence that compression is a productive lens for understanding what these systems are doing. Hutter even put money on the idea — his [Hutter Prize](http://prize.hutter1.net/) offers €500,000 for better lossless compression of Wikipedia, on the premise that compressing human knowledge more efficiently means building a smarter model of the world.

## The Arrow Only Points One Way

But the real force of a central dogma is never in what it says *can* happen. It's in what it says *cannot*.

Crick's deepest insight wasn't that DNA makes RNA makes protein — it was that the arrow is irreversible. Once sequential information has passed into protein, it cannot flow back to nucleic acid. That's what made it a dogma rather than a diagram.

The AI equivalent: **what's lost during compression cannot be recovered downstream.**

Take a concrete example. You pre-train a large language model, and it performs well on English reasoning benchmarks but poorly on Yoruba. The natural instinct is to fix this in post-training — fine-tune on more Yoruba data, adjust the objective, iterate. And it helps, but only up to a point. The ceiling is low, and no amount of fine-tuning will reach it. Why? Because the pre-training compression was shaped overwhelmingly by English text. The statistical structure of Yoruba — its morphology, its reasoning patterns, the way knowledge is organized in that language — was never deeply compressed into the weights to begin with. Fine-tuning can surface and reshape what was compressed; it can't reconstruct what wasn't.

Or consider a model that was pre-trained on text alone, with no exposure to structured logical notation or formal proofs. You can post-train it on chain-of-thought data and it will *look* like it's reasoning — producing plausible intermediate steps — but it will hit a ceiling on problems that require genuine formal manipulation. The structure of formal reasoning wasn't in the compression, so it isn't in the weights, so it can't be in the behavior. The losses propagate forward.

This is a constraint practitioners encounter constantly, even if they don't frame it this way. Every time a post-training team discovers that a capability "just isn't there" despite extensive tuning, they're running into the directionality of the arrow. Every time a model exhibits a stubborn blind spot that no amount of RLHF can fix, that's compression loss surfacing as a behavioral gap. The ceiling of any system is set during compression, not during deployment. You can reshape what's there; you can't create what isn't.

The arrow of information points one way.

## Post-Training as Recompression

If pre-training is *compression*, post-training is *recompression*. Pre-training produces a dense, undifferentiated representation of everything the model has seen — the useful and the toxic, the profound and the trivial, all folded into the same set of weights. Post-training selectively reshapes this compressed geometry under new objectives — helpfulness, safety, instruction-following, reasoning — amplifying some dimensions and suppressing others.

This clarifies what post-training can and cannot do. Recompression is not addition. You're not injecting new knowledge into the model (or at least, that's not where the leverage is). You're reshaping the *topology* of what's already there — changing which compressed representations are accessible, in what contexts, and with what probability.

This is why post-training is so much more art than science. The pre-trained model is a high-dimensional manifold of compressed knowledge, and post-training applies forces to that manifold. RLHF applies force through human preference signals. DPO applies force through contrastive pairs. Each method bends the manifold differently, and the effects interact in ways that are difficult to predict from first principles.

Anyone who has done this at scale learns the same lesson: **the hardest problem in post-training isn't making the model better at what you want. It's preserving what you don't want to lose.**

Every recompression has collateral effects. Push the model toward helpfulness and you might compress away the caution that makes it safe. Push toward safety and you might suppress the capability that makes it useful. Push toward reasoning and you might lose the fluency that makes it pleasant to interact with. The geometry of the weight space doesn't respect our taxonomies — the dimensions we care about aren't orthogonal, and pulling on one inevitably tugs on others.

The best post-training pipelines don't just optimize for a target — they manage a delicate equilibrium between competing compression objectives. Recent work on mixture-of-judges approaches (Xu et al., 2024) makes this explicit: rather than compressing all of human preference into a single reward signal, you decompose it into multiple specialized judges — safety, helpfulness, reasoning, factuality — each compressing a different dimension of what we care about, and then blend their signals during optimization. This is why techniques like PPO with KL constraints work: the KL penalty is literally a regularizer that prevents the recompression from straying too far from the original compressed representation. It's a way of saying "reshape the manifold, but not too much."

This also explains why post-training recipes are so hard to transfer between base models. Each pre-training run produces a different compressed geometry, and the recompression forces that work well on one geometry can fail on another. The field's habit of treating post-training as a recipe — "apply RLHF with these hyperparameters" — misses the fact that what you're really doing is applying forces to a manifold whose shape you don't fully understand.

## Deeper Than It Looks

Reverse transcriptase didn't destroy Crick's dogma; it revealed that the flow of genetic information was more intricate than anyone had imagined, and opened entire new fields in the process.

The same is happening in AI. The simple version — compress once during training, decode at inference — is already breaking down in important ways. Reasoning models perform additional compression at test time. Multi-agent systems distribute compression across workflows. But the extension I think matters most is the one the field is only beginning to take seriously: **compression that never stops.**

The static pipeline assumes a clean separation between training and deployment. You compress, you ship, you serve. But the most capable systems we're building now blur that boundary entirely. Agents that learn from their environment in real time. Models that update on user feedback. RL systems that continuously improve their policies through interaction. These aren't exceptions to the dogma — they're a deeper expression of it, where the output of one compression cycle becomes the input of the next.

And it changes what we're optimizing for. In the static pipeline, the question is: *how well did we compress?* In the continuous version, the question becomes: *how well does compression improve over time?* The goal shifts from building the best frozen model to building the best *learning process* — a system where every interaction makes the next compression a little better, a flywheel where intelligence compounds.

I think this may be the most consequential direction in the field right now. Not better architectures for static compression, but better frameworks for continuous compression — systems that don't just generalize from past data but that improve their ability to generalize from ongoing experience.

## The Alignment Bottleneck

If post-training is recompression, then alignment is a *compression objective*. We are trying to compress human values — messy, contextual, often contradictory — into the geometry of a neural network's weights. And many of the problems we encounter in alignment are, at root, problems of compression.

Consider Goodhart's law — the observation that optimizing for a proxy measure eventually diverges from the true objective. Through the compression lens, this is a statement about *lossy compression of values*. When we train a reward model on human preferences, we're compressing the full complexity of what humans care about into a scalar signal. Some of that complexity survives the compression. Some doesn't. And when we then optimize a policy against that compressed signal, the policy finds and exploits exactly the structure that was lost.

This isn't abstract. Every practitioner who has trained a reward model has seen it: the model that scores highly on helpfulness while being subtly sycophantic, because the reward model compressed "helpful" and "agreeable" into the same region of its output space. The model that aces safety benchmarks while becoming brittle, because the reward model couldn't distinguish between "cautious" and "useless." These aren't random failures. They're *compression artifacts* — systematic consequences of the information that was lost when human values were squeezed through a bottleneck.

And the directionality claim applies with full force. Once values have been compressed into a reward model, the nuances that were dropped cannot be recovered by the policy training. No amount of PPO can reconstruct a distinction the reward model failed to capture. This is why reward model quality matters so much — it's not just another component in the pipeline; it's the compression bottleneck for human values.

None of this solves alignment. But it does clarify *why* alignment is hard in a way that vague appeals to "value complexity" do not. Alignment is hard because compressing human values is hard, and because the losses in that compression propagate forward — silently and irreversibly, until they surface as failures in behavior.

## Why "Dogma"?

I chose that word deliberately. Not "law," not "principle."

In science, a dogma is something a field believes so deeply it becomes invisible — part of the background assumptions rather than the active research program. Dogmas are productive because they let you stop questioning the foundations and get to work. But they also constrain what questions you think to ask.

For the past decade, AI has been living in the surface layer of this idea, optimizing the compression pipeline — better architectures, better data, more compute — and the results have been extraordinary. Scaling laws, foundation models, the entire LLM revolution: all of it is Data → Weights → Behavior working as advertised.

But the next decade belongs to researchers who ask a different kind of question. Not "how do we compress more?" but "what are we losing in the compression, and does it matter?" The field has gotten very good at the forward pass of the dogma. The open frontier is understanding the losses — in capability, in values, in the structure of knowledge itself — and learning to compress what actually matters.

That's not an incremental improvement to the pipeline. But whether it's the right shift — and what we'd need to make it work — is an open question worth taking seriously.

---

### References

- Crick, F. H. C. (1958). On protein synthesis. *Symp. Soc. Exp. Biol.*, 12, 138–163.
- Crick, F. (1970). Central dogma of molecular biology. *Nature*, 227, 561–563.
- Delétang, G. et al. (2024). Language modeling is compression. *ICLR 2024*. [arXiv:2309.10668]
- Hoffmann, J. et al. (2022). Training compute-optimal large language models. *NeurIPS* 35. [arXiv:2203.15556]
- Hutter, M. (2005). *Universal Artificial Intelligence*. Springer.
- Kaplan, J. et al. (2020). Scaling laws for neural language models. [arXiv:2001.08361]
- Kolmogorov, A. N. (1965). Three approaches to the quantitative definition of information. *Prob. Inf. Transmission*, 1(1), 1–7.
- Rae, J. (2023). Compression for AGI. *Stanford MLSys Seminar*.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27, 379–423.
- Solomonoff, R. J. (1964). A formal theory of inductive inference. *Information and Control*, 7, 1–22 and 224–254.
- Sutskever, I. (2023). An observation on generalization. *Talk at Simons Institute Workshop on Large Language Models and Transformers*.
- Xu, T. et al. (2024). The perfect blend: Redefining RLHF with mixture of judges. [arXiv:2409.20370]
