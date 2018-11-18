---
layout: article
title: 关系抽取、远程监督综述
article_header:
  type: overlay
  theme: dark
  background_image:
    src: /assets/images/my_pic/Roma_3.JPG
comment: true
mathjax: true
key: 20181110
tags:
- NLP
---

持续更新的综述。

<!--more-->

## 论文列表

1. **Going out on limb: Jointly Extraction of Entity Mentions and Relations without Dependency Tree.**
	- 本文提出关于联合抽取Entity Mentions和Relations的方法。介绍依靠Dependency Tree的方法，和不依靠其的方法，并介绍了本Task的数据集。
	- ACL 2017
2. **DSGAN: Generative Adversarial Training for Distant Supervision Relation Extraction.**
	- 本文提出了一种提高RE Task的思路：即通过提高Distant Supervision的Training DATA的质量，间接地提高RE的效果。GAN的思路允许了模型完全的无监督，不需要额外的标注数据，来提高模型的效果。文中包含了GAN训练中的小tricks，还有Policy Gradient Descent方法。
	- ACL 2018
3. **Knowledge as A Bridge: Imporving Cross-domain Answer Selection with External Knowledge.**
	- 本文意在借用外部知识（Knowledge Base）来提高Answer Selection的效果。有意义的点在于：如何利用外部知识的思路上，和使用到的迁移学习（Transfer Learning）上。
	- COLING 2018. International Conference on Computational Linguistics.
4. **An interpretable Generative Adversarial Approach to Classification of Latent Entity Relations in Unstructured Sentences.**
	- 本文的点在于：
		- 作者提出了一种可解释的关系抽取方法。作者通过分析人类理解句子的方式，让神经网络来模拟整个过程--选出最重要、最相关的词，的方法，来给予神经网络以可解释性。
		- Adversarial Reinforcement Learning：同样采用类似GAN的思路，通过引入多个网络来引入对抗性，以实现无监督学习的目的。
		- Semi-Supervision：作者提出了一种可人工介入的机制，从而提出了一个新的观点。少量的人工介入，可以显著的提升模型的效果，那么该方法是有价值的。
	- AAAI 2018
5. *Exploratory Neural Relation Classification for Domain Knowledge Acquisition.*
	- COLING 2018
6. *Learning with Noise: Enhance Distantly Supervision Relation Extraction with Dynamic Transition Matrix*
	- Characteristics：
		- 提出了Bag level的Embedding。
		- Transition Matrix。
		- 课程学习（Curriculum Learning）
		- 给DS样例的噪声等级，分级。DS Noise Level.
		- 实验的设计方法。验证自己观点时，设计实验的方法。
	- ACL 2017
7. **End-to-End Task-Oriented Dialogue System with Distantly Supervision Knowledge Base Retriever.**
	- Characteristics：
		- Knowledge Base Retriever.
		- 作者提出了一种思路去在一个Task上通过一定规则生成DS训练样例，通过引入DS的方法解决问题，提高模型的效果。Distantly Supervised.
	- CCL 2018
8. *Dose William Shakespeare REALLY Write Hamlet? Knowledge Representation Learning with Confidence.*
	- AAAI 2018
9. *Translating Embedding for Modeling Multi-relation Data.*
	- 大名鼎鼎的TranE.
10. **Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Network.**
	- 大名鼎鼎的PCNN + Multi-Instances Learning.


封面照片摄于罗马🇮🇹
{:.success}