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
11. *Denoising Distantly Supervised Open-Domain Question Answering*
	- Characteristics:
		- 文中描述了DS在QA上的应用方式，数据集。以及对于远程监督过程的降噪处理。
		- 其检索与问句中实体有关的wikipedia的paragraph，并假设其中的信息能够回答该问题。这是远程监督的思路。在这些文段中，抽取一些较可信文段，并在文字中select出一部分作为问题答案。其中包含对DS语料的降噪和select answer的方式，值得参考。
	- ACL 2018
12. **Learning When to Trust Distant Supervision: An application to low-resource POS tagging using cross-lingual projection**
	- Characteristics:
		- 本文清晰地介绍了远程监督运用在POS任务上的过程。其利用双语对应语料，和其中一种语言的POS训练语料，生成另一种POS的训练语料。这体现了远程监督的思维方式。
		- 在模型的实际工作中，为了降噪，加入了标签的转移矩阵，这是一种常见的对远程监督数据降噪的一种手法。
		- 在实验设置上，其首先证明了远程监督降噪的必要，又证明了远程监督+降噪>极少量手工标注数据的道理，论证了其研究的意义
			- Gold Standard：These languages are obviously not low-resource languages, however we can use this data to simulate the low-resource setting by only using a small 1,000 tokens of the gold annotations for training.
			- There are an average of 1.85 million parallel sentences for each of the eight language pairs.
	- ACL 2016
13. Multi-Level Structureed Self-Attention for Distantly Supervised Relation Extraction
	- Characteristics;
		- 本论文讲常用的1-D attention vector， 替换为2-D attention matirx，得到了不错的效果。
	- ACL 2018
14. **Neural Relation Extraction via
 Inner-Sentence Noise Reduction and Transfer Learning**
 	- Characteristics:
 		- 本文的点主要在两部分，第一是很新奇的使用了依存树来对远程监督的句子进行处理，只关注句子中实体所在的子树，去除句子的其他所有成分，达到了降噪的效果。同时，为了近一步优化模型，模型使用实体标注task作预训练，之后将参数迁移至RE模型中
 		- 疑问：取子树环节是只针对训练过程的吗？还是也同时对Test数据有效？
 	- ACL 2018
 15. **Distant supervision for relation extraction without labeled data**
 	- 本文是DS的开山之作，提出DS的论文。
 	- Supervised：In supervised approaches, sentences in a cor- pus are first hand-labeled for the presence of en- tities and the relations between them. The NIST Automatic Content Extraction (ACE) RDC 2003 and 2004 corpora, for example, include over 1,000 documents in which pairs of entities have been la- beled with 5 to 7 major relation types and 23 to 24 subrelations, totaling 16,771 relation instances. ACE systems then extract a wide variety of lexi- cal, syntactic, and semantic features, and use su- pervised classifiers to label the relation mention holding between a given pair of entities in a test set sentence, optionally combining relation mentions.
 	- Unsupervised: An alternative approach, purely unsupervised information extraction, extracts strings of words between entities in large amounts of text, and clusters and simplifies these word strings to pro- duce relation-strings (Shinyama and Sekine, 2006; Banko et al., 2007). Unsupervised approaches can use very large amounts of data and extract very large numbers of relations, but the resulting rela- tions may not be easy to map to relations needed for a particular knowledge base.





封面照片摄于罗马🇮🇹
{:.success}