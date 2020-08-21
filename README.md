# Implementing Word2Vec SkipGram

## Visual Demo

![Word_embed_plot](https://user-images.githubusercontent.com/28582065/90884890-e3928100-e3b0-11ea-8cf4-220416dc91c9.gif)

## Introduction

This Python package uses [PyTorch](https://pytorch.org/) to implement the Word2Vec algorithm using skip-gram architecture.

We provide the following resources that were used to build this package. We suggest reading these either beforehand or while you're exploring the code.

1. [Word2Vec paper](https://arxiv.org/pdf/1301.3781.pdf) from Mikolov et al.
2. [Neural Information Processing Systems, paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) with improvements for Word2Vec also from Mikolov et al.

## Word2Vec

The Word2Vec algorithm finds much more efficient representations by finding vectors that represent the words. These vectors also contain semantic information about the words. This way, words that show up in similar **contexts**, such as code, programming or python will have vectors representation near from each other.

In this implementation, we'll be using the skip-gram architecture because it performs better than Continuous Bag-Of-Words. Here, we pass in a word and try to predict the words surrounding it in the text. In this way, we can train the network to learn representations for words that show up in similar contexts.

Hopefully, the following diagram will help to settle down the intuition:

![skip_gram](https://user-images.githubusercontent.com/28582065/90183557-ffe05d80-ddb3-11ea-81bf-530d9b27bf13.PNG)

## Data

We have used a series of Wikipedia articles provided by Matt Mahoney, you can find a broader description by clicking [here](http://mattmahoney.net/dc/textdata.html).


## Model

Below is an approximate diagram of the general structure of the network:


<img width="832" alt="skip_gram_arch" src="https://user-images.githubusercontent.com/28582065/90184551-93fef480-ddb5-11ea-8ab4-1dde6e9285eb.png">


## Results

In this section, we will show some preliminary results. But before, lest talk a bit about how can we take advantage of the embeddings.

### Cosine Similarity

We can encode a given word as vectors $\vec{a}$ using the embedding table, then calculate the similarity with each word vector $\vec{b}$ in the embedding table with the following equation:

$$
\mathrm{similarity} = \cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}||\vec{b}|}
$$

### Random Examples

The image below shows some randomly selected words, followed by a set of words with which they share a similar context:

![Random_results](https://user-images.githubusercontent.com/28582065/90186260-2f916480-ddb8-11ea-8243-c2f441665bd4.PNG)
