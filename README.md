# Implementing Word2Vec SkipGram

## Introduction

This Python package uses [PyTorch](https://pytorch.org/) to implement the Word2Vec algorithm using skip-gram architecture.

We provide the following resources that were used to build this package. We suggest reading these either beforehand or while you're exploring the code.

1. [Word2Vec paper](https://arxiv.org/pdf/1301.3781.pdf) from Mikolov et al.
2. [Neural Information Processing Systems, paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) with improvements for Word2Vec also from Mikolov et al.

## Word2Vec

The Word2Vec algorithm finds much more efficient representations by finding vectors that represent the words. These vectors also contain semantic information about the words.
