# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:08:50 2020

@author: HTRUJILLO
"""

# from utilities.data_prep import load_data, preprocess,  create_lookup_tables, subsampling
# from net.architecture import SkipGram
# from net.train import train, save_embbedings, load_embbedings
# from visualization.tsne_vis import tsne_vis

train_model = True
from word2vect.data.data_prep import (
    create_lookup_tables,
    load_data,
    preprocess,
    subsampling,
)
from word2vect.ml.models.architecture import SkipGram
from word2vect.ml.train.train import train

text = load_data("data/text8")
words = preprocess(text)

print("Total words in text: {}".format(len(words)))
print("Unique words: {}".format(len(set(words))))

vocab_to_int, int_to_vocab = create_lookup_tables(words)

int_words = [vocab_to_int[word] for word in words]

print(int_words[:30])

if train_model:
    print("We will train the model")

    train_words = subsampling(int_words, threshold=1e-5)

    print(train_words[:30])

    embedding_dim = 300

    model = SkipGram(len(vocab_to_int), embedding_dim)

    train(
        model,
        train_words,
        int_to_vocab,
        batch_size=128,
        embedding_dim=300,
        print_every=3000,
        epochs=5,
    )

    save_embbedings(
        model,
        n_vocab=len(vocab_to_int),
        n_embed=embedding_dim,
        path="results/skipgram_embed.pth",
    )

else:
    print("We will load a trained model")
    model = load_embbedings(file_path="results/skipgram_embed.pth")


tsne_vis(model, int_to_vocab, vis_words=600)
