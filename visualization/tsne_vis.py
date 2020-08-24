# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 23:03:59 2020

@author: heber
"""
import matplotlib.pyplot as plt



def tsne_vis(model, int_to_vocab, vis_words = 100):
    from sklearn.manifold import TSNE
    embeddings = model.embed.weight.to('cpu').data.numpy()
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embeddings[:vis_words, :])
    
    fig, ax = plt.subplots(figsize=(16,16))
    for idx in range(vis_words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
        



def get_tsne_vis_df(model,int_to_vocab, path = None):
    from sklearn.manifold import TSNE
    import pandas as pd
    import os
    
    embeddings = model.embed.weight.to('cpu').data.numpy()
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embeddings)
    embed_tsne_df = pd.DataFrame(embed_tsne)
    
    embed_tsne_df = embed_tsne_df.reset_index()
    embed_tsne_df.columns = ['id_word','ts1','ts2']
    int_to_vocab_df = pd.DataFrame(int_to_vocab, index = [0]).transpose().reset_index()
    int_to_vocab_df.columns = ['id_word','word']
    embed_tsne_df = embed_tsne_df.merge(
        right = int_to_vocab_df,
        how = 'left',
        on = ['id_word']
        )
    embed_tsne_df = embed_tsne_df[['id_word','word','ts1','ts2']]
    
    if not(Path is None):
        embed_tsne_df.to_csv(os.path.join(os.getcwd() + path))
    return embed_tsne_df
    
    
    
    
    
    
    
    