# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:13:10 2020

@author: HTRUJILLO
"""
import re
from collections import Counter
import random
import numpy as np


def load_data(file):
    '''
    Read a text document containing the corpus which is the embedding iinput

    Parameters
    ----------
    file : .txt o plain text file
        The file containing the corpus

    Returns
    -------
    text : str
        A long string containing the corpus 

    '''
    
    with open(file) as f:
        text = f.read()
        
    print("-----file example-----")
    print(text[:100])
    return text


def preprocess(text):
    '''
    A function that tokenize the corpus creating especial tokens for punctuation signs.
    Returns only words appearing more than 5 times

    Parameters
    ----------
    text : str
        A long string containing the corpus 

    Returns
    -------
    trimmed_words : list
        List containing words in the document as elements.

    '''
    
    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    
    # Remove all words with  5 or fewer occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]
    print(trimmed_words[:30])
    return trimmed_words



def create_lookup_tables(words):
    '''
    Creates two dictionaries to convert words to integers and back again (integers to words).

    Parameters
    ----------
    words : list
        List containing words in the document as elements.


    Returns
    -------
    vocab_to_int : dict
        words as keys and int id as valie.
    int_to_vocab : dict
        int id as keys and words as valie.

    '''
    
    word_counts = Counter(words)
    
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int ={word: ii for ii, word in int_to_vocab.items()}
    
    return vocab_to_int, int_to_vocab




def subsampling(int_words, threshold = 1e-5):
    '''
    Implements the subsampling process to remove some of the noise coming from very frequent words,
    proposed by Mikolov. For each word  𝑤𝑖  in the training set, we'll discard it with some probability.

    Parameters
    ----------
    int_words : list
        A list of words containing as each element the tokens from a corpus
    threshold : np.float, optional
        Hyperparameter to calculate the probability. The default is 1e-5.

    Returns
    -------
    train_words : list
        Sampled list of words to use in the training process.

    '''
    
    total_count = len(int_words)
    word_counts = Counter(int_words)


    freqs = {word: count/total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts} 
    train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

    return train_words


