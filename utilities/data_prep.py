# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:13:10 2020

@author: HTRUJILLO
"""
import re
from collections import Counter

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