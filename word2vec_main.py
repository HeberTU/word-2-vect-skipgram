# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:08:50 2020

@author: HTRUJILLO
"""
import os 
os.chdir("C:/Users/htrujillo/projects/Implementing_Word2Vec_SkipGram")

from utilities.data_prep import load_data, preprocess


text = load_data("data/text8")

words = preprocess(text)
