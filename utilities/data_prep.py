# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:13:10 2020

@author: HTRUJILLO
"""

def load_data(file):
    
    with open(file) as f:
        text = f.read()
        
    print("-----file example-----")
    print(text[:100])
    return text