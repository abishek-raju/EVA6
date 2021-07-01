#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:49:24 2021

@author: rampfire
"""


from .data import train_data,test_data
from .models import resnet



import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model', required = True,default=False) 
    parsed = parser.parse_args()
    print(vars(parsed))