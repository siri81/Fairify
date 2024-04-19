#!/usr/bin/env python3
# -- coding: utf-8 --

# reinterpret network symbolically using z3 variables.
import sys
from z3 import *
import numpy as np 
import pandas as pd 
import collections
import time
import datetime

from utils.verif_utils import *


def layer_net(x, w, b):
    layers = []    
    for i in range(len(w)):
        x1 = w[i].T @ x + b[i]
        y1 = x1 if i == len(w)-1 else relu(x1)
        layers.append(y1)
        x = y1
    return layers

def net(x, w, b):
    x1 = w[0].T @ x + b[0]
    y1 = relu(x1)
    x2 = w[1].T @ y1 + b[1]
    # y = 1 / (1 + math.exp(-x3)) # WP computer for sigmoid
    return x2

def z3_net(x, w, b):
    
    fl_x = np.array([FP('fl_x%s' % i, Float32()) for i in range(20)])  
    
    for i in range(len(x)):
        fl_x[i] = ToReal(x[i])
    
        
    x1 = w[0].T @ fl_x + b[0]
    y1 = z3Relu(x1)
    x2 = w[1].T @ y1 + b[1]
    # y = 1 / (1 + math.exp(-x3)) # WP computer for sigmoid
    return x2

def ground_net(x):
    layer_outs = []
    for i in range(len(w)):
        layer = []
        for j in range(len(w[i][0])):
            sum = 0
            for k in range(len(x)):
                sum += x[k] * w[i][k][j]
            sum += b[i][j]
            layer.append(sum)
        layer = np.asarray(layer, dtype=np.float64)
        y = layer if i == len(w)-1 else relu(layer)
        layer_outs.append(y)
        x = y
    return y