#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# cast.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/26
#__________________________________________________
#
# cast
#

import numpy as np

#__________________________________________________

def makeList(t_x):
    # return x if x is already a list
    # else return [x]
    if isinstance(t_x, list):
        return t_x
    else:
        return [t_x]

#__________________________________________________

def makeNumpyArray(t_x):
    # transform x to numpy array
    if isinstance(t_x, np.ndarray):
        return t_x
    else:
        return np.array(makeList(t_x))

#__________________________________________________

def configDictToConfigList(t_dict):
    l = []
    for truthParameters in t_dict:
        for filterParameters in t_dict[truthParameters]:
            l.append(t_dict[truthParameters][filterParameters])
    return l

#__________________________________________________

