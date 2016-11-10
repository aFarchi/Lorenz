#!/usr/bin/env python

#__________________________________________________
# pyLorenz/utils/auxiliary/
# stringutils.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/8
#__________________________________________________
#
# functions related to string operations
#

import numpy as np

#__________________________________________________

def stringToInt(t_string):
    return int(eval(t_string))

#__________________________________________________

def stringToFloat(t_string):
    return float(eval(t_string))

#__________________________________________________

def stringToStringList(t_string):
    if t_string == '':
        return []
    if t_string == '[' or t_string == ']':
        raise ValueError('Cannot convert string to string list')
    if t_string[0] == '[':
        t_string = t_string[1:]
    if t_string[-1] == ']':
        t_string = t_string[:-1]
    l = t_string.split(',')
    return [e.strip() for e in l]

#__________________________________________________

def stringToNumpyArray(t_string):
    a = eval(t_string)
    if isinstance(a, np.ndarray):
        return a
    elif isinstance(a, list):
        return np.array(a)
    else:
        return np.array([a])

#__________________________________________________

def removeComments(t_string, t_commentChar = '#'):
    return t_string.split(t_commentChar)[0]

#__________________________________________________

def removeEndLine(t_string):
    return t_string.replace('\n', '')

#__________________________________________________

def isBlanck(t_string):
    return len(t_string) == 0 or t_string.isspace()

#__________________________________________________

def isImport(t_string):
    return t_string.strip().startswith('import ')

#__________________________________________________

def extractImport(t_string):
    toImport = t_string.strip().replace('import ', '')
    depth    = t_string.find('import ')
    return (depth, toImport)

#__________________________________________________

def isTreeNode(t_string):
    node = t_string.strip()
    return len(node) > 2 and node[0] == '[' and node[-1] == ']'

#__________________________________________________

def extractTreeNode(t_string):
    node  = t_string.strip()[1:-1]
    depth = t_string.find(node) - 1
    return (depth, node)

#__________________________________________________

def extractTreeLeaf(t_string):
    if '=' in t_string:
        l     = t_string.split('=')
        leaf  = l[0].strip()
        value = l[1].strip()
    else:
        leaf  = t_string.strip()
        value = ''

    depth = t_string.find(leaf)
    return (depth, leaf, value)

#__________________________________________________

def hasReference(t_string, t_referenceChar = '$'):
    return len(t_string) > 2 and t_referenceChar in t_string

#__________________________________________________

def extractReferences(t_string, t_referenceChar = '$'):
    refs = []
    index = -1
    for i in range(len(t_string)):
        if t_string[i] == t_referenceChar:
            if index == -1:
                index = i
            else:
                refs.append(t_string[index:i+1])
                index = -1
    return refs

#__________________________________________________

def checkIndentationError(t_depths, t_depth, t_errorMessage):
    if ( ( len(t_depths) > 0 and t_depth <= t_depths[-1] and not t_depth in t_depths ) or
            ( len(t_depths) == 0 and t_depth > 0 ) ):
        raise IndentationError(t_errorMessage)

#__________________________________________________

def cutDepthsAndOptions(t_depths, t_options, t_depth, t_errorMessage):
    checkIndentationError(t_depths, t_depth, t_errorMessage)
    if t_depth in t_depths:
        index = t_depths.index(t_depth)
        while len(t_depths) > index:
            t_depths.pop()
        while len(t_options) > index:
            t_options.pop()

#__________________________________________________

