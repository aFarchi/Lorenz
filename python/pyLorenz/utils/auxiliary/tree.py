#!/usr/bin/env python

#__________________________________________________
# pyLorenz/utils/auxiliary/
# tree.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/8
#__________________________________________________
#
# custom tree class
#

#__________________________________________________

class Tree(object):

    #_________________________

    def __init__(self):
        # children
        self.m_children = {}

    #_________________________

    def size(self):
        # number of (sub-)children
        size = 0
        for node in self.m_children:
            if isinstance(self.m_children[node], Tree):
                size += self.m_children[node].size()
            else:
                size += 1
        return size

    #_________________________

    def clone(self):
        # clone tree
        clone = {}
        for node in self.m_children:
            if isinstance(self.m_children[node], Tree):
                clone[node] = self.m_children[node].clone()
            else:
                clone[node] = self.m_children[node]

        cloneTree            = Tree()
        cloneTree.m_children = clone
        return cloneTree

    #_________________________

    def children(self, t_childList):
        # children list
        if len(t_childList) == 0:
            return self.m_children.keys()
        child = t_childList.pop(0)
        return self.m_children[child].children(t_childList)

    #_________________________

    def subChildren(self, t_currentChildList = []):
        # list of all (sub-)children
        subChildren = []
        for child in self.m_children:
            if isinstance(self.m_children[child], Tree):
                subChildren.extend(self.m_children[child].subChildren(t_currentChildList+[child]))
            else:
                subChildren.append(t_currentChildList+[child])
        return subChildren

    #_________________________

    def removeChild(self, t_childList):
        # remove child
        if len(t_childList) == 0:
            return
        child = t_childList.pop(0)
        try:
            if len(t_childList) == 0:
                del self.m_children[child]
            else:
                self.m_children[child].removeChild(t_childList)
        except:
            pass

    #_________________________

    def write(self, t_file, t_writeNodeLine, t_writeLeafLine, t_depth = 0):
        # write to file
        for node in self.m_children:
            if isinstance(self.m_children[node], Tree):
                t_writeNodeLine(t_depth, node, t_file)
                self.m_children[node].write(t_file, t_writeNodeLine, t_writeLeafLine, t_depth+1)
                t_file.write('\n')
            else:
                t_writeLeafLine(t_depth, node, self.m_children[node], t_file)

    #_________________________

    def get(self, t_childList):
        # get function
        if len(t_childList) == 0:
            return
        child = t_childList.pop(0)
        if len(t_childList) == 0:
            return self.m_children[child]
        else:
            return self.m_children[child].get(t_childList)
        
    #_________________________

    def set(self, t_childList, t_value):
        # set function
        child = t_childList.pop(0)
        if len(t_childList) == 0:
            self.m_children[child] = t_value
        else:
            if not child in self.m_children:
                self.m_children[child] = Tree()
            self.m_children[child].set(t_childList, t_value)

    #_________________________

    def merge(self, t_childList, t_otherTree):
        # merge other tree
        if len(t_childList) > 0:
            child = t_childList.pop(0)
            if not child in self.m_children:
                self.m_children[child] = Tree()
            self.m_children[child].merge(t_childList, t_otherTree)
        else:
            for otherChild in t_otherTree.m_children:
                if isinstance(t_otherTree.m_children[otherChild], Tree) and otherChild in self.m_children and isinstance(self.m_children[otherChild], Tree):
                    self.m_children[otherChild].merge([], t_otherTree.m_children[otherChild])
                else:
                    self.m_children[otherChild] = t_otherTree.m_children[otherChild]

#__________________________________________________

