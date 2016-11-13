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

from collections import OrderedDict
from path        import Path

from utils.auxiliary.stringutils import removeComments, removeEndLine, isBlanck, isImport, hasReference, isTreeNode
from utils.auxiliary.stringutils import extractImport, extractReferences, extractTreeNode, extractTreeLeaf, writeTreeNodeLine, writeTreeLeafLine

#__________________________________________________

NoDefault = object()

#__________________________________________________

class LoopImportError(Exception):
    pass

#__________________________________________________

class CrossReferenceError(Exception):
    pass

#__________________________________________________

class Tree(object):

    #_________________________

    def __init__(self):
        # children
        self.m_children = OrderedDict()

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
        clone = OrderedDict()
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
        if not t_childList:
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
        if not t_childList:
            return
        child = t_childList.pop(0)
        try:
            if not t_childList:
                del self.m_children[child]
            else:
                self.m_children[child].removeChild(t_childList)
        except KeyError:
            pass

    #_________________________

    def tofile(self, t_fileName):
        # write to file
        with open(Path(t_fileName), 'w') as f:
            self.write(f)

    #_________________________

    def write(self, t_file, t_depth = 0):
        # write to file
        for node in self.m_children:
            if isinstance(self.m_children[node], Tree):
                writeTreeNodeLine(t_file, t_depth, node)
                self.m_children[node].write(t_file, t_depth+1)
                t_file.write('\n')
            else:
                writeTreeLeafLine(t_file, t_depth, node, self.m_children[node])

    #_________________________

    def get(self, t_childList, t_default = NoDefault):
        # get function
        if not t_childList:
            return t_default
        child = t_childList.pop(0)
        if child in self.m_children:
            if t_childList:
                return self.m_children[child].get(t_childList, t_default)
            else:
                return self.m_children[child]
        else:
            if t_default is NoDefault:
                raise KeyError
            else:
                return t_default

    #_________________________

    def set(self, t_childList, t_value):
        # set function
        child = t_childList.pop(0)
        if not t_childList:
            self.m_children[child] = t_value
        else:
            if not child in self.m_children:
                self.m_children[child] = Tree()
            self.m_children[child].set(t_childList, t_value)

    #_________________________

    def merge(self, t_childList, t_otherTree):
        # merge other tree
        if t_childList:
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

    #_________________________

    def readline(self, t_line, t_depths, t_childList, t_imports, t_fileHierarchy, t_commentChar = '#'):
        # read line
        t_line = removeComments(removeEndLine(t_line), t_commentChar)

        # blanck
        if isBlanck(t_line):
            return

        # import
        if isImport(t_line):
            extractImport(t_depths, t_childList, t_line, t_imports, t_fileHierarchy)

        # node
        elif isTreeNode(t_line):
            extractTreeNode(t_depths, t_childList, t_line)

        # leaf
        else:
            (leaf, value) = extractTreeLeaf(t_depths, t_childList, t_line)
            self.set(t_childList+[leaf], value)

    #_________________________

    def readlines(self, t_lines, t_fileHierarchy, t_commentChar = '#'):
        # read lines
        depths    = []
        childList = []
        imports   = []
        for line in t_lines:
            self.readline(line, depths, childList, imports, t_fileHierarchy, t_commentChar)
        return imports

    #_________________________

    def readfile(self, t_fileName, t_fileHierarchy, t_commentChar = '#'):
        # read file
        if t_fileName.abspath() in t_fileHierarchy:
            raise LoopImportError
        t_fileHierarchy.append(t_fileName.abspath())

        lines = t_fileName.lines(retain = False)
        return self.readlines(lines, t_fileHierarchy, t_commentChar)

    #_________________________

    def solveReferenceInString(self, t_string, t_maxRecursion, t_referenceChar = '$'):
        # solve cross references in string
        if not hasReference(t_string, t_referenceChar):
            return t_string
        if t_maxRecursion == 0:
            raise CrossReferenceError 

        references = extractReferences(t_string, t_referenceChar)
        for reference in references:
            childList = reference[1:-1].split('.')
            value     = self.solveReference(childList, t_maxRecursion-1, t_referenceChar)
            t_string  = t_string.replace(reference, value) 

        return t_string

    #_________________________

    def solveReference(self, t_childList, t_maxRecursion, t_referenceChar = '$'):
        # solve cross references for string contained in tree[options]
        value = self.get(list(t_childList))
        value = self.solveReferenceInString(value, t_maxRecursion, t_referenceChar)
        self.set(list(t_childList), value)
        return value

    #_________________________

    def readfiles(self, t_fileNames, t_commentChar = '#', t_referenceChar = '$'):
        # read files

        if isinstance(t_fileNames, str):
            t_fileNames = [t_fileNames]

        # read all files in fileNames
        imports = []
        for fileName in t_fileNames:
            imports.extend(self.readfile(Path(fileName), [], t_commentChar))

        # solve imports
        while imports:
            (hierarchy, childList, toImport) = imports.pop(0)
            # file to import
            maxRecursion = self.size() - 1
            toImport     = Path(self.solveReferenceInString(toImport, maxRecursion, t_referenceChar))
            # import file
            subTree      = Tree()
            imports.extend(subTree.readfile(toImport, hierarchy, t_commentChar))
            # merge config
            self.merge(childList, subTree)

        # solve references
        allChildren  = self.subChildren()
        maxRecursion = len(allChildren) - 1
        for childList in allChildren:
            self.solveReference(childList, maxRecursion)
            maxRecursion -= 1

#__________________________________________________

