#!/usr/bin/env python

#__________________________________________________
# pyLorenz/utils/auxiliary/
# configparser.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/8
#__________________________________________________
#
# custom ConfigParser class
# copied from toolbox
#

from os.path     import abspath
from tree        import Tree
from stringutils import stringToInt, stringToFloat, stringToStringList, stringToNumpyArray, removeComments, removeEndLine, isBlanck, isImport, extractImport 
from stringutils import isTreeNode, extractTreeNode, extractTreeLeaf, hasReference, extractReferences, cutDepthsAndOptions

#__________________________________________________

class ConfigParser(object):

    #_________________________

    def __init__(self, t_commentChar = '#', t_referenceChar = '$'):
        # tree
        self.m_tree          = Tree()
        # comment char
        self.m_commentChar   = t_commentChar
        # reference char
        self.m_referenceChar = t_referenceChar

    #_________________________

    def clone(self):
        # clone config
        config        = ConfigParser(self.m_commentChar, self.m_referenceChar)
        config.m_tree = self.m_tree.clone()
        return config

    #_________________________

    def options(self, *t_options):
        # list of options
        return self.m_tree.children(list(t_options))

    #_________________________

    def removeOption(self, *t_options):
        # remove option
        self.m_tree.removeChild(list(t_options))

    #_________________________

    def tofile(self, t_fileName):
        # write
        f = open(t_fileName, 'w')
        self.write(f)
        f.close()

    #_________________________

    def write(self, t_file):
        # write
        def writeNodeLine(t_depth, t_node, t_file):
            line  = ''
            for i in range(4*t_depth):
                line += ' '
            line += '[' + t_node + ']\n'
            t_file.write(line)
        def writeLeafLine(t_depth, t_node, t_value, t_file):
            line  = ''
            for i in range(4*t_depth):
                line += ' '
            line += t_node + ' = ' + t_value + '\n'
            t_file.write(line)
        self.m_tree.write(t_file, writeNodeLine, writeLeafLine)

    #_________________________

    def get(self, *t_options):
        # get
        return self.m_tree.get(list(t_options))

    #_________________________

    def getInt(self, *t_options):
        # get and cast to int
        return stringToInt(self.get(*t_options))

    #_________________________

    def getFloat(self, *t_options):
        # get and cast to float
        return stringToFloat(self.get(*t_options))

    #_________________________

    def getStringList(self, *t_options):
        # get and cast to string list
        return stringToStringList(self.get(*t_options))

    #_________________________

    def getNumpyArray(self, *t_options):
        # get and cast to numpy array
        return stringToNumpyArray(self.get(*t_options))

    #_________________________

    def set(self, *t_options):
        # set
        options = list(t_options)
        value   = options.pop()
        self.m_tree.set(options, value)

    #_________________________

    def readline(self, t_line, t_depths, t_options, t_imports, t_fileHierarchy):
        # read line
        t_line = removeComments(removeEndLine(t_line), self.m_commentChar)

        # blanck
        if isBlanck(t_line):
            return

        # import
        if isImport(t_line):
            (depth, toImport) = extractImport(t_line)
            cutDepthsAndOptions(t_depths, t_options, depth, t_line)
            t_imports.append((t_fileHierarchy, list(t_options), toImport))

        # node
        elif isTreeNode(t_line):
            (depth, node) = extractTreeNode(t_line)
            cutDepthsAndOptions(t_depths, t_options, depth, t_line)
            t_depths.append(depth)
            t_options.append(node)

        # leaf
        else:
            (depth, leaf, value) = extractTreeLeaf(t_line)
            cutDepthsAndOptions(t_depths, t_options, depth, t_line)
            if not len(t_depths) == len(t_options):
                raise IndentationError(t_line)
            t_depths.append(depth)
            options = list(t_options)
            options.append(leaf)
            self.m_tree.set(options, value)

    #_________________________

    def readlines(self, t_file, t_fileHierarchy):
        # read file
        lines   = t_file.readlines()
        depths  = []
        options = []
        imports = []
        for line in lines:
            self.readline(line, depths, options, imports, t_fileHierarchy)
        return imports

    #_________________________

    def readfile(self, t_fileName, t_fileHierarchy):
        # read file
        t_fileName = abspath(t_fileName)
        if t_fileName in t_fileHierarchy:
            raise RuntimeError('Trying to loop imports')
        t_fileHierarchy.append(t_fileName)

        f = open(t_fileName, 'r')
        i = self.readlines(f, t_fileHierarchy)
        f.close()
        return i

    #_________________________

    def solveReferenceInString(self, t_string, t_maxRecursion):
        #
        if not hasReference(t_string, self.m_referenceChar):
            return t_string
        if t_maxRecursion == 0:
            raise RuntimeError('Can not solve references')

        references = extractReferences(t_string, self.m_referenceChar)
        for reference in references:
            options  = reference[1:-1].split('.')
            value    = self.solveReferenceInTree(options, t_maxRecursion-1)
            t_string = t_string.replace(reference, value)

        return t_string

    #_________________________

    def solveReferenceInTree(self, t_options, t_maxRecursion):
        #
        value = self.m_tree.get(list(t_options))
        value = self.solveReferenceInString(value, t_maxRecursion)
        self.m_tree.set(list(t_options), value)
        return value

    #_________________________

    def readfiles(self, t_fileNames):

        if isinstance(t_fileNames, str):
            t_fileNames = [t_fileNames]

        # read all files in fileNames
        imports = []
        for fileName in t_fileNames:
            imports.extend(self.readfile(fileName, []))

        # solve imports
        while len(imports) > 0:
            (hierarchy, options, toImport) = imports.pop(0)
            # file to import
            maxRecursion = self.m_tree.size() - 1
            toImport     = abspath(self.solveReferenceInString(toImport, maxRecursion))
            # import file
            subConfig    = ConfigParser(self.m_commentChar, self.m_referenceChar)
            imports.extend(subConfig.readfile(toImport, hierarchy))
            # merge config
            self.m_tree.merge(options, subConfig.m_tree)

        # solve references
        allOptions   = self.m_tree.subChildren()
        maxRecursion = len(allOptions) - 1
        for options in allOptions:
            self.solveReferenceInTree(options, maxRecursion)
            maxRecursion -= 1

#__________________________________________________

