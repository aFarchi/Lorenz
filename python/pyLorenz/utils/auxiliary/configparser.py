#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/auxiliary/
# configparser.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/5
#__________________________________________________
#
# custom config parser class
#

import numpy as np

from dictutils import recDictGet
from dictutils import recDictSet
from dictutils import recMakeKeyList
from dictutils import dictWrite

#__________________________________________________

class ConfigParser(object):

    #_________________________

    def __init__(self, t_fileNames = [], t_commentChar = '#'):
        # dictionnary containing options
        self.m_options     = {}
        # comment char
        self.m_commentChar = t_commentChar
        # read fileNames
        self.m_success     = []
        for fileName in t_fileNames:
            if self.readFile(fileName):
                self.m_success.append(fileName)
        # solve references
        self.solveReferences()

    #_________________________

    def tofile(self, t_fileName):
        # write config to file
        f = open(t_fileName, 'w')

        def writeKeyLine(t_indentationLevel, t_key, t_file):
            line  = ''
            for i in range(4*t_indentationLevel):
                line += ' '
            line += '[' + t_key + ']\n'
            t_file.write(line)

        def writeValueLine(t_indentationLevel, t_key, t_value, t_file):
            line  = ''
            for i in range(4*t_indentationLevel):
                line += ' '
            line += t_key + ' = ' + t_value + '\n'
            t_file.write(line)

        dictWrite(self.m_options, f, writeKeyLine, writeValueLine)
        f.close()

    #_________________________

    def options(self, *t_args):
        # get option list
        return recDictGet(self.m_options, list(t_args)).keys()

    #_________________________

    def set(self, *t_args):
        # set value
        keys = list(t_args)
        val  = keys.pop()
        recDictSet(self.m_options, keys, val)

    #_________________________

    def get(self, *t_args):
        # get value
        keys = list(t_args)
        return recDictGet(self.m_options, keys)

    #_________________________

    def getString(self, *t_keys):
        # get value
        return self.get(*t_keys)

    #_________________________

    def getInt(self, *t_keys):
        # get value and cast to int
        return int(eval(self.get(*t_keys)))

    #_________________________

    def getFloat(self, *t_keys):
        # get value and cast to float
        return float(eval(self.get(*t_keys)))

    #_________________________

    def getStringList(self, *t_keys):
        # get value and cast to string list
        s = self.get(*t_keys)
        if s == '' or s == '[' or s == ']':
            return []
        if s[0] == '[':
            s = s[1:]
        if s[-1] == ']':
            s = s[:-1]
        l = s.split(',')
        return [e.strip() for e in l]

    #_________________________

    def getNPArray(self, *t_keys):
        # get value and cast to string numpy array
        a = eval(self.get(*t_keys))
        if isinstance(a, np.ndarray):
            return a
        elif isinstance(a, list):
            return np.array(a)
        else:
            return np.array([a])

    #_________________________

    def removeCommentsAndEndLine(self, t_line):
        # remove comments in line
        return t_line.split(self.m_commentChar)[0].replace('\n', '')

    #_________________________

    def isBlanckLine(self, t_line):
        # return true if line does not contain any information
        return len(t_line) == 0 or t_line[0] in ['#', '/'] or t_line.isspace()

    #_________________________

    def isKeyLine(self, t_line):
        # return true if line is a "key" line
        return '[' in t_line and ']' in t_line and not '=' in t_line

    #_________________________

    def extractKey(self, t_line):
        # extract key from key line
        key              = t_line.strip()[1:-1]
        indentationLevel = t_line.find(key) - 1
        return (indentationLevel, key)

    #_________________________

    def extractValue(self, t_line):
        # extract value from value line
        if '=' in t_line:
            l      = t_line.split('=')
            option = l[0].strip()
            value  = l[1].strip()
        else:
            option = t_line.strip()
            value  = ''

        indentationLevel = t_line.find(option)
        return (indentationLevel, option, value)

    #_________________________

    def readLine(self, t_line, t_indentationLevels, t_currentKeys):
        # read line
        t_line = self.removeCommentsAndEndLine(t_line)
        if self.isBlanckLine(t_line):
            return
        if self.isKeyLine(t_line):
            (il, key) = self.extractKey(t_line)
            if ( ( len(t_indentationLevels) > 0 and il <= t_indentationLevels[-1] and not il in t_indentationLevels ) or
                    ( len(t_indentationLevels) == 0 and il > 0 ) ):
                raise IndentationError
            if il in t_indentationLevels:
                i = t_indentationLevels.index(il)
                while len(t_indentationLevels) > i:
                    t_indentationLevels.pop()
                while len(t_currentKeys) > i:
                    t_currentKeys.pop()
            t_indentationLevels.append(il)
            t_currentKeys.append(key)
        else:
            (il, opt, val) = self.extractValue(t_line)
            if ( ( len(t_indentationLevels) > 0 and il <= t_indentationLevels[-1] and not il in t_indentationLevels ) or
                    ( len(t_indentationLevels) == 0 and il > 0 ) ):
                raise IndentationError
            if il in t_indentationLevels:
                i = t_indentationLevels.index(il)
                while len(t_indentationLevels) > i:
                    t_indentationLevels.pop()
                while len(t_currentKeys) > i:
                    t_currentKeys.pop()
            if not len(t_indentationLevels) == len(t_currentKeys):
                raise IndentationError

            t_indentationLevels.append(il)
            keys = list(t_currentKeys)
            keys.append(opt)
            recDictSet(self.m_options, keys, val)

    #_________________________

    def readFile(self, t_fileName):
        # read config file
        f     = open(t_fileName, 'r')
        lines = f.readlines()
        f.close()
        level = []
        keys  = []
        try:
            for line in lines:
                self.readLine(line, level, keys)
            return True
        except:
            return False

    #_________________________

    def recSolveReference(self, t_keys, t_maxDepth):
        # find the correct value for option[keys] with references solved
        value = recDictGet(self.m_options, list(t_keys))
        if len(value) < 2 or not '$' in value:
            return value
        if t_maxDepth == 0:
            raise RuntimeError('Can not solve references...')

        references = []
        iStart     = -1
        for i in range(len(value)):
            if value[i] == '$':
                if iStart == -1:
                    iStart = i
                else:
                    references.append(value[iStart:i+1])
                    iStart = -1

        for reference in references:
            keys   = reference[1:-1].split('.')
            subVal = self.recSolveReference(keys, t_maxDepth-1)
            value  = value.replace(reference, subVal)
        recDictSet(self.m_options, list(t_keys), value)
        return value

    #_________________________

    def solveReferences(self):
        # replace references with their right value
        keyList  = []
        ckl      = []
        recMakeKeyList(self.m_options, keyList, ckl)
        maxDepth = len(keyList) + 1
        for key in keyList:
            self.recSolveReference(key, maxDepth)
            maxDepth -= 1

#__________________________________________________

