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

from path import Path

from utils.auxiliary.tree       import Tree, NoDefault
from utils.auxiliary.decoration import castDefaultKWArgsDecorator

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
        self.m_tree.tofile(t_fileName)

    #_________________________

    def write(self, t_file):
        # write
        self.m_tree.write(t_file)

    #_________________________

    def readfiles(self, t_fileNames):
        # read files
        self.m_tree.readfiles(t_fileNames, self.m_commentChar, self.m_referenceChar)

    #_________________________

    def get(self, *t_options, **t_default):
        # get
        default = t_default.get('default', NoDefault)
        return self.m_tree.get(list(t_options), default)

    #_________________________

    getInt = castDefaultKWArgsDecorator('string', 'int', KeyError)(get)

    #_________________________

    getFloat = castDefaultKWArgsDecorator('string', 'float', KeyError)(get)

    #_________________________

    getStringList = castDefaultKWArgsDecorator('string', 'stringlist', KeyError)(get)

    #_________________________

    getNumpyArray = castDefaultKWArgsDecorator('string', 'numpyarray', KeyError)(get)

    #_________________________

    def set(self, *t_options):
        # set
        (*options, value) = t_options
        self.m_tree.set(options, value)

#__________________________________________________

