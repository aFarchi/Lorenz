#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/trigger/
# modulustrigger.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/29
#__________________________________________________
#
#
#

import numpy as np

#__________________________________________________

class ModulusTrigger(object):

    #_________________________

    def __init__(self, t_ntMod, t_ntFst):
        self.setModulusTriggerParameters(t_ntMod, t_ntFst)

    #_________________________

    def setModulusTriggerParameters(self, t_ntMod, t_ntFst):
        self.m_ntMod = t_ntMod
        self.m_ntFst = t_ntFst

    #_________________________

    def trigger(self, t_value, t_nt):
        return ( np.mod(t_nt, self.m_ntMod) == self.m_ntFst )

#__________________________________________________

