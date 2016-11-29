#!/usr/bin/env python

#__________________________________________________
# pyLorenz/utils/localisation/
# taper.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/28
#__________________________________________________
#
# taper functions for localisation
#

import numpy as np

#__________________________________________________

def gaussian_tapering(t_distance, t_radius):
    return np.exp(-0.5*(t_distance/t_radius)**2)

#__________________________________________________

def gaspari_cohn_tapering(t_distance, t_radius):
    R  = t_radius * 1.7386
    r1 = t_distance / R
    r2 = r1 * r1
    r3 = r2 * r1
    return ( (r1<=1.0)*(1.0+r2*(r2/2.0-r3/4.0)+r3*(5.0/8.0)-r2*(5.0/3.0)) +
            (r1>1)*(r1<2)*(r2*(r3/12.0-r2/2.0)+r3*(5.0/8.0)+r2*(5.0/3.0)-r1*5.0+4.0-(2.0/3.0)/np.maximum(r1, 1.0)) )
#__________________________________________________

def heaviside_tapering(t_distance, t_radius):
    return 1.0*(t_distance<=t_radius)

#__________________________________________________

