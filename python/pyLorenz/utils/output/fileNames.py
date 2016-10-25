#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/output/
# fileNames.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/23
#__________________________________________________
#
# fonctions used to determine file names
#

#__________________________________________________

def outputSubDir(t_outputDir, t_observation_dt, t_observation_var):
    # output sub dir
    return ( t_outputDir                                   + 
            str(t_observation_dt).replace('.', 'p')  + '/' +
            str(t_observation_var).replace('.', 'p') + '/' )

#__________________________________________________

def filterLabel(t_filter, t_filter_Ns, t_filter_inflation, t_filter_res_thr):
    # filter label
    return ( t_filter                                       +
            '_' + str(t_filter_Ns)                          +
            '_' + str(t_filter_inflation).replace('.', 'p') +
            '_' + str(t_filter_res_thr).replace('.', 'p')   )

#__________________________________________________

