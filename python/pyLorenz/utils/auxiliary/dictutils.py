#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/auxiliary/
# dictutils.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/10
#__________________________________________________
#
# fonctions related to dict
#

#__________________________________________________

def makeKeyDictRecListDict(t_dict, t_currentKeys = []):
    # for a dict of ... of dict of list with an arbitrary depth level
    # make a dict kd such that:
    # for all e in kd, t_dict[kd[e][0]]...[kd[e][-1]] = [..., e, ...]
    kd = {}
    if isinstance(t_dict, dict):
        for key in t_dict:
            skd = makeKeyDictRecListDict(t_dict[key], t_currentKeys+[key])
            for skey in skd:
                kd[skey] = skd[skey]
    else:
        for key in t_dict:
            kd[key] = t_currentKeys
    return kd

#__________________________________________________

