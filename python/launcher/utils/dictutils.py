#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# dictutils.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/14
#__________________________________________________
#
# fonctions related to dict
# copied from toolbox
#

from collections import OrderedDict

#__________________________________________________

def makeKeyDictRecListDict(t_dict, t_currentKeys = []):
    # for a dict of ... of dict of list with an arbitrary depth level
    # make a dict kd such that:
    # for all e in kd, dict[kd[e][0]]...[kd[e][-1]] = [..., e, ...]
    kd = OrderedDict()
    if isinstance(t_dict, dict) or isinstance(t_dict, OrderedDict):
        for key in t_dict:
            skd = makeKeyDictRecListDict(t_dict[key], t_currentKeys+[key])
            kd.update(skd)
    else:
        for key in t_dict:
            kd[key] = t_currentKeys
    return kd

#__________________________________________________

def makeKeyDictRecDict(t_dict, t_currentKeys = []):
    # for a dict of ... of dict with an arbitrary depth level
    # make a dict kd such that
    # for all e in kd, dict[kd[e][0]]...[kd[e][-1]] = e
    kd = OrderedDict()
    for key in t_dict:
        if isinstance(t_dict[key], dict) or isinstance(t_dict[key], OrderedDict):
            skd = makeKeyDictRecDict(t_dict[key], t_currentKeys + [key])
            kd.update(skd)
        else:
            kd[t_dict[key]] = t_currentKeys + [key]
    return kd

#__________________________________________________

def recDictSet(t_dict, t_keys, t_value):
    # set dict[keys[0]]...[keys[-1]] = value
    if not t_keys:
        return
    key = t_keys.pop(0)
    if t_keys:
        if not key in t_dict:
            t_dict[key] = OrderedDict()
        recDictSet(t_dict[key], t_keys, t_value)
    else:
        t_dict[key] = t_value

#__________________________________________________

