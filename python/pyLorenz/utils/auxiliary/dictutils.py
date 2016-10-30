#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/auxiliary/
# dictutils.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/23
#__________________________________________________
#
# fonctions related to dict
#

#__________________________________________________

def recDictElement(t_dict, t_keyList):
    # for a dict of dict of ... of dict of list, with an arbitrary depth level,
    # return dict[keyList[-1]]...[keyList[0]]
    # note: this function modifies keyList
    if t_keyList == []:
        return t_dict
    else:
        key = t_keyList.pop()
        return recDictElement(t_dict[key], t_keyList)

#__________________________________________________

def dictElement(t_dict, t_keyList):
    # auxiliary function for recDictElement()
    # that does not modify keyList
    return recDictElement(t_dict, list(t_keyList))

#__________________________________________________

def recMakeKeyListDict(t_dict, t_keyListDict, t_currentKeyList):
    # for a dict of dict of ... of dict of list, with an arbitrary depth level
    # build a dict keyListDict such that :
    # t_keyListDict[e] is the list of keys for t_dict giving the list containing e
    if isinstance(t_dict, dict):
        for key in t_dict:
            recMakeKeyListDict(t_dict[key], t_keyListDict, [key]+t_currentKeyList)
    elif isinstance(t_dict, list):
        for element in t_dict:
            t_keyListDict[element] = [element] + t_currentKeyList

#__________________________________________________

def makeKeyListDict(t_dict):
    # auxiliary function for recMakeKeyListDict()
    kld = {}
    recMakeKeyListDict(t_dict, kld, [])
    return kld

#__________________________________________________

