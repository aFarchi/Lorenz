#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/auxiliary/
# dictutils.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/5
#__________________________________________________
#
# fonctions related to dict
#

#__________________________________________________

def recDictWrite(t_dict, t_indentationLevel, t_lastKey, t_file, t_writeKeyLine, t_writeValueLine):
    # for a dict of dict of ... of dict, with an arbitrary depth level,
    # write dict to file
    if isinstance(t_dict, dict):
        t_writeKeyLine(t_indentationLevel, t_lastKey, t_file)
        t_file.write('\n')
        for key in t_dict:
            recDictWrite(t_dict[key], t_indentationLevel+1, key, t_file, t_writeKeyLine, t_writeValueLine)
        t_file.write('\n')
    else:
        t_writeValueLine(t_indentationLevel, t_lastKey, t_dict, t_file)

#__________________________________________________

def dictWrite(t_dict, t_file, t_writeKeyLine, t_writeValueLine):
    # for a dict of dict of ... of dict, with an arbitrary depth level,
    # write dict to file
    for key in t_dict:
        recDictWrite(t_dict[key], 0, key, t_file, t_writeKeyLine, t_writeValueLine)

#__________________________________________________

def recDictGet(t_dict, t_keyList):
    # for a dict of dict of ... of dict, with an arbitrary depth level,
    # return dict[keyList[0]]...[keyList[-1]]
    # note: this function modifies keyList
    if len(t_keyList) == 0:
        return t_dict
    else:
        key = t_keyList.pop(0)
        return recDictGet(t_dict[key], t_keyList)

#__________________________________________________

def recDictSet(t_dict, t_keyList, t_value):
    # for a dict of dict of ... of dict, with an arbitrary depth level,
    # set dict[keyList[0]]...[keyList[-1]] = value
    # note: this function modifies keyList
    if len(t_keyList) == 1:
        t_dict[t_keyList[0]] = t_value
    else:
        key = t_keyList.pop(0)
        if not key in t_dict:
            t_dict[key] = {}
        recDictSet(t_dict[key], t_keyList, t_value)

#__________________________________________________

def recMakeKeyList(t_dict, t_keyList, t_currentKeyList):
    # for a dict of dict of ... of dict, with an arbitrary depth level,
    # construct the list of all possible keys
    if isinstance(t_dict, dict):
        for key in t_dict:
            recMakeKeyList(t_dict[key], t_keyList, t_currentKeyList+[key])
    else:
        t_keyList.append(t_currentKeyList)

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

