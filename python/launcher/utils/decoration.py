#!/usr/bin/env python

#__________________________________________________
# launcher/utils/
# decoration.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/8
#__________________________________________________
#
# functions related to decoration
# copied from toolbox
#

from functools         import wraps

from utils.stringutils import stringToInt, stringToFloat, stringToStringList, stringToNumpyArray

#__________________________________________________

def afterDecorator(t_functionAfter):
    def decorator(t_function):
        @wraps(t_function)
        def wrapper(*t_args, **t_kwargs):
            return t_functionAfter(t_function(*t_args, **t_kwargs))
        return wrapper
    return decorator

#__________________________________________________

def afterDefaultKWArgsDecorator(t_functionAfter, t_exceptions):
    def decorator(t_function):
        @wraps(t_function)
        def wrapper(*t_args, **t_default):
            try:
                return t_functionAfter(t_function(*t_args))
            except t_exceptions:
                return t_default['default']
        return wrapper
    return decorator

#__________________________________________________

def castDefaultKWArgsDecorator(t_from, t_to, t_exceptions):

    if issubclass(t_exceptions, Exception):
        exceptions = [t_exceptions]
    else:
        exceptions = list(t_exceptions)
    exceptions.append(TypeError)
    exceptions = tuple(exceptions)

    if t_from == 'string' and t_to == 'int':
        return afterDefaultKWArgsDecorator(stringToInt, exceptions)

    if t_from == 'string' and t_to == 'float':
        return afterDefaultKWArgsDecorator(stringToFloat, exceptions)

    if t_from == 'string' and t_to == 'stringlist':
        return afterDefaultKWArgsDecorator(stringToStringList, exceptions)

    if t_from == 'string' and t_to == 'numpyarray':
        return afterDefaultKWArgsDecorator(stringToNumpyArray, exceptions)

#__________________________________________________

