
#__________________________________________________
# ../
# Sconstruct
#__________________________________________________
# author        : colonel
# last modified : 2016/9/16
#__________________________________________________
#
# compile script
#

class Configuration:
    pass

#______________
# Configuration
#______________

common_configuration            = Configuration()

common_configuration.precision  = 'double' # one in 'double', 'float'

common_configuration.proof      = False
common_configuration.test       = False

debug_configuration             = Configuration()
debug_configuration.label       = 'debug'
debug_configuration.build       = True
debug_configuration.directory   = 'debug/'
debug_configuration.common      = common_configuration

release_configuration           = Configuration()
release_configuration.label     = 'release'
release_configuration.build     = False
release_configuration.directory = 'release/'
release_configuration.common    = common_configuration

#___________________
# Common environment
#___________________

common_env = DefaultEnvironment()

# C++ compiler
common_env.Replace(CXX='/usr/local/bin/g++')

# Flags
common_flags               = {}
common_flags['CCFLAGS']    = ['-std=c++0x', '-Wall', '-Wextra', '-Wshadow', '-Wnon-virtual-dtor', '-Wunused', '-Woverloaded-virtual', '-Wold-style-cast', '-pedantic']
common_flags['VERSION']    = [1]
common_flags['LINKFLAGS']  = []
common_flags['CPPDEFINES'] = []

if common_configuration.precision == 'double':
    common_flags['CPPDEFINES'] += ['PRECISION_DOUBLE']
elif common_configuration.precision == 'float':
    common_flags['CPPDEFINES'] += ['PRECISION_FLOAT']

if common_configuration.proof:
    common_flags['CPPDEFINES'] += ['PROOF']
if common_configuration.test:
    common_flags['CPPDEFINES'] += ['TEST']

common_env.MergeFlags(common_flags)

#__________________
# Debug environment
#__________________

debug_env               = common_env.Clone()
debug_env.configuration = debug_configuration

# Flags

debug_flags               = {}
debug_flags['CPPDEFINES'] = ['DEBUG']
debug_flags['CCFLAGS']    = ['-Og']

debug_env.MergeFlags(debug_flags)

# Building directory

debug_env.VariantDir('build/'+debug_env.configuration.directory, 'source', duplicate=0)

#____________________
# Release environment
#____________________

release_env               = common_env.Clone()
release_env.configuration = release_configuration

# Flags

release_flags               = {}
release_flags['CPPDEFINES'] = ['RELEASE']
release_flags['CCFLAGS']    = ['-Ofast']

release_env.MergeFlags(release_flags)

# Building directory

release_env.VariantDir('build/'+release_env.configuration.directory, 'source', duplicate=0)

#______________
# Build targets
#______________

for env in [debug_env, release_env]:
    if env.configuration.build:
        env.SConscript('build/'+env.configuration.directory+'SConscript', dict(env=env))

