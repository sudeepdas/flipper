"""
A module for error reporting
"""

default = "default"
traceDict = { default: 0 }

def getLevel( traceName ):
    """
    @brief get trace verbosity level
    @param traceName string - the name of the trace
    @return integer - trace verbosity level
    """
    if traceName in traceDict.keys():
        level = traceDict[traceName]
    else:
        level = traceDict[default]
    return level


def setLevel( traceName, level ):
    """
    @brief set the reporting level of a given trace
    @param traceName string - the name of the trace
    @param level integer - the trace verbosity (lower=quieter) 
    @return integer - the old trace verbosity level
    """
    oldLevel = getLevel(traceName)
    traceDict[traceName] = level
    return oldLevel

def issue( traceName, level, message ):
    """
    @brief issue a trace statement
    @param traceName string - the name of the trace
    @param level integer - the trace verbosity (lower=quieter) 
    @param message string - message to issue
    """
    curLevel = getLevel(traceName)
    if level <= curLevel:
        space = " "*level
        print "%s%s: %s" % (space, traceName, message)
