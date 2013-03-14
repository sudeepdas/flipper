#use to import all Flipper routines with:
#from flipper import *


#Useful python builtins
import os,sys,time
import numpy

#Pylab
import pylab
pylab.load = numpy.loadtxt
#Other dependencies
import healpy

#flipper specific

import utils
import flipperDict
import fftTools
import liteMap
import prewhitener
import flTrace
import astLib
import mtm


pylab.load  = numpy.loadtxt
