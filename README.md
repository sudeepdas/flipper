flipper
=======

Documentation can be found at:
http://www.hep.anl.gov/sdas/flipperDocumentation/


Flipper is a light-weight python tool for working with CMB data which broadly provides three main functionalities:
A suite of tools for operations performed on maps, like application of filters, taking gradients etc.
An FFT and power spectrum tool, 
implementing the core concepts from Das, Hajian and Spergel (2008) http://arxiv.org/abs/0809.1092v1

Flipper has become one of the most heavily used base codes in the Atacama Cosmology Telescope collaboration 
and led to a slew of papers with over 300 citations. 


Dependencies
==============

Flipper should work out of the box if it can import the following:

numpy 

scipy http://www.scipy.org/

pyfits http://www.stsci.edu/resources/software_hardware/pyfits

astLib http://astlib.sourceforge.net/

matplotlib http://matplotlib.sourceforge.net/

healpy http://code.google.com/p/healpy/


Installation
===============

Flipper does not as yet have a standard python package installation. It will come soon! 

For now, after forking and cloning flipper, 
put the following commands in your startup script (e.g., .bashrc) :

export FLIPPER_DIR=/path/to/your/flipper/

export PATH=$PATH:$FLIPPER_DIR/bin 

export PYTHONPATH=$PYTHONPATH:$FLIPPER_DIR/python
