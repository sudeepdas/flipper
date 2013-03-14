import liteMap
import numpy
import scipy.special as sp
import utils
import pylab
import flTrace

def discDiffLspace(radius,ell):
    radiusRadian = radius*numpy.pi/(180.*60.)
    lR = ell*radiusRadian
    return (2.*(sp.j1(lR)/(lR)-sp.j1(3*lR)/(3*lR)))
    
def discDifference(radius,ltMap,discKern1=None,discKern3=None):
    """
    @radius radius in arcmin
    
    """
    
    radiusRadian = radius*numpy.pi/(180.*60.)
    #semiY = int(radiusRadian/ltMap.pixScaleY+0.5)
    #semiX = int(radiusRadian/ltMap.pixScaleX+0.5)
    semiY = (radiusRadian/ltMap.pixScaleY)
    semiX = (radiusRadian/ltMap.pixScaleX)
    
    dataR = utils.discSmooth(ltMap.data,semiY,semiX,discKrn=discKern1)
    data3R = utils.discSmooth(ltMap.data,3*semiY,3*semiX,discKrn=discKern3)

    discDiffMap = ltMap.copy()
    discDiffMap.data[:] = dataR[:]-data3R[:]
    return discDiffMap


class prewhitener:
    """
    @brief Class describing pre-whitening operation
    """
    def  __init__(self,radius,addBackFraction = 0., smoothingFWHM = 0.,map=None):
        """
        @brief contructor for prewhitener class
        @param radius radius for disc differencing (arcmin)
        @param addBackFraction Fractio of original map added back
        @param smoothingFWHM smoothing beam FWHM (arcmin)
        
        """
        self.radius = radius
        self.addBackFraction = addBackFraction
        self.smoothingFWHM = smoothingFWHM
        if map != None:
            radiusRadian = radius*numpy.pi/(180.*60.)
            #semiY = int(radiusRadian/map.pixScaleY+0.5)
            #semiX = int(radiusRadian/map.pixScaleX+0.5)
            semiY = (radiusRadian/map.pixScaleY)
            semiX = (radiusRadian/map.pixScaleX)
            self.discKern1 = utils.discKern(semiY,semiX)
            self.discKern3 = utils.discKern(3*semiY,3*semiX)
        else:
            self.discKern1 = None
            self.discKern3 = None
            
    def apply(self,ltMap):
        discDiffMap = discDifference(self.radius,ltMap,\
                                     discKern1=self.discKern1,\
                                     discKern3=self.discKern3)
        flTrace.issue("flipper.prewhitener",5,"Applying disc differencing with R=%f arcmin"%self.radius)
        
        discDiffMap.data[:] += self.addBackFraction*ltMap.data[:]
        if self.smoothingFWHM>0.:
            flTrace.issue("flipper.prewhitener",5,"Smoothing with FWHM: %f arcmin"%self.smoothingFWHM)
            discDiffMap = discDiffMap.convolveWithGaussian(self.smoothingFWHM)
        return discDiffMap

    
    def correctSpectrum(self,l,Cl):
        """
        @brief remove the prewhitening window from the power spectrum
        @param l multipoles
        @param Cl PS of prewhitened map
        @return corrected Cl        
        """
        # remove beam
        beamFWHM = self.smoothingFWHM*numpy.pi/(180.*60.)
        invBeamWindow = numpy.exp(l*(l+1.)*beamFWHM**2/(8*numpy.log(2.)))
        Clcor = Cl.copy()
        Clcor *= invBeamWindow
        # remove added fraction and discDifferencing window
        
        W = discDiffLspace(self.radius,l)
        # print"w =", W, invBeamWindow
        W[:] += self.addBackFraction
        
        Clcor[:] /=W[:]**2
        # pylab.semilogy(l,1./W**2)
        # pylab.show()
        return Clcor
    
