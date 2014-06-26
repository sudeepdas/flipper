import numpy
import sys
import os
import pylab
import fftTools
import utils
import trace

class mtm:
    def __init__(self,map,Nres,Ntap,Niter,map2=None,mask=None):
        assert(Ntap>0)
        self.map = map
        self.Nres = Nres
        self.Ntap = Ntap
        self.Niter = Niter
        self.map2 = self.map.copy()
        if map2 !=None:
            assert(map.data.shape == map2.data.shape)
            self.map2 = map2
        
        self.mask = self.map.copy()
        self.mask.data[:] = 1.0

        if mask != None:
            assert(mask.data.shape == map.data.shape)
            self.mask = mask

        tapers,eigs = utils.slepianTapers(map.Ny,map.Nx,Nres,Ntap)
        self.tapers = tapers
        self.eigs = eigs

        del tapers, eigs

    def _getWeights(self,p2d):
        mean = self.map.data.mean()
        mean2 = self.map2.data.mean()

        sigmaSq = numpy.mean((self.map.data-mean)*(self.map2.data-mean2))
        sigmaSq *= self.map.pixScaleX*self.map.pixScaleY
        weights = numpy.zeros([self.map.Ny,self.map.Nx,self.Ntap,self.Ntap])
        if self.Niter == 0:
            weights[:,:,:,:] = 1.
        else:
            for k in xrange(self.Ntap*self.Ntap):
                i = k/self.Ntap
                j = numpy.mod(k,self.Ntap)
                
                weights[:,:,i,j]=p2d.powerMap[:,:]/(self.eigs[i,j]*p2d.powerMap[:,:]+\
                                                (1.-self.eigs[i,j])*sigmaSq)

                #pylab.matshow((numpy.fft.fftshift(weights[:,:,i,j])))
                #pylab.title('weights_%3.1f%d%d_%d_%d'%(self.Nres,self.Ntap,self.Niter,i,j))
                #pylab.colorbar()
                #pylab.savefig('weights_%3.1f%d%d_%d_%d.png'%(self.Nres,self.Ntap,self.Niter,i,j))
                #pylab.clf()
        return weights
    
    
    def generatePower(self):

        #empty place-holder for final power spectrum
        p2d = fftTools.powerFromLiteMap(self.map)
        p2d.powerMap[:] = 0.

        #Best guess power
        tempMap = self.map.copy()
        tempMap2 = self.map2.copy()
        tempMap.data[:,:] *= self.mask.data[:,:]
        tempMap2.data[:,:] *= self.mask.data[:,:]
        p2dBest = fftTools.powerFromLiteMap(tempMap,tempMap2,applySlepianTaper=True) #00 taper
        
        del tempMap, tempMap2
        
        #place-holder for total weights
        totWeight = numpy.zeros([self.map.Ny,self.map.Nx]) 
        
        num_iter = self.Niter
        if self.Niter == 0:
            num_iter = 1

        for k in  xrange(num_iter):
            trace.issue('mtm',2,'Iteration ...%02d'%k)
            weights = self._getWeights(p2dBest)
            p2d.powerMap[:] = 0.
            totWeight[:] = 0.
            for i in xrange(self.Ntap):
                for j in xrange(self.Ntap):
                                    
                    tempMap = self.map.copy()
                    tempMap2 = self.map2.copy()
                    tempMap.data[:,:] *= self.tapers[:,:,i,j]*self.mask.data[:,:]
                    tempMap2.data[:,:] *= self.tapers[:,:,i,j]*self.mask.data[:,:]
                    p2dRunning = fftTools.powerFromLiteMap(tempMap,tempMap2)
                    p2d.powerMap[:,:] += self.eigs[i,j]*weights[:,:,i,j]**2*p2dRunning.powerMap[:,:]
                    totWeight[:,:] += self.eigs[i,j]*weights[:,:,i,j]**2
                    del tempMap
                    del tempMap2
                    del p2dRunning
                    
                    
            p2d.powerMap[:] /= totWeight[:]
            p2dBest.powerMap[:] = p2d.powerMap[:] #new best guess
            

        return p2d
