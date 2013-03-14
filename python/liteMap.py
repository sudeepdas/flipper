"""
@file liteMap.py
@brief Tools for real space maps
@author Sudeep Das and Toby Marriage
"""

import os, sys, copy
import numpy, scipy
import pylab
import copy
import pyfits
import astLib
from utils import *
from fftTools import fftFromLiteMap
import fftTools
import flTrace
import healpy
import utils
import time
from scipy.interpolate import splrep,splev
class gradMap:
    """
    @brief  Class describing gradient of a liteMap
    """
    def __init__(self):
        pass
    def plot(self,show=True,title=''):
        self.gradX.plot(title = title + '(X derivative)',show=show)
        self.gradY.plot(title = title + '(Y deriative)',show=show)
        
        

class liteMap:
    """
    @brief Class describing a 2-D real space map.
    """
    def __init__(self):
        pass

    def copy(self):
        """
        @brief Creates a copy of the liteMap object
        """
        return copy.copy(self)

    def __getstate__(self):
        dic =  self.__dict__.copy()
        dic['header'] = dic['header'].copy()
        dic['wcs'] = dic['wcs'].copy()
        dic['data'] = dic['data'].copy()
        return dic

    def __setstate__(self,dict):
        self.__dict__ = dict
        
    def info(self,showHeader=False):
        """
        @brief pretty print informstion sbout the litMap
        """
        arcmin = 180*60./numpy.pi
        print "Dimensions (Ny,Nx) = (%d,%d)"%(self.Ny,self.Nx)
        print "Pixel Scales: (%f,%f) arcmins. "%(self.pixScaleY*arcmin,self.pixScaleX*arcmin)
        print "Map Bounds: [(x0,y0), (x1,y1)]: [(%f,%f),(%f,%f)] (degrees)"%(self.x0,self.y0,self.x1,self.y1)
        print "Map Bounds: [(x0,y0), (x1,y1)]: [(%s,%s),(%s,%s)]"%\
              (astLib.astCoords.decimal2hms(self.x0,':'),\
               astLib.astCoords.decimal2dms(self.y0,':'),\
               astLib.astCoords.decimal2hms(self.x1,':'),\
               astLib.astCoords.decimal2dms(self.y1,':'))
        
        print "Map area = %f sq. degrees."%(self.area)
        print "Map mean = %f"%(self.data.mean())
        print "Map std = %f"%(self.data.std())
        
        if showHeader:
            print "Map header \n %s"%(self.header)


    def fillWithGaussianRandomField(self,ell,Cell,bufferFactor = 1):
        """
        Generates a GRF from an input power spectrum specified as ell, Cell 
        BufferFactor =1 means the map will be periodic boundary function
        BufferFactor > 1 means the map will be genrated on  a patch bufferFactor times 
        larger in each dimension and then cut out so as to have non-periodic bcs.
        
        Fills the data field of the map with the GRF realization
        """
        
        ft = fftFromLiteMap(self)
        Ny = self.Ny*bufferFactor
        Nx = self.Nx*bufferFactor
        
        bufferFactor = int(bufferFactor)
        
        
        realPart = numpy.zeros([Ny,Nx])
        imgPart  = numpy.zeros([Ny,Nx])
        
        ly = numpy.fft.fftfreq(Ny,d = self.pixScaleY)*(2*numpy.pi)
        lx = numpy.fft.fftfreq(Nx,d = self.pixScaleX)*(2*numpy.pi)
        #print ly
        modLMap = numpy.zeros([Ny,Nx])
        iy, ix = numpy.mgrid[0:Ny,0:Nx]
        modLMap[iy,ix] = numpy.sqrt(ly[iy]**2+lx[ix]**2)
        
        s = splrep(ell,Cell,k=3)
        
        ll = numpy.ravel(modLMap)
        kk = splev(ll,s)
        id = numpy.where(ll>ell.max())
        kk[id] = 0.
        #add a cosine ^2 falloff at the very end
        #id2 = numpy.where( (ll> (ell.max()-500)) & (ll<ell.max()))
        #lEnd = ll[id2]
        #kk[id2] *= numpy.cos((lEnd-lEnd.min())/(lEnd.max() -lEnd.min())*numpy.pi/2)
        
        #pylab.loglog(ll,kk)

        area = Nx*Ny*self.pixScaleX*self.pixScaleY
        p = numpy.reshape(kk,[Ny,Nx]) /area * (Nx*Ny)**2
                
        
        realPart = numpy.sqrt(p)*numpy.random.randn(Ny,Nx)
        imgPart = numpy.sqrt(p)*numpy.random.randn(Ny,Nx)
        
        
        kMap = realPart+1j*imgPart
        
        data = numpy.real(numpy.fft.ifft2(kMap)) 
        
        b = bufferFactor
        self.data = data[(b-1)/2*self.Ny:(b+1)/2*self.Ny,(b-1)/2*self.Nx:(b+1)/2*self.Nx]
        
        


    def fillWithGRFFromTemplate(self,twodPower,bufferFactor = 1):
        """
        Generates a GRF from an input power spectrum specified as a 2d powerMap
        BufferFactor =1 means the map will be periodic boundary function
        BufferFactor > 1 means the map will be genrated on  a patch bufferFactor times 
        larger in each dimension and then cut out so as to have non-periodic bcs.
        
        Fills the data field of the map with the GRF realization
        """
        
        ft = fftFromLiteMap(self)
        Ny = self.Ny*bufferFactor
        Nx = self.Nx*bufferFactor
        
        bufferFactor = int(bufferFactor)
        assert(bufferFactor>=1)
        
        
        realPart = numpy.zeros([Ny,Nx])
        imgPart  = numpy.zeros([Ny,Nx])
        
        ly = numpy.fft.fftfreq(Ny,d = self.pixScaleY)*(2*numpy.pi)
        lx = numpy.fft.fftfreq(Nx,d = self.pixScaleX)*(2*numpy.pi)
        #print ly
        modLMap = numpy.zeros([Ny,Nx])
        iy, ix = numpy.mgrid[0:Ny,0:Nx]
        modLMap[iy,ix] = numpy.sqrt(ly[iy]**2+lx[ix]**2)

        if bufferFactor > 1:
            ell = numpy.ravel(twodPower.modLMap)
            Cell = numpy.ravel(twodPower.powerMap)
            print ell
            print Cell
            s = splrep(ell,Cell,k=3)
        
            
            ll = numpy.ravel(modLMap)
            kk = splev(ll,s)
            
            
            id = numpy.where(ll>ell.max())
            kk[id] = 0.
            # add a cosine ^2 falloff at the very end
            # id2 = numpy.where( (ll> (ell.max()-500)) & (ll<ell.max()))
            # lEnd = ll[id2]
            # kk[id2] *= numpy.cos((lEnd-lEnd.min())/(lEnd.max() -lEnd.min())*numpy.pi/2)
            
            # pylab.loglog(ll,kk)

            area = Nx*Ny*self.pixScaleX*self.pixScaleY
            p = numpy.reshape(kk,[Ny,Nx]) /area * (Nx*Ny)**2
        else:
            area = Nx*Ny*self.pixScaleX*self.pixScaleY
            p = twodPower.powerMap/area*(Nx*Ny)**2
        
        realPart = numpy.sqrt(p)*numpy.random.randn(Ny,Nx)
        imgPart = numpy.sqrt(p)*numpy.random.randn(Ny,Nx)
        
        
        kMap = realPart+1j*imgPart
        
        data = numpy.real(numpy.fft.ifft2(kMap)) 
        
        b = bufferFactor
        self.data = data[(b-1)/2*self.Ny:(b+1)/2*self.Ny,(b-1)/2*self.Nx:(b+1)/2*self.Nx]
        
        




    def selectSubMap(self,x0,x1,y0,y1, safe = False):
        """
        Returns a submap given new map bounds e.g. ra0,ra1,dec0,dec1

        If safe, then only clip part of submap falling in map
        """
        if safe:
            epsilon = self.pixScaleY
            cosdec = numpy.cos((y0+y1)/2.*numpy.pi/180.)
            if x0 < self.x1:
                x0 = self.x1 + epsilon/cosdec
            if x1 > self.x0:
                x1 = self.x0 - epsilon/cosdec
            if y0 < self.y0:
                y0 = self.y0 + epsilon
            if y1 > self.y1:
                y1 = self.y1 - epsilon

        ix0,iy0 = self.skyToPix(x0,y0)
        ix1,iy1 = self.skyToPix(x1,y1)
        assert((ix0 >0) & (ix1>0))
        assert((iy0 >0) & (iy1>0))
        i0 = numpy.int(ix0+0.5)
        j0 = numpy.int(iy0+0.5)
        i1 = numpy.int(ix1+0.5)
        j1 = numpy.int(iy1+0.5)
        ixx = numpy.sort([i0,i1])
        iyy = numpy.sort([j0,j1])
        #print ixx,iyy
        data = (self.data.copy())[iyy[0]:iyy[1],ixx[0]:ixx[1]]
        wcs = self.wcs.copy()
        naxis2,naxis1 = data.shape
        wcs.header.update('NAXIS1',naxis1)
        wcs.header.update('NAXIS2',naxis2)
        wcs.header.update('CRPIX1',wcs.header['CRPIX1']-ixx[0])
        wcs.header.update('CRPIX2',wcs.header['CRPIX2']-iyy[0])
        wcs.updateFromHeader()
        smallMap = liteMapFromDataAndWCS(data,wcs)
        del data,wcs
        return smallMap
    
    def addHeaderKeyword( self, key, val ):
        """
        @brief add key/value pair to the header
        """
        self.wcs.header.update(key, val)
            
    def plot(self,valueRange = None,\
             show = True,\
             saveFig = None,\
             colorMapName = 'jet',\
             colBarLabel = None,\
             colBarOrient ='vertical',\
             colBarShrink = 1.0,\
             useImagePlot = False,\
             **kwd_args):
        """
        @brief Plots a liteMap using astLib.astPlots.ImagePlot.

        The axes can be marked in either sexagesimal or decimal celestial coordinates.
        If RATickSteps or decTickSteps are set to "auto", the appropriate axis scales will
        be determined automatically from the size of the image array and associated WCS.
        The tick step sizes can be overidden.
        If the coordinate axes are in sexagesimal format a dictionary in the format
        {'deg', 'unit'}. If the coordinate axes are in
        decimal format, the tick step size is specified simply in RA, dec decimal degrees.
        
        
        @param valueRange A tuple e.g. [-300,300] specifying the limits of the colorscale
        @param show Show the plot instead of saving
        @param saveFig save to a file
        @param colorMapName name of pylab.cm colorMap
        @param colBarLabel add label to the colorBar
        @param colBarOrient orientation of the colorbar (can be 'vertical'(default) or 'horizontal'
        @param colBarShrink shrink the color
        @param kwd_args all keywords accepted by astLib.astPlots.ImagePlot:
        @type axesLabels: string
        @param axesLabels: either "sexagesimal" (for H:M:S, D:M:S), "decimal" (for decimal degrees)
        or None (for no coordinate axes labels)
        @type axesFontFamily: string
        @param axesFontFamily: matplotlib fontfamily, e.g. 'serif', 'sans-serif' etc.
        @type axesFontSize: float
        @param axesFontSize: font size of axes labels and titles (in points)
        @param RATickSteps See docstring above
        @param decTickSteps  See docstring above
        """
        if valueRange!= None:
            vmin = valueRange[0]
            vmax = valueRange[1]
        else:
            vmin = self.data.min()
            vmax = self.data.max()

        # Convert name to a matplotlib.cm.Colormap instance
        try:
            cmap = pylab.cm.__dict__[colorMapName]
        except KeyError:
            cmap = pylab.cm.hsv

        if not(useImagePlot):
            pylab.imshow(self.data,origin="down",vmin=vmin,vmax=vmax,\
                         extent=[self.x0,self.x1,self.y0,self.y1],\
                         aspect=1./(numpy.cos(0.5*numpy.pi/180.*(self.y0+self.y1))),\
                         cmap=cmap)
        else:
            astLib.astPlots.ImagePlot(self.data,self.wcs,colorMapName=colorMapName,\
                                      cutLevels=[vmin,vmax],colorBar=False,**kwd_args)
        
        cb = pylab.colorbar(orientation=colBarOrient,shrink=colBarShrink)
        if colBarLabel != None:
            cb.set_label(colBarLabel)
            
        #pylab.xlabel('Ra (degrees)')
        #pylab.ylabel('Dec (degrees)')
        #pylab.title(title)
        if saveFig!=None:
            pylab.savefig(saveFig)
        if show:
            pylab.show()

    def takeGradient(self):
        """
        @brief takes gradient of a liteMap
        @return a gradMap object
        """
        gradY,gradX = numpy.gradient(self.data,self.pixScaleY,self.pixScaleX)
        gradm = gradMap()
        gradm.gradX = self.copy()
        gradm.gradY = self.copy()
        
        gradm.gradX.data[:,:] = gradX[:,:]
        gradm.gradY.data[:,:] = gradY[:,:]
        return gradm

    
    def takeLaplacian(self):
        """
        @brief Takes the laplacian of a liteMap
        """
        gradm = self.takeGradient()
        d1 = takeDivergence(gradm.gradX,gradm.gradY)
        del gradm
        return d1

    def convolveWithGaussian(self,fwhm=1.4,nSigma=5.0):
        """
        @brief convolve a map with a Gaussian beam (real space operation)
        @param fwhm Full Width Half Max in arcmin
        @param nSigma Number of sigmas the Gaussian kernel is defined out to.
        """

        fwhm *= numpy.pi/(180.*60.)
        sigmaY = fwhm/(numpy.sqrt(8.*numpy.log(2.))*self.pixScaleY)
        sigmaX = fwhm/(numpy.sqrt(8.*numpy.log(2.))*self.pixScaleX)
        smMap = self.copy()
        
        data = gaussianSmooth(self.data,sigmaY,sigmaX,nSigma=nSigma)
        smMap.data[:,:] = data[:,:]
        del data
        return smMap

    def filterFromList(self,lFl,setMeanToZero=False):
        """
        @brief Given an l-space filter as a tuple [\f$\ell,F_\ell\f$] returns a filtered map
        @param lFl A tuple [\f$\ell,F_\ell\f$], where \f$\ell\f$ and \f$ F_\ell \f$
        are 1-D arrays representing the filter.
        @return The filtered liteMap
        """
        ft = fftFromLiteMap(self)
        filtData = ft.mapFromFFT(kFilterFromList=lFl,setMeanToZero=setMeanToZero)
        filtMap = self.copy()
        filtMap.data[:] = filtData[:]
        del ft
        del filtData
        return filtMap

    def createGaussianApodization(self,pad=10,kern=5,extraYPad = 0):
        """
        @brief Creates a liteMap containing an apodization window
        @param pad the number of pixels that are first zeroed out at the edges
        @param kern the width of the Gaussian with which the above map is then convolved
        @return LiteMap contaning the apodization window
        """
        apod = self.copy()
        apod.data[:] = 0.0
        apod.data[pad+extraYPad:apod.Ny-pad-extraYPad,pad:apod.Nx-pad] = 1.0
        apod.data = scipy.ndimage.gaussian_filter(apod.data,kern,mode="constant")
        return apod

    def writeFits(self,filename,overWrite=False):
        """ @brief Write  a liteMap as a Fits file"""
        pyfits.writeto(filename,self.data,self.header,clobber=overWrite)

    def pixToSky(self,ix,iy):
        """
        @brief given pixel indices, returns position in WCS coordinate system.
        @param ix  x index of pixel; can be an array
        @param iy  y index of pixel; can be an array 
        @return lon (ra), lat (dec) degrees.
        @return If ix, iy are arrays, returns [[ra,dec],[ra,dec],...]
        """
        return self.wcs.pix2wcs(ix,iy)
    
    def skyToPix(self,lon,lat):
        """
        @brief Given lon (ra) ,lat (dec) in degrees returns pixel indices ix,, iy (real)
        @return ix,iy
        @return [[ix,iy],[ix,iy],....] if lon and lat are arrays
         """
        return self.wcs.wcs2pix(lon,lat)

    def loadDataFromHealpixMap(self, hpm, interpolate = False, hpCoords = "J2000"):
        """
        @brief copy data from a Healpix map (from healpy), return a lite map
        @param hpm healpy map
        @param interpolate use interpolation when copying 
        @param hpCoords coordinates of hpm (e.g., "J2000"(RA, Dec) or "GALACTIC")

        Assumes that liteMap is in J2000 RA Dec. The Healpix map must contain the liteMap.
        """
        inds = numpy.indices([self.Nx, self.Ny])
        x = inds[0].ravel()
        y = inds[1].ravel()
        skyLinear = numpy.array(self.pixToSky(x,y))
        ph = skyLinear[:,0]
        th = skyLinear[:,1]
        thOut = []
        phOut = []
        if hpCoords != "J2000":
            for i in xrange(len(th)):
                crd = astLib.astCoords.convertCoords("J2000", hpCoords, ph[i], th[i], 0.)
                phOut.append(crd[0])
                thOut.append(crd[1])
            thOut = numpy.array(thOut)
            phOut = numpy.array(phOut)
        else:
            thOut = th
            phOut = ph
        flTrace.issue("flipper.liteMap", 3, "theta (min, max): %f, %f" % (th.min(), th.max()))
        flTrace.issue("flipper.liteMap", 3, "phi (min, max): %f, %f" % (ph.min(), ph.max()))
        flTrace.issue("flipper.liteMap", 3, "phiOut (min, max): (%f, %f)  " %  ( phOut.min(), phOut.max() ))
        flTrace.issue("flipper.liteMap", 3, "thetaOut (min, max): (%f, %f)  " %  ( thOut.min(), thOut.max() ))
        phOut *= numpy.pi/180
        thOut = 90. - thOut #polar angle is 0 at north pole
        thOut *= numpy.pi/180
        flTrace.issue("flipper.liteMap", 3, "phiOut rad (min, max): (%f, %f)  " %  ( phOut.min(), phOut.max() ))
        flTrace.issue("flipper.liteMap", 3, "thetaOut rad (min, max): (%f, %f)  " %  ( thOut.min(), thOut.max() ))
        if interpolate:
            self.data[y,x] = healpy.get_interp_val(hpm, thOut, phOut)
        else:
            ind = healpy.ang2pix( healpy.get_nside(hpm), thOut, phOut )
            flTrace.issue("flipper.liteMap", 3, "healpix indices (min,max): %d, %d" % (ind.min(), ind.max()))
            self.data[:] = 0.
            self.data[[y,x]]=hpm[ind]


    def convertToMicroKFromJyPerSr(self, freqGHz, Tcmb = 2.726):
        """
        @brief Converts a map from Jy/Sr to uK.
        @param freqGHZ frequqncy in GHz
        @param optional value of Tcmb in Kelvin 
        """
        factor = _deltaTOverTcmbToJyPerSr(freqGHz,Tcmb)
        self.data[:] /= factor
        self.data[:] *= Tcmb*1e6

    def convertToJyPerSrFromMicroK(self,freqGHz,Tcmb = 2.726):
        """
        @brief Converts a map from uK to Jy/Sr.
        @param freqGHZ frequqncy in GHz
        @param optional value of Tcmb in Kelvin 
        """
        
        factor = _deltaTOverTcmbToJyPerSr(freqGHz,Tcmb)
        self.data[:] /= Tcmb*1e6
        self.data[:] *= factor

    def convertToComptonYFromMicroK(self,freqGHz,Tcmb = 2.726):
        """
        @brief Converts a map from uK to the Compton y-parameter.
        @param freqGHZ frequqncy in GHz
        @param optional value of Tcmb in Kelvin 
        """
        
        factor = _deltaTOverTcmbToY(freqGHz,Tcmb)
        self.data[:] /= Tcmb*1e6
        self.data[:] *= factor
    
    def convertToMicroKFromComptonY(self,freqGHz,Tcmb = 2.726):
        """
        @brief Converts a map from the Compton y-parameter to uK.
        @param freqGHZ frequqncy in GHz
        @param optional value of Tcmb in Kelvin 
        """
        
        factor = _deltaTOverTcmbToY(freqGHz,Tcmb)
        self.data[:] /= factor
        self.data[:] *= Tcmb*1e6

    def normalizeWCSWith(self,map):
        """
        Normalizes WCS of map
        """
        self.wcs.header.update('CDELT1',map.wcs.header['CDELT1'])
        self.wcs.header.update('CDELT2',map.wcs.header['CDELT2'])
        self.wcs.header.update('PV2_1',map.wcs.header['PV2_1'])
        self.wcs.updateFromHeader()
        self.header = self.wcs.header.copy()

def liteMapFromFits(file,extension=0):
    """
    @brief Reads in a FITS file and creates a liteMap object out of it.
    @param extension specify the FITS HDU where the map image is stored
    """
    ltmap = liteMap()
    hdulist = pyfits.open(file)
    header = hdulist[extension].header
    flTrace.issue('flipper.liteMap',3,"Map header \n %s"%header)
    
    ltmap.data = hdulist[extension].data.copy() 

    [ltmap.Ny,ltmap.Nx] = ltmap.data.shape

    wcs = astLib.astWCS.WCS(file,extensionName = extension)
    ltmap.wcs = wcs.copy()
    ltmap.header = ltmap.wcs.header
    ltmap.x0,ltmap.y0 = wcs.pix2wcs(0,0)
    ltmap.x1,ltmap.y1 = wcs.pix2wcs(ltmap.Nx-1,ltmap.Ny-1)
    
    #[ltmap.x0,ltmap.x1,ltmap.y0,ltmap.y1] = wcs.getImageMinMaxWCSCoords()
    if ltmap.x0 > ltmap.x1:
        ltmap.pixScaleX = numpy.abs(ltmap.x1-ltmap.x0)/ltmap.Nx*numpy.pi/180.\
                          *numpy.cos(numpy.pi/180.*0.5*(ltmap.y0+ltmap.y1))
    else:
        ltmap.pixScaleX = numpy.abs((360.-ltmap.x1)+ltmap.x0)/ltmap.Nx*numpy.pi/180.\
                          *numpy.cos(numpy.pi/180.*0.5*(ltmap.y0+ltmap.y1))
        
    ltmap.pixScaleY = numpy.abs(ltmap.y1-ltmap.y0)/ltmap.Ny*numpy.pi/180.
    #print 0.5*(ltmap.y0+ltmap.y1)
    ltmap.area = ltmap.Nx*ltmap.Ny*ltmap.pixScaleX*ltmap.pixScaleY*(180./numpy.pi)**2
    #print numpy.cos(numpy.pi/180.*0.5*(ltmap.y0+ltmap.y1))
    flTrace.issue('flipper.liteMap',1,'Reading file %s'%file)
    flTrace.issue("flipper.liteMap",1, "Map dimensions (Ny,Nx) %d %d"%\
                (ltmap.Ny,ltmap.Nx))
    flTrace.issue("flipper.liteMap",1, "pixel scales Y, X (degrees) %f %f"%\
                (ltmap.pixScaleY*180./numpy.pi,ltmap.pixScaleX*180./numpy.pi))
    
    return ltmap

        
def liteMapFromDataAndWCS(data,wcs):
    """
    @brief Given a numpy array: data and a astLib.astWCS instance: wcs creates a liteMap
    
    """
    ltmap = liteMap()
        
    ltmap.data = data.copy()

    [ltmap.Ny,ltmap.Nx] = ltmap.data.shape
    ltmap.wcs = wcs.copy()
    ltmap.header = ltmap.wcs.header
    
    #[ltmap.x0,ltmap.x1,ltmap.y0,ltmap.y1] = wcs.getImageMinMaxWCSCoords()
    ltmap.x0,ltmap.y0 = wcs.pix2wcs(0,0)
    ltmap.x1,ltmap.y1 = wcs.pix2wcs(ltmap.Nx-1,ltmap.Ny-1)

    if ltmap.x0 > ltmap.x1:
        ltmap.pixScaleX = numpy.abs(ltmap.x1-ltmap.x0)/ltmap.Nx*numpy.pi/180.\
                          *numpy.cos(numpy.pi/180.*0.5*(ltmap.y0+ltmap.y1))
    else:
        ltmap.pixScaleX = numpy.abs((360.-ltmap.x1)+ltmap.x0)/ltmap.Nx*numpy.pi/180.\
                          *numpy.cos(numpy.pi/180.*0.5*(ltmap.y0+ltmap.y1))
    ltmap.pixScaleY = numpy.abs(ltmap.y1-ltmap.y0)/ltmap.Ny*numpy.pi/180.
    
    #print 0.5*(ltmap.y0+ltmap.y1)
    ltmap.area = ltmap.Nx*ltmap.Ny*ltmap.pixScaleX*ltmap.pixScaleY*(180./numpy.pi)**2
    #print numpy.cos(numpy.pi/180.*0.5*(ltmap.y0+ltmap.y1))
    flTrace.issue('flipper.liteMap',1,'Reading file %s'%file)
    flTrace.issue("flipper.liteMap",1, "Map dimensions (Ny,Nx) %d %d"%\
                (ltmap.Ny,ltmap.Nx))
    flTrace.issue("flipper.liteMap",1, "pixel scales Y, X (degrees) %f %f"%\
                (ltmap.pixScaleY*180./numpy.pi,ltmap.pixScaleX*180./numpy.pi))
    
    return ltmap


def takeDivergence(vecMapX,vecMapY):
    """
    @brief Takes the divergence of a vector field whose x and y componenets are specified by
    two liteMaps
    """
    gradXY,gradXX = numpy.gradient(vecMapX.data,vecMapX.pixScaleY,vecMapX.pixScaleX)
    gradYY,gradYX = numpy.gradient(vecMapY.data,vecMapY.pixScaleY,vecMapY.pixScaleX)
    divMap = vecMapX.copy()
    divMap.data[:,:] = gradXX[:,:] + gradYY[:,:]
    return divMap

def _deltaTOverTcmbToJyPerSr(freqGHz,T0 = 2.726):
    """
    @brief the function name is self-eplanatory
    @return the converstion factor
    """
    kB = 1.380658e-16
    h = 6.6260755e-27
    c = 29979245800.
    nu = freqGHz*1.e9
    x = h*nu/(kB*T0)
    cNu = 2*(kB*T0)**3/(h**2*c**2)*x**4/(4*(numpy.sinh(x/2.))**2)
    cNu *= 1e23
    return cNu

def _deltaTOverTcmbToY(freqGHz, T0 = 2.726):
    kB = 1.380658e-16
    h = 6.6260755e-27
    c = 29979245800.
    nu = freqGHz*1.e9
    x = h*nu/(kB*T0)
    f_nu = x*(numpy.exp(x)+1)/(numpy.exp(x)-1) - 4
    return 1./f_nu 


def addLiteMapsWithSpectralWeighting( liteMap1, liteMap2, kMask1Params = None, kMask2Params = None, signalMap = None ):
    """
    @brief add two maps, weighting in Fourier space
    Maps must be the same size.
    @param kMask1Params mask params for liteMap1 (see fftTools.power2D.createKspaceMask)
    @param kMask2Params mask params for liteMap2 (see fftTools.power2D.createKspaceMask)
    @param signalMap liteMap with best estimate of signal to use when estimating noise weighting
    @return new map
    """
    #Get fourier weights
    flTrace.issue("liteMap", 0, "Computing Weights")
    #np1 = fftTools.noisePowerFromLiteMaps(liteMap1, liteMap2, applySlepianTaper = False)
    #np2 = fftTools.noisePowerFromLiteMaps(liteMap2, liteMap1, applySlepianTaper = False)
    data1 = copy.copy(liteMap1.data)
    data2 = copy.copy(liteMap2.data)
    if signalMap != None:
        liteMap1.data[:] = (liteMap1.data - signalMap.data)[:]
        liteMap2.data[:] = (liteMap2.data - signalMap.data)[:]
    np1 = fftTools.powerFromLiteMap(liteMap1)#, applySlepianTaper = True)
    np2 = fftTools.powerFromLiteMap(liteMap2)#), applySlepianTaper = True)
    print "testing", liteMap1.data == data1
    liteMap1.data[:] = data1[:]
    liteMap2.data[:] = data2[:]

    n1 = np1.powerMap
        
    n2 = np2.powerMap
#     n1[numpy.where( n1<n1.max()*.002)] = n1.max()*.001
#     n2[numpy.where( n2<n2.max()*.002)] = n2.max()*.001

    w1 = 1/n1
    w2 = 1/n2
    
    m1 = numpy.median(w1)
    m2 = numpy.median(w2)
    w1[numpy.where(abs(w1)>4*m1)]=4*m1
    w2[numpy.where(abs(w2)>4*m2)]=4*m2

    #w1[:] = 1.
    #w2[:] = 1.
    #pylab.hist(w1.ravel())
    #pylab.savefig("hist1.png")
    #pylab.clf()
    yrange = [4,5000]
    np1.powerMap = w1
    #np1.plot(pngFile="w1.png", log=True, zoomUptoL=8000, yrange = yrange)
    np2.powerMap = w2
    #np2.plot(pngFile="w2.png", log=True, zoomUptoL=8000, yrange = yrange)

    if kMask1Params != None:
        np1.createKspaceMask(**kMask1Params)
        w1 *= np1.kMask
        
    if kMask2Params != None:
        np2.createKspaceMask(**kMask2Params)
        w2 *= np2.kMask
    pylab.clf()
    
    invW = 1.0/(w1+w2)
    invW[numpy.where(numpy.isnan(invW))] = 0.
    invW[numpy.where(numpy.isinf(invW))] = 0.

    flTrace.issue("liteMap", 3, "NaNs in inverse weight: %s" % str(numpy.where(numpy.isnan(invW))))
    flTrace.issue("liteMap", 3, "Infs in inverse weight: %s" % str(numpy.where(numpy.isinf(invW))))


    flTrace.issue("liteMap", 2, "Adding Maps")
    f1  = fftTools.fftFromLiteMap( liteMap1, applySlepianTaper = False )
    f2  = fftTools.fftFromLiteMap( liteMap2, applySlepianTaper = False )
    kTot = (f1.kMap*w1 + f2.kMap*w2)*invW
    flTrace.issue("liteMap", 3, "NaNs in filtered transform: %s" % str(numpy.where(numpy.isnan(kTot))))
    f1.kMap = kTot
    finalMap = liteMap1.copy()
    finalMap.data[:] = 0.
    finalMap.data = f1.mapFromFFT()
    return finalMap



def normalizeWCS(map0,map1):
    """
    In case there is a small difference in some specific WCS keyword values (due to precision) of two maps
    this routine can be used  to make those keywords the same. The keywords affected are
    CDELT1, CDELT2 and PV2_1
    The routine copies the WCS values of map0 to map1.
    @param map0 a liteMap from which WCS keywords (CDELT,PV) are copied
    @param map1 a liteMap to which WCS keywords (CDELT,PV) are copied
    """

    wcs0 = map0.wcs.copy()
    wcs1 = map1.wcs.copy()
    wcs1.header.update('CDELT1',wcs0.header['CDELT1'])
    wcs1.header.update('CDELT2',wcs0.header['CDELT2'])
    wcs1.header.update('PV2_1',wcs0.header['PV2_1'])
    
    wcs1.updateFromHeader()
    data = map1.data.copy()
    del map1
    map1 = liteMapFromDataAndWCS(data,wcs1)


def upgradePixelPitch( m, N = 1 ):
    """
    @brief go to finer pixels with fourier interpolation
    @param m a liteMap
    @param N go to 2^N times smaller pixels
    @return the map with smaller pixels
    """
    Ny = m.Ny*2**N
    Nx = m.Nx*2**N
    npix = Ny*Nx

    ft = numpy.fft.fft2(m.data)
    ftShifted = numpy.fft.fftshift(ft)
    newFtShifted = numpy.zeros((Ny, Nx), dtype=numpy.complex128)

    # From the numpy.fft.fftshift help:
    # """
    # Shift zero-frequency component to center of spectrum.
    #
    # This function swaps half-spaces for all axes listed (defaults to all).
    # If len(x) is even then the Nyquist component is y[0].
    # """
    #
    # So in the case that we have an odd dimension in our map, we want to put
    # the extra zero at the beginning
    
    if m.Nx % 2 != 0:
        offsetX = (Nx-m.Nx)/2 + 1
    else:
        offsetX = (Nx-m.Nx)/2
    if m.Ny % 2 != 0:
        offsetY = (Ny-m.Ny)/2 + 1
    else:
        offsetY = (Ny-m.Ny)/2
    
    newFtShifted[offsetY:offsetY+m.Ny,offsetX:offsetX+m.Nx] = ftShifted
    del ftShifted
    ftNew = numpy.fft.ifftshift(newFtShifted)
    del newFtShifted

    # Finally, deconvolve by the pixel window
    mPix = numpy.copy(numpy.real(ftNew))
    mPix[:] = 0.0
    mPix[mPix.shape[0]/2-(2**(N-1)):mPix.shape[0]/2+(2**(N-1)),mPix.shape[1]/2-(2**(N-1)):mPix.shape[1]/2+(2**(N-1))] = 1./(2.**N)**2
    ftPix = numpy.fft.fft2(mPix)
    del mPix
    inds = numpy.where(ftNew != 0)
    ftNew[inds] /= numpy.abs(ftPix[inds])
    newData = numpy.fft.ifft2(ftNew)*(2**N)**2
    del ftNew
    del ftPix

    x0_new,y0_new = m.pixToSky(0,0)
    
    m = m.copy() # don't overwrite original 
    m.wcs.header.update('NAXIS1',  2**N*m.wcs.header['NAXIS1'] )
    m.wcs.header.update('NAXIS2',  2**N*m.wcs.header['NAXIS2'] )
    m.wcs.header.update('CDELT1',  m.wcs.header['CDELT1']/2.**N)
    m.wcs.header.update('CDELT2',  m.wcs.header['CDELT2']/2.**N)
    m.wcs.updateFromHeader()
    
    p_x, p_y = m.skyToPix(x0_new, y0_new)
    
    m.wcs.header.update('CRPIX1', m.wcs.header['CRPIX1'] - p_x)
    m.wcs.header.update('CRPIX2', m.wcs.header['CRPIX2'] - p_y)
    m.wcs.updateFromHeader()

    mNew = liteMapFromDataAndWCS(numpy.real(newData), m.wcs)
    mNew.data[:] = newData[:]
    return mNew

def getEmptyMapWithDifferentDims(m,Ny,Nx):
    """
    @brief Creates an empty map on the same patch of the sky as m 
    but with different dimensions Ny ,Nx
    @param m the template map
    @param Ny 
    @param Nx
    
    """
    data = numpy.zeros([Ny,Nx])
    m = m.copy()
    m.wcs.header.update('NAXIS1',Nx)
    m.wcs.header.update('NAXIS2',Ny)
    m.wcs.header.update('CDELT1',  m.wcs.header['CDELT1']*(m.Nx/(Nx*1.0)))
    m.wcs.header.update('CDELT2',  m.wcs.header['CDELT2']*(m.Ny/(Ny*1.0)))
    m.wcs.updateFromHeader()
    p_x, p_y = m.skyToPix(m.x0,m.y0)
    m.wcs.header.update('CRPIX1', m.wcs.header['CRPIX1'] - p_x)
    m.wcs.header.update('CRPIX2', m.wcs.header['CRPIX2'] - p_y)
    m.wcs.updateFromHeader()
    mNew = liteMapFromDataAndWCS(data, m.wcs)
    return mNew




def getCoordinateArrays( m ):
    """
    return two arrays containing the "sky" coordinates for all pixels in map m
    """
    ra = numpy.copy(m.data)
    dec = numpy.copy(m.data)
    for i in xrange(m.Ny):
        for j in xrange(m.Nx):
            ra[i,j], dec[i,j] = m.pixToSky(j,i)
    return ra, dec


def resampleFromHiResMap(highResMap, lowResTemp):
    """
    @brief resample a high resolution map into a template with lower resolution (bigger pixels)
    Basically takes average of the smaller pixel values that fall into a larger pixel
    Maps must be on the same patch of sky
    @return low res map 
    """
    #print highResMap.y0 - lowResTemp.y0
    assert(numpy.abs(highResMap.x0-lowResTemp.x0)/highResMap.x0 < 0.0001)
    assert(numpy.abs(highResMap.y0-lowResTemp.y0)/highResMap.x0 < 0.0001)
    #assert(highResMap.y0 == lowResTemp.y0)
    assert(highResMap.Nx> lowResTemp.Nx)
    assert(highResMap.Ny> lowResTemp.Ny)
    
    m = lowResTemp.copy()
    w = lowResTemp.copy()
    m.data[:] = 0.
    w.data[:] = 0.
    t0 = time.time()
    for i in xrange(highResMap.Ny):
        for j in xrange(highResMap.Nx):
            ra, dec = highResMap.pixToSky(j,i)
            ix,iy = m.skyToPix(ra,dec)
            m.data[iy,ix] += highResMap.data[i,j]
            w.data[iy,ix] += 1.0
    t1 = time.time()
    assert(numpy.all(w.data[:] >0.))
    m.data[:] /=w.data[:]
    return m 

def getRadiusAboutPoint( m, x0, y0, coordinateArrays = None):
    """
    return map of distances from point (x0,y0). Use precomputed coordinate Arrays (output of
    getCoordinateArrays) if provided.
    """
    if coordinateArrays == None:
        x, y = getCoordinateArrays(m)
    else:
        (x,y) = coordinateArrays
    dx  = (x  - x0 ) * numpy.cos(y*numpy.pi/180.)
    dy = (y - y0)
    theta = (dx**2 + dy**2)**0.5
    return theta

def binDataAroundPoint( m, x0, y0, bins, median = False ):
    """
    radially bin data in map m around (x0, y0) given bins (arcseconds)
    @return bin centers, average in bins, standard deviation in bins
    """
    x, y = getCoordinateArrays(m)
    dx  = (x  - x0 ) * numpy.cos(y*numpy.pi/180.)
    dy = (y - y0)
    theta = (dx**2 + dy**2)**0.5
    avg = []
    std = []
    cen = []
    for bin in bins:
        inds = numpy.where((theta >= float(bin[0])/3600.)*(theta < float(bin[1]/3600.)))
        if median:
            avg.append(numpy.median(m.data[inds]))
        else:
            avg.append(numpy.mean(m.data[inds]))
        std.append(numpy.std(m.data[inds])/numpy.sqrt(len(inds[0])))
        cen.append(numpy.mean(bin))

    return numpy.array(cen), numpy.array(avg), numpy.array(std)

def makeEmptyCEATemplate(raSizeDeg, decSizeDeg,meanRa = 180., meanDec = 0.,\
                      pixScaleXarcmin = 0.5, pixScaleYarcmin=0.5):
    assert(meanDec == 0.,'mean dec other than zero not implemented yet')

    
    cdelt1 = -pixScaleXarcmin/60.
    cdelt2 = pixScaleYarcmin/60.
    naxis1 = numpy.int(raSizeDeg/pixScaleXarcmin*60.+0.5)
    naxis2 = numpy.int(decSizeDeg/pixScaleYarcmin*60.+0.5)
    refPix1 = naxis1/2.
    refPix2 = naxis2/2.
    pv2_1 = 1.0
    cardList = pyfits.CardList()
    cardList.append(pyfits.Card('NAXIS', 2))
    cardList.append(pyfits.Card('NAXIS1', naxis1))
    cardList.append(pyfits.Card('NAXIS2', naxis2))
    cardList.append(pyfits.Card('CTYPE1', 'RA---CEA'))
    cardList.append(pyfits.Card('CTYPE2', 'DEC--CEA'))
    cardList.append(pyfits.Card('CRVAL1', meanRa))
    cardList.append(pyfits.Card('CRVAL2', meanDec))
    cardList.append(pyfits.Card('CRPIX1', refPix1+1))
    cardList.append(pyfits.Card('CRPIX2', refPix2+1))
    cardList.append(pyfits.Card('CDELT1', cdelt1))
    cardList.append(pyfits.Card('CDELT2', cdelt2))
    cardList.append(pyfits.Card('CUNIT1', 'DEG'))
    cardList.append(pyfits.Card('CUNIT2', 'DEG'))
    hh = pyfits.Header(cards=cardList)
    wcs = astLib.astWCS.WCS(hh, mode='pyfits')
    data = numpy.zeros([naxis2,naxis1])
    ltMap = liteMapFromDataAndWCS(data,wcs)
    
    return ltMap

def makeEmptyCEATemplateAdvanced(ra0, dec0, \
                                 ra1, dec1,\
                                 pixScaleXarcmin = 0.5, \
                                 pixScaleYarcmin= 0.5):

    """
    ALL RA DEC IN DEGREES
    """
    assert(ra0<ra1)
    assert(dec0<dec1)
    refDec = (dec0+dec1)/2.
    cosRefDec =  numpy.cos(refDec/180.*numpy.pi)
    raSizeDeg  = (ra1 - ra0)*cosRefDec
    decSizeDeg = (dec1-dec0)
    
    cdelt1 = -pixScaleXarcmin/(60.*cosRefDec)
    cdelt2 = pixScaleYarcmin/(60.*cosRefDec)
    naxis1 = numpy.int(raSizeDeg/pixScaleXarcmin*60.+0.5)
    naxis2 = numpy.int(decSizeDeg/pixScaleYarcmin*60.+0.5)
    refPix1 = numpy.int(-ra1/cdelt1+0.5)
    refPix2 = numpy.int(numpy.sin(-dec0*numpy.pi/180.)\
                        *180./numpy.pi/cdelt2/cosRefDec**2+0.5)
    pv2_1 = cosRefDec**2
    cardList = pyfits.CardList()
    cardList.append(pyfits.Card('NAXIS', 2))
    cardList.append(pyfits.Card('NAXIS1', naxis1))
    cardList.append(pyfits.Card('NAXIS2', naxis2))
    cardList.append(pyfits.Card('EXTEND', True))
    cardList.append(pyfits.Card('CTYPE1', 'RA---CEA'))
    cardList.append(pyfits.Card('CTYPE2', 'DEC--CEA'))
    cardList.append(pyfits.Card('CRVAL1', 0))
    cardList.append(pyfits.Card('CRVAL2', 0))
    cardList.append(pyfits.Card('CRPIX1', refPix1+1))
    cardList.append(pyfits.Card('CRPIX2', refPix2+1))
    cardList.append(pyfits.Card('CDELT1', cdelt1))
    cardList.append(pyfits.Card('CDELT2', cdelt2))
    cardList.append(pyfits.Card('CUNIT1', 'DEG'))
    cardList.append(pyfits.Card('CUNIT2', 'DEG'))
    cardList.append(pyfits.Card('PV2_1', pv2_1))
    cardList.append(pyfits.Card('EQUINOX',2000))
    cardList.append(pyfits.Card('PC1_1',1))
    cardList.append(pyfits.Card('PC1_2',0))
    cardList.append(pyfits.Card('PC2_1',0))
    cardList.append(pyfits.Card('PC2_2',1))
    
    hh = pyfits.Header(cards=cardList)
    wcs = astLib.astWCS.WCS(hh, mode='pyfits')
    data = numpy.zeros([naxis2,naxis1])
    ltMap = liteMapFromDataAndWCS(data,wcs)
    
    return ltMap
