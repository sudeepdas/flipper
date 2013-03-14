import numpy
import scipy
import sys, os
import pylab
from scipy.signal import convolve

def dpssFast(N,W,K):
    import dpss
    sines = numpy.zeros(N)
    taperOrder = K
    v,sig,totit,ifault = dpss.dpss(N,taperOrder,W,sines,sines.copy(),sines.copy(),sines.copy())
    sig += 1.
    return v, sig


def slepianTapers(Ny,Nx,Nres,Ntap):
    """
    @brief Generates 2D Slepian Taper
    @param Ny Number of Y pixels
    @param Nx Number of X pixels
    @param Nres The resolution paramter
    @param Ntap Number of 1-D to create 2-D tapers from.
    Total number of tapers produced will be NtapXNtap
    @return tapers(Ny,Nx,Ntap,Ntap),eigenvalues(Ntap,Ntap)
    """
    tapers = numpy.zeros([Ny,Nx,Ntap,Ntap])
    eigs = numpy.zeros([Ntap,Ntap])
    v0,sig0 = dpssFast(Nx,Nres*1.0/Nx,Ntap-1)
    v1,sig1 = dpssFast(Ny,Nres*1.0/Ny,Ntap-1)
    for i in xrange(Ntap):
        for j in xrange(Ntap):
            tapers[:,:,i,j] = numpy.outer(v1[:,i],v0[:,j])
            eigs[i,j] = sig0[i]*sig1[j]
    tapers *= numpy.sqrt(Ny*Nx*1.0)
    return tapers,eigs

def dpss0Fast(N,W):
    import dpss
    sines = numpy.arange(N)
    Ntapers = 0 #with this code all tapers can be generated very fast
    v,sig,totit,ifault = dpss.dpss(N,Ntapers,W,sines,sines.copy(),sines.copy(),sines.copy())
    return v

def dpss0(N,W):
    
    twoF = 2.0*W
    
    alpha = (N-1)/2.0
    m = numpy.arange(0,N)-alpha
    n = m[:,numpy.newaxis]
    k = m[numpy.newaxis,:]
    
    AF = twoF*numpy.sinc(twoF*(n-k))
    [lam,vec] = scipy.linalg.eig(AF)
    ind = numpy.argmax(abs(lam),axis=-1)
    
    w = abs(vec[:,ind])
    w = w / numpy.sqrt( numpy.sum(w*w) )
    return w

def slepianTaper00(Nx,Ny,Nres):
    try:
        w0 = dpss0Fast(Nx,Nres*1.0/Nx)
        w1 = dpss0Fast(Ny,Nres*1.0/Ny)
    except:
        print 'Switching to slower algorithm for Taper generation.'
        print 'Retry after running: '
        print 'f2py -c dpss.f -m dpss'
        print 'in the python/ directory in flipper'
        w0 = dpss0(Nx,Nres*1.0/Nx)
        w1 = dpss0(Ny,Nres*1.0/Ny)
    taper = numpy.outer(w1,w0)*numpy.sqrt(Nx*Ny*1.0)
    #print taper.shape
    return taper

def gaussKern(sigmaY,sigmaX,nSigma=5.0):
    """
    @ brief Returns a normalized 2D gauss kernel array for convolutions
    ^
    | Y
    |
    ------>
      X 
    """
    sizeY = int(nSigma*sigmaY)
    sizeX = int(nSigma*sigmaX)
    
    y, x = scipy.mgrid[-sizeY:sizeY+1, -sizeX:sizeX+1]
    g = scipy.exp(-(x**2/(2.*sigmaX**2)+y**2/(2.*sigmaY**2)))
    return g / g.sum()


def gaussianSmooth(im, sigmaY, sigmaX,nSigma=5.0) :
    """
    @brief smooths an image with a Gaussian given sigmas in x and y directions.
    @param sigmaY standard deviation of Gaussian in pixel units in the Y direction
    @param sigmaX standard deviation of Gaussian in pixel units in the X direction
    
    """
    g = gaussKern(sigmaY, sigmaX,nSigma=nSigma)
    improc = convolve(im,g, mode='same')
    return improc


def discKern(semiY,semiX):
    """
    @ brief Returns a disc kernel for convolution
    @ param semiY semiMajor axis (number of pixels)
    @ param semiX semiMajor axis (number of pixels)
    
    """
    print "in discKern: SemiY, SemiX", semiY, semiX
    assert(semiY > 0)
    assert(semiX > 0)
    sizeY = int(2*semiY+0.5)
    sizeX = int(2*semiX+0.5)
    print "sizes",sizeY,sizeX
    y, x = scipy.mgrid[-sizeY:sizeY+1, -sizeX:sizeX+1]
    g = numpy.zeros(y.shape)
    
    #print y, x
    nRand = 1000000
    rand = numpy.random.rand(nRand,2)
    for i in xrange(y.shape[0]*y.shape[1]):
        mtric = ((rand[:,0] - (y.flatten())[i] -0.5)/(semiY*1.0))**2\
                + ((rand[:,1] - (x.flatten())[i]-0.5)/(semiX*1.0))**2
        idd = numpy.where(mtric <1.)
        g[numpy.mod(i,y.shape[0]),i/y.shape[1]] = (numpy.array(idd).size)/(nRand*1.0)
        
    #print metric.shape,y.shape,x.shape
    #pylab.matshow(g)
    #pylab.colorbar()
    #pylab.show()
    #metric = (y**2/(semiY*1.0)**2+x**2/(semiX*1.0)**2)
    #idx = numpy.where(metric<1.)
    #print idx
    #g[idx] = 1.0
    #sys.exit()
    return g/g.sum()

def discSmooth(im,semiY,semiX,discKrn=None):
    """
    @ brief smooths with a disc
    @ param semiY semiMajor axis (number of pixels)
    @ param semiX semiMajor axis (number of pixels)
    
    """
    if discKrn != None:
        g = discKrn
    else:
        g = discKern(semiY, semiX)
    #pylab.matshow(g)
    #pylab.show()
    improc = convolve(im,g, mode='same')
    
    return improc


def bin( x, y, dx, xMin = None, xMax = None ):
    """
    Bin y by x in intervals of dx

    Params:
    xMin - Start binning here
    ymin - Stop binning here

    Returns three 1D numpy arrays:
    binned y
    bin centers
    bin counts
    bin standard deviations
    """

    x = numpy.array(x)
    y = numpy.array(y)

    if xMin == None:
        xMin = min(x)
    if xMax == None:
        xMax = max(x)
    xSpan = xMax - xMin
  
    #Initialize our bin arrays 
    binGinnings  = numpy.arange( xMin , xMax , dx, dtype = float )  # Get it Beginnings?!?
    nBin    = len(binGinnings)
    binVals = numpy.zeros(nBin, dtype = float)
    binCnts = numpy.zeros(nBin, dtype = float)
    binStds = numpy.zeros(nBin, dtype = float)
 
    for i in xrange(nBin):
        binBool = (x > binGinnings[i]) * (x < binGinnings[i] + dx)
        if not binBool.any():
            continue
        binCnts[i] = len(x[binBool])
        binVals[i] = y[binBool].sum()/binCnts[i]
        squares = (y[binBool]-binVals[i])**2
        binStds[i] = (squares.sum()**0.5)/binCnts[i]

    return binVals, binGinnings + dx/2., binStds, binCnts

def saveAndShow(fileName='.tmp.png'):
    """
    @brief In absence of GUI backend, save and show a pylab object
    """
    pylab.savefig(fileName)
    os.system('display %s &'%fileName)
    pylab.clf()
    pylab.close()

def bin( x, y, dx, xMin = None, xMax = None ):
    """
    Bin y by x in intervals of dx

    Params:
    xMin - Start binning here
    ymin - Stop binning here

    Returns three 1D numpy arrays:
    binned y
    bin centers
    bin counts
    bin standard deviations
    """

    x = numpy.array(x)
    y = numpy.array(y)

    if xMin == None:
        xMin = min(x)
    if xMax == None:
        xMax = max(x)
    xSpan = xMax - xMin
  
    #Initialize our bin arrays 
    binGinnings  = numpy.arange( xMin , xMax , dx, dtype = float )  # Get it Beginnings?!?
    nBin    = len(binGinnings)
    binVals = numpy.zeros(nBin, dtype = float)
    binCnts = numpy.zeros(nBin, dtype = float)
    binStds = numpy.zeros(nBin, dtype = float)
 
    for i in xrange(nBin):
        binBool = (x > binGinnings[i]) * (x < binGinnings[i] + dx)
        if not binBool.any():
            continue
        binCnts[i] = len(x[binBool])
        binVals[i] = y[binBool].sum()/binCnts[i]
        squares = (y[binBool]-binVals[i])**2
        binStds[i] = (squares.sum()**0.5)/binCnts[i]

    binCenter = binGinnings + dx/2.
    return binCenter, binVals, binStds, binCnts

def linearFit( x, y, err=None ):
    """
    Fit a linear function y as a function of x.  Optional parameter err is the
    vector of standard errors (in the y direction).

    Returns:
    solution - where y = solution[0] + solution[1]*x
    """
    x = numpy.copy(x)
    y = numpy.copy(y)
    N = len(x)
    A = numpy.ones((2, N), x.dtype)
    A[1] = x
    if err!=None: A /= err
    A = numpy.transpose(A)
    if err!=None: y /= err

    solution, residuals, rank, s = scipy.linalg.lstsq(A, y)
    return solution
