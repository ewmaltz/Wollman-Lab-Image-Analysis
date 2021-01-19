# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:46:30 2016

@author: Alonyan
"""
import numpy as np
import scipy.io as sio

      
def periodic_smooth_decomp(I: np.ndarray) -> (np.ndarray, np.ndarray):
    '''Performs periodic-smooth image decomposition
    Parameters
    ----------
    I : np.ndarray
        [M, N] image. will be coerced to a float.
    Returns
    -------
    P : np.ndarray
        [M, N] image, float. periodic portion.
    S : np.ndarray
        [M, N] image, float. smooth portion.
        
        Code from: https://github.com/jacobkimmel/ps_decomp
    '''
    def u2v(u: np.ndarray) -> np.ndarray:
        '''Converts the image `u` into the image `v`
        Parameters
        ----------
        u : np.ndarray
            [M, N] image
        Returns
            -------
        v : np.ndarray
            [M, N] image, zeroed expect for the outermost rows and cols
        '''
        v = np.zeros(u.shape, dtype=np.float64)

        v[0, :] = np.subtract(u[-1, :], u[0,  :], dtype=np.float64)
        v[-1,:] = np.subtract(u[0,  :], u[-1, :], dtype=np.float64)

        v[:,  0] += np.subtract(u[:, -1], u[:,  0], dtype=np.float64)
        v[:, -1] += np.subtract(u[:,  0], u[:, -1], dtype=np.float64)
        return v

    def v2s(v_hat: np.ndarray) -> np.ndarray:
        '''Computes the maximally smooth component of `u`, `s` from `v`
        s[q, r] = v[q, r] / (2*np.cos( (2*np.pi*q)/M )
            + 2*np.cos( (2*np.pi*r)/N ) - 4)
        Parameters
        ----------
        v_hat : np.ndarray
            [M, N] DFT of v
        '''
        M, N = v_hat.shape

        q = np.arange(M).reshape(M, 1).astype(v_hat.dtype)
        r = np.arange(N).reshape(1, N).astype(v_hat.dtype)

        den = (2*np.cos( np.divide((2*np.pi*q), M) ) \
             + 2*np.cos( np.divide((2*np.pi*r), N) ) - 4)
        s = np.divide(v_hat, den, out=np.zeros_like(v_hat), where=den!=0)
        s[0, 0] = 0
        return s

    u = I.astype(np.float64)
    v = u2v(u)
    v_fft = np.fft.fftn(v)
    s = v2s(v_fft)
    s_i = np.fft.ifftn(s)
    s_f = np.real(s_i)
    p = u - s_f # u = p + s
    return p, s_f




def perdecomp_3D(u):
    '''
    3D version of periodic plus smooth decomposition
    author: AOY
    '''
    u = u.astype(np.float64)
    s=np.zeros_like(u)
    nx=s.shape[0]
    ny=s.shape[1]
    nz=s.shape[2]
    
    b1 = u[-1,:,:] - u[0,:,:]
    b2 = u[:,-1,:] - u[:,0,:]
    b3 = u[:,:,-1] - u[:,:,0]
    
    s[0,:,:]  = - b1
    s[-1,:,:] =  b1 
  
    s[:,0,:]  = s[:,0,:] - b2
    s[:,-1,:] = s[:,-1,:] + b2
  
    s[:,:,0]  = s[:,:,0] - b3
    s[:,:,-1] = s[:,:,-1] + b3
    
    fft3_s=np.fft.fftn(s)
 
    cx = 2.0*np.pi/nx
    cy = 2.0*np.pi/ny
    cz = 2.0*np.pi/nz
    
    mat_x=np.expand_dims(np.concatenate((np.arange(np.round(nx/2)),np.arange(np.round(nx/2),0,-1))),(1,2))
    mat_y=np.expand_dims(np.concatenate((np.arange(np.round(ny/2)),np.arange(np.round(ny/2),0,-1))),(0,2))
    mat_z=np.expand_dims(np.concatenate((np.arange(np.round(nz/2)),np.arange(np.round(nz/2),0,-1))),(0,1))
       
    b=0.5/(3.0-np.cos(cx*mat_x)-np.cos(cy*mat_y)-np.cos(cz*mat_z))
    fft3_s = fft3_s * b
    fft3_s[0,0,0]=0
    s=np.real(np.fft.ifftn(fft3_s));
    return u-s, s










#%%
def trithresh(pix, nbins=256):
    imhist, edges = np.histogram(pix[:],nbins)
    centers = (edges[1:]+edges[:-1])/2
    
    a = centers[np.argmax(np.cumsum(imhist)/np.sum(imhist)>0.9999)] #brightest
    b = centers[np.argmax(imhist)] #most probable
    h = np.max(imhist) #response at most probable
    
    m = h/(b-a)
    
    x1=np.arange(0,a-b, 0.1)
    y1=np.interp(x1+b,centers,imhist)
    
    L = (m**2+1)*((y1-h)*(1/(m**2-1))-x1*m/(m**2-1))**2 #Distance between line m*x+b and curve y(x) maths!
    
    triThresh = b+x1[np.argmax(L)]
    return triThresh



def awt(I,nBands=None):
    """
    A description of the algorithm can be found in:
    J.-L. Starck, F. Murtagh, A. Bijaoui, "Image Processing and Data
    Analysis: The Multiscale Approach", Cambridge Press, Cambridge, 2000.
    
    W = AWT(I, nBands) computes the A Trou Wavelet decomposition of the
    image I up to nBands scale (inclusive). The default value is nBands =
    ceil(max(log2(N), log2(M))), where [N M] = size(I).
    
    Output:
    W contains the wavelet coefficients, an array of size N x M x nBands+1.
    The coefficients are organized as follows:
    W(:, :, 1:nBands) corresponds to the wavelet coefficients (also called
    detail images) at scale k = 1...nBands
    W(:, :, nBands+1) corresponds to the last approximation image A_K.
    
    
    Sylvain Berlemont, 2009
    Vectorized version - Alon Oyler-Yaniv, 2018
    python version - Alon Oyler-Yaniv, 2020
    """
    if np.ndim(I)==2:
        I = np.expand_dims(I,2)
        
    [N, M, L] = np.shape(I)
    
    K = np.ceil(max([np.log2(N), np.log2(M), np.log2(L)]))


    if nBands is None:
        nBands = K;
    assert(nBands<=K), "nBands must be <= %d" % K
    
    W = np.zeros((N, M, L, nBands + 1));
    
    lastA = I.astype('float')
    

    from numba import jit
    @jit(nopython = True)
    def convx(tmp,k1,k2): 
        I = 6*tmp[k2:-k2, : , :] + 4*(tmp[k2+k1:-k2+k1, :, :] + tmp[k2-k1:-k2-k1, :, :]) + tmp[2*k2:, :, :] + tmp[0:-2*k2, :, :]
        return I

    from numba import jit
    @jit(nopython = True)
    def convy(tmp,k1,k2): 
        I = 6*tmp[:,k2:-k2, :] + 4*(tmp[:,k2+k1:-k2+k1, :] + tmp[:,k2-k1:-k2-k1, :])+ tmp[:,2*k2:, :] + tmp[:,0:-2*k2, :]
        return I

    def convolve(I,k):
        k1 = 2**(k - 1);
        k2 = 2**k;
        
        tmp = np.pad(I, ((k2,k2),(0,0),(0,0)), 'edge')
        # Convolve the columns
        #I = 6*tmp[k2:-k2, : , :] + 4*(tmp[k2+k1:-k2+k1, :, :] + tmp[k2-k1:-k2-k1, :, :]) + tmp[2*k2:, :, :] + tmp[0:-2*k2, :, :]
        I = convx(tmp,k1,k2)
        tmp = np.pad(I * .0625, ((0,0),(k2,k2),(0,0)), 'edge');
        #I = 6*tmp[:,k2:-k2, :] + 4*(tmp[:,k2+k1:-k2+k1, :] + tmp[:,k2-k1:-k2-k1, :]) + tmp[:,2*k2:, :] + tmp[:,0:-2*k2, :]
        I = convy(tmp,k1,k2)
        
        return I * .0625
    
    for k in np.arange(1, nBands+1):
        newA = convolve(lastA, k)
        W[:, :, :, k-1] = lastA - newA;
        lastA = newA;
    
    W[:, :, :, nBands] = lastA;
    
    return np.squeeze(W)

'''
Python implementation of the imimposemin function in MATLAB.
Reference: https://www.mathworks.com/help/images/ref/imimposemin.html
'''


def imimposemin(I, BW, conn=None, max_value=255):
    import numpy as np
    import math
    from skimage.morphology import reconstruction, square, disk, cube

    if not I.ndim in (2, 3):
        raise Exception("'I' must be a 2-D or 3D array.")

    if BW.shape != I.shape:
        raise Exception("'I' and 'BW' must have the same shape.")

    if BW.dtype is not bool:
        BW = BW != 0

    # set default connectivity depending on whether the image is 2-D or 3-D
    if conn == None:
        if I.ndim == 3:
            conn = 26
        else:
            conn = 8
    else:
        if conn in (4, 8) and I.ndim == 3:
            raise Exception("'conn' is invalid for a 3-D image.")
        elif conn in (6, 18, 26) and I.ndim == 2:
            raise Exception("'conn' is invalid for a 2-D image.")

    # create structuring element depending on connectivity
    if conn == 4:
        selem = disk(1)
    elif conn == 8:
        selem = square(3)
    elif conn == 6:
        selem = ball(1)
    elif conn == 18:
        selem = ball(1)
        selem[:, 1, :] = 1
        selem[:, :, 1] = 1
        selem[1] = 1
    elif conn == 26:
        selem = cube(3)

    fm = I.astype(float)

    try:
        fm[BW]                 = -math.inf
        fm[np.logical_not(BW)] = math.inf
    except:
        fm[BW]                 = -float("inf")
        fm[np.logical_not(BW)] = float("inf")

    if I.dtype == float:
        I_range = np.amax(I) - np.amin(I)

        if I_range == 0:
            h = 0.1
        else:
            h = I_range*0.001
    else:
        h = 1

    fp1 = I + h

    g = np.minimum(fp1, fm)

    # perform reconstruction and get the image complement of the result
    if I.dtype == float:
        J = reconstruction(1 - fm, 1 - g, selem=selem)
        J = 1 - J
    else:
        J = reconstruction(255 - fm, 255 - g, method='dilation', selem=selem)
        J = 255 - J

    try:
        J[BW] = -math.inf
    except:
        J[BW] = -float("inf")

    return J


from numba import jit
@jit(nopython = True)
def DownScale(imgin): #use 2x downscaling for scrol speed   
        #imgout = trans.downscale_local_mean(imgin,(Sc, Sc))
    imgout = (imgin[0::2,0::2]+imgin[1::2,0::2]+imgin[0::2,1::2]+imgin[1::2,1::2])/4
    return imgout




    



class segmentation:
    """
    class for all segmentation functions
    All functions accept an img and any arguments they need and return a labeled matrix.
    """
    def segtype_to_segfun(segment_type):
        if segment_type=='watershed':
            seg_fun=segmentation._segment_nuclei_watershed
        elif segment_type=='cellpose_nuclei':
            seg_fun=segmentation._segment_nuclei_cellpose
        elif segment_type=='evan':
            seg_fun=segmentation._segment_evan
        return seg_fun
            
    def _segment_nuclei_watershed(img, voronoi=None,cellsize=5, hThresh=0.001):
        from skimage import filters, measure
        from skimage.util import invert
        from skimage.morphology import watershed  
        from oyLabCode.Processing.improcutils import trithresh, awt, imimposemin
        from skimage.feature import peak_local_max
        from scipy.ndimage.morphology import distance_transform_edt 
        from skimage.morphology import erosion, dilation, opening, closing, h_maxima, disk
                
        #wavelet transform and SAR
        W = awt(img, 9)
        img = np.sum(W[:,:,1:8],axis=2)

        if voronoi is None:
            #Smoothen
            voronoi = {}
            imgSmooth = filters.gaussian(img,sigma=cellsize)
            img_hmax = h_maxima(imgSmooth,hThresh) #threshold
            RegionMax = peak_local_max(img_hmax,footprint=np.ones((30, 30)), indices=False).astype('int')
            se = disk(cellsize)
            RegionMax = closing(RegionMax,se)
            imgBW = dilation(RegionMax,se);
            dt = distance_transform_edt(1-imgBW)
            DL = watershed(dt,watershed_line = 1)
            RegionBounds = DL == 0 #the region bounds are==0 voronoi cells
            voronoi['imgBW'] = imgBW
            voronoi['RegionBounds'] = RegionBounds


        imgBW = voronoi['imgBW'];
        RegionBounds = voronoi['RegionBounds'];

        #gradient magnitude
        GMimg = filters.sobel(filters.gaussian(img, sigma=cellsize))
        GMimg[np.logical_or(imgBW,RegionBounds)]=0
        L = watershed(GMimg, markers=measure.label(imgBW), watershed_line = 1)

        #import matplotlib.pyplot as plt
        #fig = plt.figure(figsize=(12,15))
        #plt.imshow(L, cmap="gray")
        
        #We use regionprops 
        props = measure.regionprops(L)
        Areas = np.array([r.area for r in props])
        
        #remove BG region and non-cells
        Areas>10000;
        BG = [i for i, val in enumerate(Areas>10000) if val] 

        if any(BG):
            for i in np.arange(len(BG)):
                L[L==BG[i]+1]=0

        L = measure.label(L);
        return L
    
    def _segment_nuclei_cellpose(img, diameter=50, scale=0.5,**kwargs):
        from cellpose import models, io
        from cv2 import resize, INTER_NEAREST, INTER_AREA
        from numpy import squeeze
        from skimage.transform import rescale

        model = models.Cellpose(gpu=False, model_type='nuclei')
        img = np.squeeze(img)
        assert(img.ndim==2), "_segment_nuclei_cellpose accepts 2D images" 
   
        masks, _, _, _ = model.eval([rescale(img,scale)], diameter=diameter*scale, channels=[[0,0]],**kwargs)        

        dim = (img.shape[1], img.shape[0])
        # resize masks to original image size using nearest neighbor interpolation to preserve masks 
        L = resize(masks[0], dim, interpolation = INTER_NEAREST)
 
        return L

    def _segment_cytoplasm_cellpose(imgNuc, imgCyto, diameter=75, scale=0.5,**kwargs):
        from cellpose import models, io
        from cv2 import resize, INTER_NEAREST, INTER_AREA
        from numpy import squeeze
        from skimage.transform import rescale

        model = models.Cellpose(gpu=False, model_type='cyto')
        
        imgNucCyto = rescale(np.concatenate((np.expand_dims(imgNuc,2), np.expand_dims(imgCyto,2)), axis=2),(scale, scale, 1))
        
        masks, _, _, _ = model.eval([imgNucCyto], diameter=diameter, channels=[[2,1]],**kwargs) 

        dim = (imgNuc.shape[1], imgNuc.shape[0])
        # resize masks to original image size using nearest neighbor interpolation to preserve masks 
        return resize(masks[0], dim, interpolation = INTER_NEAREST)
    
    
    def _segment_nuccyto_cellpose(imgNuc, imgCyto, diameter_nuc=50,diameter_cyto=75, scale=0.5,**kwargs):
        Lnuc = segmentation._segment_nuclei_cellpose(imgNuc, diameter=diameter_nuc, scale=scale,**kwargs)
        Lcyto = segmentation._segment_cytoplasm_cellpose(imgNuc, imgCyto, diameter=diameter_cyto, scale=scale,**kwargs)
        
        Lcyto_new = np.zeros_like(Lcyto)

        for i in np.arange(1,np.max(Lnuc)+1):
            ind_in_cyto = np.median(Lcyto[Lnuc==i])
            if ind_in_cyto:
                Lcyto_new[Lcyto==ind_in_cyto]=i
            else:
                Lnuc[Lnuc==i]=0
                
        
        #Lcyto_new = Lcyto_new-Lnuc;
        return Lnuc, Lcyto_new
    
    def _segment_evan(img, params=None, **kwargs):
        
        from oyLabCode.Processing.generalutils import label_wvt, watershed_wvt
        from skimage import measure 
        ori_shape = np.shape(img)
        img = img.reshape(img.shape[0], img.shape[1], -1) 
        wvt_nuclear = label_wvt(img)
        L = watershed_wvt(img,wvt_nuclear)
        
        #We use regionprops 
        props = measure.regionprops(L)
        Areas = np.array([r.area for r in props])
        
        #remove BG region and non-cells
        Areas>10000;
        BG = [i for i, val in enumerate(Areas>10000) if val] 

        if any(BG):
            for i in np.arange(len(BG)):
                L[L==BG[i]+1]=0

        L = measure.label(L);
        L = L.reshape(ori_shape)
        return L
    
    def test_segmentation_params(img,segfun=None, segment_type='watershed'):
        import numpy as np
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from matplotlib.colors import ListedColormap

        if segfun==None:
            segfun = segmentation.segtype_to_segfun(segment_type)
            print('\nusing ' + segfun.__name__ )
        
        if img.ndim==3:
            img = np.squeeze(img[0,:,:])

        nargs = segfun.__code__.co_argcount
        args = [segfun.__code__.co_varnames[i] for i in range(1, nargs)]
        defaults = list(segfun.__defaults__)
        input_dict = {args[i]: defaults[i] for i in range(0, nargs-1)} 

        L = segfun(img)

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.1)

        ax.imshow(img, cmap='gray',vmin=np.percentile(img, 10), vmax=np.percentile(img, 99))
        cmap = ListedColormap( np.random.rand (256,3))
        cmap.colors[0,:]=[0,0,0]
        l=ax.imshow(L, alpha=0.5, cmap=cmap)

        def clean_dict(d):
            result = {}
            for key, value in d.items():
                if value=='None':
                    value = None
                else:
                    value = float(value)
                result[key] = value
            return result

        def update_mask(b):
            print('calculating with new parameters')
            #new_input_dict = {args[i]: eval('text_box_' + str(i)+ '.value') for i in range(0, nargs-1)} 
            new_input_dict = {args[i]: tblist[i].value for i in range(0, nargs-1)} 
            new_input_dict = clean_dict(new_input_dict)
            L = segfun(img,**new_input_dict)
            l.set_data(L)
            l.set_alpha=0.5
            plt.draw()
            plt.show()
            print('Done!')

        num=0
        tblist = []
        for arg, de in input_dict.items():  
            exec('text_box_' + str(num) + '=widgets.Text(value=str(de), description=str(arg))')
            tblist.append(eval('text_box_' + str(num)))
            num+=1
        exButton = widgets.Button(description='Segment cells!')
        exButton.on_click(update_mask)
        
        box = widgets.HBox([widgets.VBox(tblist), exButton])
        return box


##Filters

def _prepare_frequency_map(pix):
    nx=pix.shape[0]
    ny=pix.shape[1]

    cx = 2.0*np.pi/nx
    cy = 2.0*np.pi/ny

    mat_x=np.expand_dims(np.concatenate((np.arange(np.round(nx/2),0,-1),np.arange(np.round(nx/2)))),1)
    mat_y=np.expand_dims(np.concatenate((np.arange(np.round(ny/2),0,-1),np.arange(np.round(ny/2)))),0)
    f = np.sqrt(cx*mat_x**2 + cy*mat_y**2)
    return f



def log_filter(p, sigma=5):
    #p,s =  periodic_smooth_decomp(p)
    
    img_fft = np.fft.fft2(p);
    img_fft = np.fft.fftshift(img_fft)
    
    f = _prepare_frequency_map(p)
    kernel = np.exp(-(sigma*sigma*(f**2))/(2*(2*np.pi**2)**2))*(f**2)
    img_ridges = np.real(np.fft.ifft2(np.fft.ifftshift(img_fft*kernel)))
    return img_ridges

def gaussian_filter(p, sigma=10):
    #p,s =  periodic_smooth_decomp(p)
    
    img_fft = np.fft.fft2(p);
    img_fft = np.fft.fftshift(img_fft)
      
    f = _prepare_frequency_map(p)

    kernel = np.exp(-(sigma*sigma*(f**2))/(2*(2*np.pi**2)**2))
    img_smooth = np.real(np.fft.ifft2(np.fft.ifftshift(img_fft*kernel)))
    return img_smooth