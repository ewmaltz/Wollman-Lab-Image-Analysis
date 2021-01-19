"""
Created on Thu Jul 21 15:46:30 2016

@author: Alonyan
"""
import numpy as np
import scipy.io as sio

def num2str(num, precision): 
    return "%0.*f" % (precision, num)
    
def colorcode(datax, datay):
    from scipy import interpolate
    import numpy as np
    H, xedges, yedges = np.histogram2d(datax,datay, bins=30)
    xedges = (xedges[:-1]+xedges[1:])/2
    yedges = (yedges[:-1]+yedges[1:])/2
    f = interpolate.RectBivariateSpline(xedges,yedges , H)
    
    z = np.array([])
    for i in datax.index:
        z = np.append(z,f(datax[i],datay[i]))
    #z=(z-min(z))/(max(z)-min(z))   
    z[z<0] = 0
    idx = z.argsort()
    return z, idx


class kmeans:        
    def __init__(self, X, K):
        # Initialize to K random centers
        oldmu = X.sample(K).values#np.random.sample(X, K)
        mu = X.sample(K).values#np.random.sample(X, K)
        while not _has_converged(mu, oldmu):
            oldmu = mu
            # Assign all points in X to clusters
            clusters = _cluster_points(X, mu)
            # Reevaluate centers
            mu = _reevaluate_centers(oldmu, clusters)
        self.mu = mu
        self.clusters = clusters
        #return(mu, clusters)
        
    def _cluster_points(X, mu):
        clusters  = {}
        for x in X:
            bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                        for i in enumerate(mu)], key=lambda t:t[1])[0]
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
        return clusters

    def _reevaluate_centers(mu, clusters):
        newmu = []
        keys = sorted(clusters.keys())
        for k in keys:
            newmu.append(np.mean(clusters[k], axis = 0))
        return newmu

    def _has_converged(mu, oldmu):
        return (set(mu) == set(oldmu))


      
def regionprops_to_df(im_props):
    """
    Read content of all attributes for every item in a list
    output by skimage.measure.regionprops
    """
    import pandas as pd

    def scalar_attributes_list(im_props):
        """
        Makes list of all scalar, non-dunder, non-hidden
        attributes of skimage.measure.regionprops object
        """
        attributes_list = []

        for i, test_attribute in enumerate(dir(im_props[0])):

            #Attribute should not start with _ and cannot return an array
            #does not yet return tuples
            if test_attribute[:1] != '_' and not\
                    isinstance(getattr(im_props[0], test_attribute), np.ndarray):                
                attributes_list += [test_attribute]

        return attributes_list


    attributes_list = scalar_attributes_list(im_props)

    # Initialise list of lists for parsed data
    parsed_data = []

    # Put data from im_props into list of lists
    for i, _ in enumerate(im_props):
        parsed_data += [[]]
        
        for j in range(len(attributes_list)):
            parsed_data[i] += [getattr(im_props[i], attributes_list[j])]

    # Return as a Pandas DataFrame
    return pd.DataFrame(parsed_data, columns=attributes_list)
'''
Start of Evan's general scripts for segmentation
'''

from skimage.filters import threshold_local
from skimage.morphology import label, watershed, binary_opening,binary_closing, binary_erosion, disk, remove_small_objects,convex_hull
from skimage.morphology import binary_dilation as bin_dil
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.measure import find_contours
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries, mark_boundaries
from skimage.color import gray2rgb
from sklearn.preprocessing import minmax_scale

from skimage import filters
from skimage import morphology
from scipy import ndimage

import pywt

def clipping(im, val):

    im_temp = im.copy()

    if val != 0:
        im_temp[im > val] = val

    return im_temp


def background(im, val):

    from skimage import filters

    im_temp = im.copy()

    if val != 0:
        im_temp = im_temp - filters.gaussian(im_temp, val)

    return im_temp

def blur(im, val):

    if val != 0:

        if val <= 5:
            im = filters.gaussian(im, val)

        else:

            im = filters.gaussian(im, (val / 2))
            im = filters.gaussian(im, (val / 2))
            im = filters.gaussian(im, (val / 2))

    im -= np.min(im.flatten())
    im /= np.max(im.flatten())

    return im


def threshold(im, val):

    im_bin = im > val

    return im_bin


def object_filter(im_bin, val):

    im_bin = morphology.remove_small_objects(im_bin, val)

    return im_bin


def cell_centers(im, im_bin, val):

    d_mat = ndimage.distance_transform_edt(im_bin)
    d_mat /= np.max(d_mat.flatten())

    im_cent = (1 - val) * im + val * d_mat
    im_cent[np.logical_not(im_bin)] = 0

    return [im_cent, d_mat]

def expand_im(im, wsize):

    vinds = np.arange(im.shape[0], im.shape[0] - wsize + 1, -1) - 1
    hinds = np.arange(im.shape[1], im.shape[1] - wsize + 1, -1) - 1
    rev_inds = np.arange(wsize + 1, 0, -1)

    vinds = vinds.astype(int)
    hinds = hinds.astype(int)
    rev_inds = rev_inds.astype(int)

    conv_im_temp = np.vstack((im[rev_inds, :], im, im[vinds, :]))
    conv_im = np.hstack((conv_im_temp[:, rev_inds], conv_im_temp, conv_im_temp[:, hinds]))

    return conv_im

def im_probs(im, clf, wsize, stride):

    conv_im = expand_im(im, wsize)
    X_pred = classifyim.classify_im(conv_im, wsize, stride, im.shape[0], im.shape[1])

    y_prob = clf.predict_proba(X_pred)
    y_prob = y_prob[:, 1]

    return y_prob.reshape(im.shape)

def open_close(im, val):

    val = int(val)

    if 4 > val > 0:

        k = morphology.octagon(val, val)

        im = filters.gaussian(im, val)

        im = morphology.erosion(im, k)

        im = morphology.dilation(im, k)

    if 8 > val >= 4:
        k = morphology.octagon(val//2 + 1, val//2 + 1)

        im = filters.gaussian(im, val)
        im = filters.gaussian(im, val)

        im = morphology.erosion(im, k)
        im = morphology.erosion(im, k)

        im = morphology.dilation(im, k)
        im = morphology.dilation(im, k)

    if val >= 8:
        k = morphology.octagon(val // 4 + 1, val // 4 + 1)

        im = filters.gaussian(im, val)
        im = filters.gaussian(im, val)
        im = filters.gaussian(im, val)
        im = filters.gaussian(im, val)

        im = morphology.erosion(im, k)
        im = morphology.erosion(im, k)
        im = morphology.erosion(im, k)
        im = morphology.erosion(im, k)

        im = morphology.dilation(im, k)
        im = morphology.dilation(im, k)
        im = morphology.dilation(im, k)
        im = morphology.dilation(im, k)

    return im

def fg_markers(im_cent, im_bin, val, edges):

    local_maxi = peak_local_max(im_cent, indices=False, min_distance=int(val), labels=im_bin, exclude_border=int(edges))
    k = morphology.octagon(2, 2)

    local_maxi = morphology.dilation(local_maxi, selem=k)
    markers = ndimage.label(local_maxi)[0]
    markers[local_maxi] += 1

    return markers


def sobel_edges(im, val):

    if val != 0:
        if val <= 5:
            im = filters.gaussian(im, val)

        else:

            im = filters.gaussian(im, (val / 2))
            im = filters.gaussian(im, (val / 2))
            im = filters.gaussian(im, (val / 2))

    im = filters.sobel(im) + 1
    im /= np.max(im.flatten())

    return im


def custom_watershed(markers, im_bin, im_edge, d_mat, val, edges):

    k = morphology.octagon(2, 2)

    im_bin = morphology.binary_dilation(im_bin, selem=k)
    im_bin = morphology.binary_dilation(im_bin, selem=k)
    im_bin = morphology.binary_dilation(im_bin, selem=k)

    markers_temp = markers + np.logical_not(im_bin)
    shed_im = (1 - val) * im_edge - val * d_mat

    labels = morphology.watershed(image=shed_im, markers=markers_temp)
    labels -= 1

    if edges == 1:
        edge_vec = np.hstack((labels[:, 0].flatten(), labels[:, -1].flatten(), labels[0, :].flatten(),
                              labels[-1, :].flatten()))
        edge_val = np.unique(edge_vec)
        for val in edge_val:
            if not val == 0:
                labels[labels == val] = 0

    return labels

# for visualization
def create_contours(img, coords):
    img_contour = np.zeros_like(img)
    img_contour[coords[:,0],coords[:,1]] = 1
    contour_coords = find_contours(img_contour, 0)[0].astype(np.int)
    img_contour = np.zeros_like(img)
    img_contour[contour_coords[:,0], contour_coords[:,1]] = 1
    return img_contour


# for visualization
def create_single_cell_nuc_cyto_visualization(well_arr_cyto, frame, nuc_coords, cyto_coords):
    rgb_img = np.stack([np.zeros_like(well_arr_cyto[:,:,frame]), 
                                     well_arr_cyto[:,:,frame]/well_arr_cyto[:,:,frame].max(), 
                                     np.zeros_like(well_arr_cyto[:,:,frame])], axis=-1)
    nuc_contour = create_contours(well_arr_cyto[:,:,frame], nuc_coords)
    cyto_contour = create_contours(well_arr_cyto[:,:,frame], cyto_coords)
    rgb_img[:,:,0]+=nuc_contour
    rgb_img[:,:,2]+=cyto_contour
    return rgb_img


# for visualization
def create_nuc_cyto_visualization(well_arr, nuc_label_img, all_cyto_coords):
    cell_contours = []  # k, h, w, c
    for k in range(well_arr.shape[-1]):
        cell_contour = mark_boundaries(well_arr[:,:,k], nuc_label_img[:,:,k])
        blank = np.zeros_like(well_arr[:,:,k])
        blank[all_cyto_coords[k][:,0], all_cyto_coords[k][:,1]] = 1
        cyto_bounds = find_boundaries(blank)
        cyto_bound_coords = np.squeeze(list(zip(np.where(cyto_bounds)))).T
        cell_contour[cyto_bound_coords[:,0], cyto_bound_coords[:,1]] = [1,0,1]
        cell_contours.append(cell_contour)
    return np.stack(cell_contours)


def wavelet_transform(img, keep_list=[3,4,5,6], wavelet='db9'):
    coeffs = pywt.wavedec2(img,wavelet)
    for i in range(1,len(coeffs)+1):
        if i in keep_list:
            continue
        coeffs[-i] = tuple([np.zeros_like(v) for v in coeffs[-i]])
    Result = pywt.waverec2(coeffs,wavelet)
    return Result


def wavelet_segment(x, keep=[3,4,5], wv='coif11', disk_size=6):
    y = wavelet_transform(x, keep_list=keep, wavelet=wv)
    return binary_erosion(y>threshold_local(y, block_size=311),disk(disk_size))


def label_wvt(well_arr):
    """
    generating labels
    returns list of labeled_wvt from image
    """
    labeled_wvts = []
    for i in range(well_arr.shape[2]):
        labeled_wvt = label(remove_small_objects(wavelet_segment(well_arr[:,:,i]), min_size=100))
        labeled_wvts.append(labeled_wvt)
    labeled_wvts = np.stack(labeled_wvts)
    labeled_wvts = np.swapaxes(labeled_wvts.T, 0,1)
    return labeled_wvts


def watershed_wvt(well_arr,wvt):
    watershed_wvts = []
    for k in range(well_arr.shape[2]):
        img = gaussian_filter(well_arr[:,:,k],9)
        img_mask = wvt[:,:,k]
        _dte = ndi.distance_transform_edt(img_mask)
        _peaks = peak_local_max(img, indices=False, min_distance=5)
        watershed_img = watershed(-_dte,label(_peaks), mask=img_mask)
        watershed_wvts.append(label(watershed_img))  # must relabel the image so that arange(1, watershed_img.max()).shape[0]==watershed_img.max()
    watershed_wvts = np.stack(watershed_wvts)
    watershed_wvts= np.swapaxes(watershed_wvts.T, 0,1)
    return watershed_wvts


def features(watershed_wvt_img, well_img):
    """
    extract features from labeled objects.
    returns a list of properties for each object in each image in well.
    """
    props = []
    for k in range(watershed_wvt_img.shape[2]):
        img_props = regionprops(watershed_wvt_img[:,:,k],intensity_image=well_img[:,:,k])
        props.append(img_props)
    return props


def ninety_percentile(well_arr, cell_props, watershed_wvt, nuc_cyto_ring=False, report=False):
    """
    Most likely can be rewritten for speed...
    """
    nuc_frames = []
    cyto_frames = []
    nuc_var_frames = []
    cyto_var_frames = []
    frames = []
    for k in range(well_arr.shape[2]):
        nucs = []
        cytos = []
        nuc_vars = []
        cyto_vars = []
        percent90s = []
        for i in range(len(cell_props[k])):
            labeled_object = cell_props[k][i]
            if nuc_cyto_ring:
                # cyto_feature uses nuclear labels and cyto image
                nuc_percent90 = np.percentile(well_arr[labeled_object.coords[:,0],labeled_object.coords[:,1], k], 90)
                selected_label = (watershed_wvt[:,:,k]==labeled_object.label).astype(np.int)
                # uncertain how many dilations to do, covert lab does 8, I'll do 8
                cyto_outer = bin_dil(bin_dil(bin_dil(bin_dil(bin_dil(bin_dil(bin_dil(bin_dil(bin_dil(bin_dil(selected_label))))))))))
                cyto_ring = cyto_outer ^ bin_dil(bin_dil(selected_label))  # using the bitwise x_or operator to subtract 2 ring+nuc from 10 rings+nuc = 8 pixel rings 
                cyto_percent90 = np.percentile(well_arr[np.where(cyto_ring)][:,k], 90)
                percent90 = nuc_percent90/cyto_percent90
                if report:
                    nucs.append(labeled_object.coords)  # nuclear coords for each cell
                    cytos.append(np.squeeze(list(zip(np.where(cyto_ring)))).T)  # cytoplasmic coords for each cell
                    nuc_vars.append(np.var(well_arr[labeled_object.coords[:,0],labeled_object.coords[:,1], k]))
                    cyto_vars.append(np.var(well_arr[np.where(cyto_ring)][:,k]))
            else:
                percent90 = np.percentile(well_arr[labeled_object.coords[:,0],labeled_object.coords[:,1], k], 90)
            percent90s.append(percent90)
        frames.append(percent90s)
        if report:
            nuc_frames.append(nucs)
            cyto_frames.append(cytos)
            nuc_var_frames.append(nuc_vars)
            cyto_var_frames.append(cyto_vars)

    if report:
        return frames, nuc_frames, cyto_frames, nuc_var_frames, cyto_var_frames
    else:
        return frames, [], [], [], []


def cell_frames(watershed_wvt,well_arr,cell_props, nuc_cyto_ring=False, report=False):
    """
    create appropriately sized dataframe and fill it with features
    returns a filled dataframe of properties/features for each object in each image in the given well
    """
    cellFrames = []
    nineties, nuc_coords, cyto_coords, nuc_vars, cyto_vars = ninety_percentile(well_arr, cell_props, watershed_wvt, nuc_cyto_ring, report=report)
    for k in range(watershed_wvt.shape[2]):
        cell_frame = pd.DataFrame(index=range(1, watershed_wvt[:,:,k].max()+1), columns=['y','x','Filled_Area','90_intensity','frame'])
        cellFrames.append(cell_frame)
    for k in range(len(cellFrames)):
        cellFrames[k]['y'] = [prop.centroid[0] for prop in cell_props[k]]
        cellFrames[k]['x'] = [prop.centroid[1] for prop in cell_props[k]]
        cellFrames[k].Filled_Area = [prop.area for prop in cell_props[k]]
        cellFrames[k]['90_intensity'] = nineties[k]
        cellFrames[k]['frame'] = k
        if report:
            cellFrames[k]['nuc coords'] = nuc_coords[k]
            cellFrames[k]['cyto coords'] = cyto_coords[k]
            cellFrames[k]['nuc variability'] = nuc_vars[k]
            cellFrames[k]['cyto variability'] = cyto_vars[k]
        
    return cellFrames


def get_particles(df, min_track_length=35):
    """
    Filters trackpy dataframe to get particles with specified track length
    """
    particle_counts = df.groupby(df.particle,as_index=False).size()
    particle_df = df.set_index(df.particle).loc[particle_counts.loc[particle_counts>min_track_length].index]
    return particle_df


def intensities_df(df):
    """
    Not necessary for the pipeline
    df: Particle dataframe
    returns: intensities
    """
    intensities = pd.DataFrame(index=df.particle.drop_duplicates().values, columns=df.frame.drop_duplicates().values)
    
    df = df.set_index(df.frame)
    for p in df.particle.drop_duplicates().values:
        frame_intensities = df.loc[df.particle==p,['particle','90_intensity']]
        #particle column not necessary
        #probably don't also need the new dataframe 'frame_intensities'
        for f in df.frame.drop_duplicates().values:
            if f not in frame_intensities.index:
                intensities.loc[p,f] = None
            else:
                intensities.loc[p,f] = frame_intensities.loc[f, '90_intensity']
    for col in intensities:
        intensities[col] = pd.to_numeric(intensities[col], errors='coerce')
    return intensities


def track_cyto_intensities(position, save_path, cytoplasmic_channels, metadata, keep_frames='all', nuclear_channel='DeepBlue', 
                           memory=8, stub_length=5, min_track_length=35, nuc_cyto_ring=False, report=False):
    """
    Trackpy method
    calculates the trajectories of evry object from every image in every well for arbitrary channels.
    uses the intensities from cyto channel(s) to track reliably
    nuclear_channel: Can only be one channel, usually DeepBlue
    cytoplasmic channels: {marker: channel} e.g. {'Virus':'Cyan'}
    keep_frames: Frame indices that are good, can exclude unusable, e.g. blurry, frames (numpy array, both 'all' and None give all frames). 
                 Using None to mean all is a bit confusing but it is the straightforward interpretation when keep_frames = [1,2,3,...]
                 which would indeed keep frames 1,2,3,...
    Also saves intensities of tracks for arbitrary channels in a new dataframe
    """
    #get images from metadata
    if keep_frames ==  'all':
        keep_frames = None  # when keep_frames==None, it will keep all frames because numpy treats empy axis as all
    well_arr_nuclear = metadata.stkread(Channel=nuclear_channel, Position=position)[:,:,keep_frames]
    well_arr_nuclear = well_arr_nuclear.reshape(well_arr_nuclear.shape[0], well_arr_nuclear.shape[1], -1)  # guarantees shape = height, width, channels (must be 3D)
    well_arr_cytos = {}
    for marker, channel in cytoplasmic_channels.items():
        well_arr_cyto = metadata.stkread(Channel=channel, Position=position)[:,:,keep_frames]
        well_arr_cytos[marker] = well_arr_nuclear.reshape(well_arr_cyto.shape[0], well_arr_cyto.shape[1], -1)  # guarantees shape = height, width, channels (must be 3D)
    
    #segement the whole well using nuclear channel
    wvt_nuclear = label_wvt(well_arr_nuclear)
    watershed_wvt_nuclear = watershed_wvt(well_arr_nuclear,wvt_nuclear)

    #extract cyto intensities via regionprops in features function
    cyto_intensities = {}
    for marker, well_arr_cyto in well_arr_cytos.items():
        cyto_feature = features(watershed_wvt_nuclear, well_arr_cyto)

        #dataframe with viral/death intensities
        cellFrames_cyto = pd.concat(cell_frames(watershed_wvt_nuclear, well_arr_cyto, cyto_feature, nuc_cyto_ring, report=report))
    
        #correct for artifacts and false areas
        cellFrames_cyto = cellFrames_cyto.loc[(cellFrames_cyto.Filled_Area>=100)]
    
        #track cells using cyto features
        traj_cyto = tp.link_df(cellFrames_cyto, 50, memory=memory)
        traj_cyto = tp.filter_stubs(traj_cyto, stub_length)
        
        #filter to include only particles of trajectory => 35
        traj_cyto = get_particles(traj_cyto, min_track_length=min_track_length)

        #save the trajectory + intensities dataframe
        if save_path:
            traj_cyto.to_pickle(save_path+'/Track_'+marker+'_'+ position+'.pkl')

max_pools = 48

def multiprocess_segmentation(positions, save_path, nuclear_channel, cytoplasmic_channels, metadata, memory, stub_length, min_track_length, nuc_cyto_ring, keep_frames='all', pools=max_pools, report=False):
    """
    Save path will become a repository for all results, trajectories
    """
    if pools>max_pools:
        print('Sorry, maximum number of pools is currently: ', max_pools)
        print('Must go to segmentation.py to increase max_pools.')

    with multiprocessing.Pool(pools) as ppool:
        ppool.map(partial(track_cyto_intensities, save_path=save_path, nuclear_channel=nuclear_channel, cytoplasmic_channels=cytoplasmic_channels, 
                          metadata=metadata, keep_frames=keep_frames, memory=memory, stub_length=stub_length, min_track_length=min_track_length,
                          nuc_cyto_ring=nuc_cyto_ring, report=report), 
                  [position for position in positions])