## I don't know if this should live in a separate file or as a method of the dataclasses 
## All of these functions will return masks that can be put into the apply_mask methods

import numpy as np

## QUALITY MASKS BELOW HERE
def clean_snr(band, error, threshold=5):
    return (band/error > threshold)

def clean_lsst(data, bands):
    mask = (data['detect_isIsolated'] == True)
    for band in bands:
        mask &= (data[f'{band}_psfFlux_flag'] == 0)
    return mask

def clean_euclid(data, flags, bands = None, fwhm_limit = 1.5):
    mask = (data['SPURIOUS_FLAG'] == 0)
    ## I think the det_quality_flag encompasses the per band flags
    #for band in bands:
    #    mask &= (data[f'FLAG_{band}'] == 0)    
    ## see https://euclid.esac.esa.int/dr/q1/dpdd/merdpd/merphotometrycookbook.html abt which flags to enforce
    for flag in flags:
        mask |= (data['DET_QUALITY_FLAG'] == flag)
    ## enforcing that euclid sources having FWHM above 1.5" is spurious (Zerjal et al. suggestion)
    mask &= (data['FWHM'] <= fwhm_limit)
    return mask

def clean_des(data):
    mask = (data['flags_gold'] == 0)
    mask &= (data['flags_footprint']==1)
    #mask &= (data['flags_foreground'])
    return mask    

def clean_nans(band, band_err):
    mask = (np.isfinite(band))
    mask &= (np.isfinite(band_err))
    return mask


## STELLAR MASKS BELOW HERE
def niroptical_color_stars(MergedData_object):
    colormask_left = ((MergedData_object.g.mag - MergedData_object.z.mag) <= 0.3)
    colormask_left &= ((MergedData_object.z.mag - MergedData_object.H.mag) < (-.5 + 1.7*(MergedData_object.g.mag - MergedData_object.z.mag)))
    #should I be enforcing a left end cut?
    colormask_left &= ((MergedData_object.g.mag - MergedData_object.z.mag) > -1)
   
    colormask_right = ((MergedData_object.g.mag - MergedData_object.z.mag) >= 0.3)
    colormask_right &= ((MergedData_object.z.mag - MergedData_object.H.mag) < (-0.1 + 0.25*(MergedData_object.g.mag - MergedData_object.z.mag)))
    #should I be enforcing a right end cut?
    colormask_right &= ((MergedData_object.g.mag - MergedData_object.z.mag) < 4.5)

    #preserve those with no H detections
    colormask_Hnans = (np.isnan(MergedData_object.H.mag)) | (MergedData_object.H.mag/MergedData_object.H.magerr < 3)
    
    colormask = colormask_left | colormask_right | colormask_Hnans
    return colormask
    
def Zerjal_stars(MergedData_object):
    '''
    based on Zerjal et al. 2025, Euclid morphology stellar cut
    requires a MergedData object with .ellipticity and .mumax_minus_mag attributes
    returns a stellar selection
    '''
    stars_mask = (MergedData_object.ellipticity < 0.2) 
    stars_mask &= (MergedData_object.mumax_minus_mag < -2.7) & (MergedData_object.mumax_minus_mag > -3.25)
    return stars_mask

def POINT_LIKE_PROB_stars(MergedData_object, threshold):
    '''
    requires a MergedData object with .pointlikeprob attribute
    returns a stellar selection with POINT_LIKE_PROB above the threshold
    '''
    stars_mask = (MergedData_object.pointlikeprob > threshold)
    return stars_mask

def POINT_LIKE_FLAG_stars(MergedData_object):
    '''
    requires a MergedData object with .data attribute and POINT_LIKE_FLAG column
    returns a stellar selection (where POINT_LIKE_FLAG is true)
    '''
    stars_mask = (MergedData_object.data['POINT_LIKE_FLAG'] == 1)
    return stars_mask