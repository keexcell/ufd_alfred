#standard
import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml
import os
#astro specific
from astropy.table import Table
#project specific
import simple_adl.simple_adl.isochrone as isochrone
from alfred import plotting_functions, utils


def isochrone_search(band1, band2, distance_modulus, starData, SearchRegion, age=12.0, Z=0.0002, mag_max=26, save_graph=True):
    #the isochrone with Euclid, Roman, and LSST bands is 'mixed'
    if 'des' in starData.survey.lower():
        survey='des'
    else:
        survey='mixed'
    iso = isochrone.Isochrone(
                              age=age,
                              metallicity=Z,
                              distance_modulus=distance_modulus,
                              survey= survey,
                              band_1= band1.str,
                              band_2= band2.str)
    
    iso_sel = cut_isochrone_path(band1.mag, band2.mag,
                             band1.magerr, band2.magerr,
                             iso, radius = 0.1, mag_max=mag_max)

    '''
    Note about scipy interpolate used in cut_isochrone_path:
        Neither of these two approaches (masked arrays and filling missing values with Nans) is directly supported 
        in scipy.interpolate. Individual routines may offer partial support, and/or workarounds, but in general, 
        the library firmly adheres to the IEEE 754 semantics where a NaN means not-a-number, 
        i.e. a result of an illegal mathematical operation (e.g., division by zero), not missing.
    '''
    
    iso_starsData = starData.apply_mask(iso_sel)
    isostars_band1 = band1.apply_mask(iso_sel)
    isostars_band2 = band2.apply_mask(iso_sel)

    if save_graph == True:
        title = f'''{starData.survey.replace('_', ' & ')} Stars (nside {SearchRegion.nside} pixel {SearchRegion.pixel}) 
                    \n Isochrone: {age} Gyr, Z={Z} at Dist Mod = {distance_modulus}'''
        # this plotting function is to be edited/generalized
        plotting_functions.isochrone_plot([iso], 
                                          {'Stars within isochrone' : (isostars_band1, isostars_band2), 
                                           'All stars' : (band1, band2)},
                                          title=title, 
                                          save = True, filename = f'{starData.survey}_iso_cut_{distance_modulus}distmod')

    return iso_sel, iso_starsData, iso
        
def cut_isochrone_path(g, r, g_err, r_err, isochrone, radius=0.01, mag_max = 26, return_all=False):
    #Authors: Keith Bechtol, Sid Mau from the "simple" algorithm: https://github.com/DarkEnergySurvey/simple/tree/master
    """
    Cut to identify objects within isochrone cookie-cutter.
    """
    if np.all(isochrone.stage == 'Main'):
        # Dotter case
        index_transition = len(isochrone.stage)
    else:
        # Other cases
        index_transition = np.nonzero(isochrone.stage >= isochrone.hb_stage)[0][0] + 1    

    mag_1_rgb = isochrone.mag_1[0: index_transition] + isochrone.distance_modulus
    mag_2_rgb = isochrone.mag_2[0: index_transition] + isochrone.distance_modulus
    
    mag_1_rgb = mag_1_rgb[::-1]
    mag_2_rgb = mag_2_rgb[::-1]

    # Cut one way...
    f_isochrone = scipy.interpolate.interp1d(mag_2_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = np.fabs((g - r) - f_isochrone(r))
    cut_2 = (color_diff < np.sqrt(radius**2 + r_err**2 + g_err**2))

     # ...and now the other
    f_isochrone = scipy.interpolate.interp1d(mag_1_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = np.fabs((g - r) - f_isochrone(g))
    cut_1 = (color_diff < np.sqrt(radius**2 + r_err**2 + g_err**2))

    cut = np.logical_or(cut_1, cut_2)

    #mag_bins = np.arange(17., 24.1, 0.1)
    mag_bins = np.arange(17., mag_max+0.1, 0.1)
    mag_centers = 0.5 * (mag_bins[1:] + mag_bins[0:-1])
    magerr = np.tile(0., len(mag_centers))
    for ii in range(0, len(mag_bins) - 1):
        cut_mag_bin = (g > mag_bins[ii]) & (g < mag_bins[ii + 1])
        magerr[ii] = np.median(np.sqrt(radius**2 + r_err[cut_mag_bin]**2 + g_err[cut_mag_bin]**2))

    if return_all:
        return cut, mag_centers[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) + magerr)[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) - magerr)[f_isochrone(mag_centers) < 100]
    else:
        return cut

def search_by_distance(survey, Region, distance_modulus, iso_sel, extension=None, verbose=True, plot=False):
    #credit to the authors of simple_adl
        #I had to copy/paste to avoid things in their package overriding my config file variables
        #and I had to adjust one thing to get it working with my Region object
            #changed the line -> if (len(region.data.data[iso_sel]))
    """
    Idea: 
    Send a data extension that goes to faint magnitudes, e.g., g < 24.
    Use the whole region to identify hotspots using a slightly brighter 
    magnitude threshold, e.g., g < 23, so not susceptible to variations 
    in depth. Then compute the local field density using a small annulus 
    around each individual hotspot, e.g., radius 0.3 to 0.5 deg.
    """
    if (len(Region.data.data[iso_sel]) == 0):
        return [], [], [], [], [], [], [], []

    ra_peak_array = []
    dec_peak_array = []
    r_peak_array = []
    sig_peak_array = []
    distance_modulus_array = []
    n_obs_peak_array = []
    n_obs_half_peak_array = []
    n_model_peak_array = []

    Region.density = Region.characteristic_density(iso_sel, verbose=verbose)
    x_peak_array, y_peak_array, angsep_peak_array, hist = Region.find_peaks(iso_sel, distance_modulus, survey, plot=plot)
    for x_peak, y_peak, angsep_peak in zip(x_peak_array, y_peak_array, angsep_peak_array):
        # Aperture fitting
        if verbose: print('Fitting aperture to hotspot...')
        ra_peaks, dec_peaks, r_peaks, sig_peaks, n_obs_peaks, n_obs_half_peaks, n_model_peaks, density = Region.fit_aperture(iso_sel, x_peak, y_peak, angsep_peak, distance_modulus, survey, verbose=verbose, extension=extension, plot=plot)
        
        ra_peak_array.append(ra_peaks)
        dec_peak_array.append(dec_peaks)
        r_peak_array.append(r_peaks)
        sig_peak_array.append(sig_peaks)
        distance_modulus_array.append(distance_modulus*np.ones(len(ra_peaks)))
        n_obs_peak_array.append(n_obs_peaks)
        n_obs_half_peak_array.append(n_obs_half_peaks)
        n_model_peak_array.append(n_model_peaks)
        
    try:
        ra_peak_array = np.concatenate(ra_peak_array)
        dec_peak_array = np.concatenate(dec_peak_array)
        r_peak_array = np.concatenate(r_peak_array)
        sig_peak_array = np.concatenate(sig_peak_array)
        distance_modulus_array = np.concatenate(distance_modulus_array)
        n_obs_peak_array = np.concatenate(n_obs_peak_array)
        n_obs_half_peak_array = np.concatenate(n_obs_half_peak_array)
        n_model_peak_array = np.concatenate(n_model_peak_array)
    except ValueError:
        print('No arrays to concatenate')

    #added this line and changed return statement to return h_region
    results = np.asarray([ra_peak_array, dec_peak_array, r_peak_array, sig_peak_array, distance_modulus_array, n_obs_peak_array, n_obs_half_peak_array, n_model_peak_array])
    return results, hist
