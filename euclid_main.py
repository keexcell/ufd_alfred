#standard
import yaml
import os
import sys
import gc
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Annulus
#astro specific
from astropy.coordinates import SkyCoord
from astropy.table import Table
#project specific
from ugali.utils import projector
import simple_adl.simple_adl.isochrone as isochrone
from alfred import utils, DataObjects, RegionObjects, masks_and_filters, search_tools, plotting_functions, mapmaking

nside_arg = int(sys.argv[1])
pixel = int(sys.argv[2])

with open('config.yaml', 'r') as ymlfile:
# this is a bit hard coded too but idk another work around
# if main.py stays same folder as the config then this should work
    cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    # assuming that it's cool that the whole github repo is considered "home"
    where = cfg['setup']['where']
    home_dir = os.path.expandvars(cfg['setup']['home_dir'][where])
    pckg_dir = os.path.expandvars(cfg['setup']['pckg_dir'])
    data_dir = os.path.expandvars(cfg['setup']['data_dir'][where])
    valid_pixel_path = os.path.join(pckg_dir, cfg['setup']['valid_pixel_table'])

    output_dir =  os.path.expandvars(cfg['output']['output_dir'])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    results_dir = os.path.join(output_dir, cfg['output']['results_dir'])
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    plots_dir = os.path.join(output_dir, cfg['output']['plots_dir'])
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    nside_shards = cfg['nside_shards']
    nside_search = nside_arg #cfg['nside_search']

    ir_survey = cfg['ir_survey']
    ir_INCOLS = cfg[ir_survey]['INCOLS']
    ir_bands = cfg[ir_survey]['bands']


## Define Which Area We're Looking At -- by nside and by coordinate
#coord = (53.16, -28.10) #just using ECDFS center for now
#index = float(sys.argv[1])
#valid_pixel_table = pd.read_csv(valid_pixel_path)
#pixel = valid_pixel_table.at[index, f'nside{nside_search}']
        #should be the pixel from breaking up the sky into nside_search
print(pixel)
location = pixel #could also switch this to coord, Region can handle both
SearchRegion = RegionObjects.Region(nside_search, location)
#corners_ra, corners_dec = SearchRegion.corners
#buffer_ra, buffer_dec = SearchRegion.buffer_region(corners_ra, corners_dec, SearchRegion.coord_center)

if 'euclid' in ir_survey: # I suppose I'll still keep it general for Roman?
    INCOLS = utils.columns_to_query(ir_INCOLS, ir_bands, output_type = 'string')
    IRData_raw = SearchRegion.euclid_query(INCOLS, preload = True, nside_datashard=nside_shards)
    print(IRData_raw.data.colnames)
    print(f'Data queried, {len(IRData_raw.data)} objects found. Starting cleaning')

snr_mask = masks_and_filters.clean_snr(IRData_raw.VIS.mag, IRData_raw.VIS.magerr, 3)
    # Q: which flags should I enforce?
    # 0=no flags, 8=source close to a border, 512=source within an extended object area
flag_mask = IRData_raw.clean(flags=[0,8,512], bands = None, fwhm_limit = 1.5)
## mix em together
total_mask = snr_mask & flag_mask
## clean up data
Data = IRData_raw.apply_mask(total_mask)
del IRData_raw
gc.collect()
SearchRegion.data_dict[ir_survey] = Data
print('Data cleaned and stored', len(Data.data), ' objects left')

## Select for Stars -- Zerjal + point_like_prob Cuts
morphology_mask = masks_and_filters.Zerjal_stars(Data)
point_like_prob_mask = masks_and_filters.POINT_LIKE_PROB_stars(Data, 0.7)
stars_mask = morphology_mask & point_like_prob_mask
## fill in the masked values with np.nan
stars_table = Data.data[stars_mask]
if len(stars_table)==0:
    print('No stars found. Ending search prematurely. Look at data for debugging.')
    nan_array = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    noPeak = DataObjects.Peak(nan_array, np.nan, np.nan, np.nan)
    utils.write_peak_result(noPeak, results_dir+f'/EMPTY{SearchRegion.nside}_{SearchRegion.pixel}_{Stars.survey}')
    exit()

stars_table_filled = utils.handle_ma_arr(stars_table, solution='fill')
Stars = type(Data)(stars_table_filled, survey=Data.survey)
del Data
gc.collect()

SearchRegion.data_dict['stellar catalog'] = Stars
SearchRegion.data = Stars
print('Stellar catalog made, ', len(Stars.data), ' stars found')

## plot CMD + isochrone for verification
iso1 = isochrone.Isochrone(age=12.0, metallicity=0.0002,
                            distance_modulus=projector.distanceToDistanceModulus(50),
                            survey= 'mixed', band_1= 'Y', band_2= 'VIS')
iso2 = isochrone.Isochrone(age=12.0, metallicity=0.0002,
                             distance_modulus=projector.distanceToDistanceModulus(900),
                             survey= 'mixed', band_1= 'Y', band_2= 'VIS')

plotting_functions.isochrone_plot([iso1, iso2], {'Stellar Sample Number Counts' : {'data':(Stars.VIS, Stars.Y),'markers':'hist'}},
                                  title = 'Euclid only', legend = True, colorbar=False,
                                  x_lim = (-3, 3), y_lim = (26,16), 
                                  save = True, ax=None, filename = f'{SearchRegion.nside}_{SearchRegion.nside}_VIS_Y_{Stars.survey}')
print('CMD isochrone plot ran and saved')


## hotspot search
#distance_array = np.arange(50,3000,50) #distance is given in kpc
distance_mod_array = np.arange(17, 29, .5)
Peaks = []

for distance_modulus in distance_mod_array:
    print(f'Searching at d mod = {distance_modulus}')
    distance = projector.distanceModulusToDistance(distance_modulus)
    iso_sel, iso_stars, iso = search_tools.isochrone_search(Stars.VIS, Stars.Y, 
                                                       distance_modulus, Stars,
                                                       SearchRegion,
                                                       age=12.0, Z=0.0002, 
                                                       save_graph=False)

#     ## need fracdet eventually, but not prioritizing for now
    
    results, hist, hist_smooth = search_tools.search_by_distance(Stars.survey, SearchRegion, distance_modulus, iso_sel, 
                                                    verbose = False, plot = False) 
    
    one_peak_per_row = results.T
    peak_number = np.shape(one_peak_per_row)[0]
    if peak_number==0:
        continue
    for i in range(peak_number):
        #ra_peak, dec_peak, r_peak, sig_peak, distance_modulus, n_obs_peak, n_obs_half_peak, n_model_peak = results_transpose[i]
        Peak_i = DataObjects.Peak(one_peak_per_row[i], hist, hist_smooth, iso_stars, iso, SearchRegion)
        Peaks.append(Peak_i)
        
if len(Peaks)==0:
    print('No significant hotspots found.')
    utils.write_peak_result(Peaks, results_dir+f'/{SearchRegion.nside}_{SearchRegion.pixel}_{Stars.survey}')

        
# for overlapping peaks, save the one with higher sig
print('Number of peaks before: ', len(Peaks))
# first, sort in order of significance
Peaks.sort(key=lambda x: x.sig, reverse=True)
utils.write_peak_result(Peaks, results_dir+f'/BEFOREsimple_{SearchRegion.nside}_{SearchRegion.pixel}_{Stars.survey}', save_format='csv')


fig, ax = plt.subplots(1,2, figsize=(10,5))
overlaps = []
for i in range(len(Peaks)):
    radius = Annulus((Peaks[i].ra, Peaks[i].dec), Peaks[i].r, 0.0002, color='r')
    ax[0].add_artist(radius)
    ax[0].scatter(Peaks[i].ra, Peaks[i].dec, s=1, c='r')
    ax[0].set_title('before')
    for j in range(i+1, len(Peaks)):
        angsep = projector.angsep(Peaks[i].ra, Peaks[i].dec, Peaks[j].ra, Peaks[j].dec)
        if (angsep<=Peaks[i].r) | (angsep<=Peaks[j].r):
            overlaps.append((Peaks[i],Peaks[j]))
            
for pair in overlaps:
    more_sig = pair[0] #because in the pair, i always comes first. i < j and this list is sorted by significance
    less_sig = pair[1]
    if more_sig.sig==less_sig.sig: #I don't know what to do if they have the same significance yet, this should save both
        more_sig.samesig_duplicates.append({'distance' : less_sig.distance, 'distance modulus' : less_sig.distance_modulus,
                                            'r' : less_sig.r,
                                            'ra' : less_sig.ra,  'dec' : less_sig.dec, 
                                            'iso' : less_sig.iso})
    try:
        del Peaks[Peaks.index(less_sig)] #this (hopefully) works because they're all different objects 
                                #therefore none are exactly alike and they won't be deleted 2x
    #    print('successfully deleted above')
    except:
        continue
# as far as I can tell the above code works for duplicates (commented out code is for debugging)

print('Number of peaks after: ', len(Peaks))

for Peak in Peaks:
    radius = Annulus((Peak.ra, Peak.dec), Peak.r, 0.0002, color='r')
    ax[1].add_artist(radius)
    ax[1].scatter(Peak.ra, Peak.dec, s=1, c='r')
    ax[1].set_title('after')
#save duplicates diagnostic plot
save_dir = plots_dir + f'/{ir_survey}_nside{SearchRegion.nside}_pixel{SearchRegion.pixel}'
if not os.path.exists(save_dir):
        os.mkdir(save_dir)
plt.savefig(save_dir + '/peaks_duplicates_beforeafter.png')
plt.close()


final_Peaks = []
for Peak in Peaks:
    print(f'{Peak.sig} sigma; (RA, Dec, d) = ({Peak.ra} deg, {Peak.dec} deg, {Peak.distance} kpc); r = {Peak.r} deg; mu = {Peak.distance_modulus} mag')
    print(Peak.samesig_duplicates)
    
    stars_within_radius = Peak.stars_within_the_radius(scale=1.2, set_attr = True)
    background_r = 0.1
    iso_background_stars = Peak.stars_within_the_radius(starsData=Peak.iso_stars, r=background_r, set_attr = False)
    all_background_stars = Peak.stars_within_the_radius(starsData=Stars, r=background_r, set_attr = False)

    Peak.diagnostic_plots(iso_background_stars, all_background_stars, background_r, SearchRegion,
                          plots_dir, data_dir, 
                          preload=False, save=True, cutouts=True)  

    if len(stars_within_radius.data)<=1:
        #if type(stars_within_radius)==DataObjects.Data:
        #    print('Has type Data')
        #elif type(stars_within_radius)==DataObjects.DESnEuclidData:
        #    print('Has type DESnEuclidData')
        print(' ')
        continue
    else:
        #if type(stars_within_radius)==DataObjects.DESnEuclidData:
        #    print('Has type DESnEuclidData')
        final_Peaks.append(Peak)
        print(' ')
#save all the peaks that are more than 1 star...
utils.write_peak_result(final_Peaks, results_dir+f'/simplecandidates_{SearchRegion.nside}_{SearchRegion.pixel}_{Stars.survey}', save_format='csv')


# now put those peaks into ugali...


## when I need maps:
# maybe put these functions as methods of region object
#full_map = mapmaking.euclid_fullmap('q1.vmpz_healpix_coverage', 'vis', 'coverage', preload=True)
#masked_map = mapmaking.match_map_polygon(full_map, tract.corners)
#tract_map,fracdet_map = mapmaking.rubin_maps(butler, tract.tract, 
#                                             map_name = 'deepCoadd_psf_maglim_map_weighted_mean', band = 'i', 
#                                             nside=2048, 
#                                             save_plot=True, map_title = f'Tract {tract.tract} Rubin i MagLim Map')
#plotting_functions.map_plot(full_map, 'Full Euclid VIS Coverage Map', color_lims = (24,26), 
#         save = True, filename = 'full_coverage_vis_q1')
#plotting_functions.map_plot(masked_map, f'Tract {tract.tract} Euclid VIS Coverage Map', color_lims = (24,26), 
#         save = True, filename = f'{tract.tract}masked_coverage_vis_{euclid_survey}')
#plotting_functions.map_plot(tract_map, f'Tract {tract.tract} Rubin i MagLim Map', color_lims = (24,26), 
#         save = True, filename = f'{tract.tract}_maglim_i_{survey}')


