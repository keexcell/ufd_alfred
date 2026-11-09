import healpy as hp
import numpy as np
import gc
import sys
import os
import yaml
from alfred import utils, plotting_functions, DataObjects
from astropy.table import Table, vstack, join
#Goes up a directories to get the updated astroquery
#I really need to fix this, maybe github submodules or enforcing a version of astroquery
#I think it's version 0.4.11 ?
#sys.path.append(os.path.abspath('../'))
from ugali.utils.projector import match
import ugali.utils.healpix as healpix
## function to create euclid + rubin datasets and register to the data registry on NERSC
## don't know where I want this to live quite yet

with open('config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    #assuming that it's cool that the whole github repo is considered "home"
    where = cfg['setup']['where']
    home_dir = os.path.expandvars(cfg['setup']['home_dir'][where])
    #external data is gonna be in a directory above - subject to change
    data_dir = os.path.join(home_dir, cfg['setup']['data_dir'])
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(data_dir + f'/merged'):
        os.mkdir(data_dir + f'/merged')
    results_dir = os.path.join(home_dir, cfg['output']['results_dir'])
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)


## INSERT function to add it to the data registry

# function to merge catalogs and return the appropriate merged object
def merge_catalogs(PrimaryData, SecondaryData, SearchRegion, preload = True, validation_needed = False, nside_datashard=32):
    # I'm thinking that Primary is the one you want to match to (maybe for reasons of less coverage)
    # don't want to assert yet that one is optical and one is ir
    
    pixel_datashard = healpix.superpixel(SearchRegion.pixel, SearchRegion.nside, nside_datashard)
    file_path = data_dir + f'/merged/{nside_datashard}_{pixel_datashard}_{PrimaryData.survey}_{SecondaryData.survey}_merged.parquet'
    # function to check if the data doesn't exist already and if I want to rewrite it
    if not utils.check_if_query(file_path, preload):
        print("Check tells me data exists and you don't want to remerge. Opening existing file now")
        merged_table = Table.read(file_path)
    else:
        print('Check tells me to start the merge, starting now')

        prim_ra, prim_dec = PrimaryData.ra.data, PrimaryData.dec.data

        NSIDE=4096
        if SearchRegion.nside == NSIDE:
            print('Warning: might cause some issues that NSIDE is already smallest resolution possible. Make sure to check merge')
        ## get the unique pixels of primary dataset
        prim_pix4096 = hp.ang2pix(NSIDE, prim_ra, prim_dec, lonlat=True)
        prim_upix4096 = np.unique(prim_pix4096, return_counts=False)
        ## then get the pixels of secondary data
        secun_pix4096 = hp.ang2pix(NSIDE, SecondaryData.ra, SecondaryData.dec, lonlat=True)
        ## We only keep the sources that lie in the Primary survey coverage
        spatial_mask = np.isin(secun_pix4096, prim_upix4096) #enforce [prim_cts > 8]) ?
        SecondaryData_masked = SecondaryData.apply_mask(spatial_mask)
        secun_ra, secun_dec = SecondaryData_masked.ra, SecondaryData_masked.dec

        del NSIDE, prim_upix4096, secun_pix4096, spatial_mask
        gc.collect()
        ## match() is a spatial match from ugali tools, tol controls how generous you are in saying the sources overlap
        if len(secun_ra) == 0:
            print('uh oh, no overlap detected')
            return 0

        indexprim, indexsecun, ds = match(prim_ra, prim_dec, secun_ra, secun_dec, tol = 0.0003)
        matchesPrim = PrimaryData.apply_mask(indexprim)
        unmatchedPrim = PrimaryData.apply_mask(~indexprim)
        matchesSecun = SecondaryData_masked.apply_mask(indexsecun)
        unmatchedSecun = SecondaryData_masked.apply_mask(~indexsecun)
        if len(matchesPrim.data) != len(matchesSecun.data):
            print("Something isn't right: those lengths don't match")
        del indexprim, indexsecun, prim_ra, prim_dec, secun_ra, secun_dec
        gc.collect()
        
        ## now merging our matches into one catalog with all LSST and Euclid columns
        matchesPrim.data['_match_id'] = np.arange(len(matchesPrim.data))
        matchesSecun.data['_match_id'] = np.arange(len(matchesSecun.data))
        merged_table = join(matchesPrim.data, matchesSecun.data, keys='_match_id')
        merged_table.write(file_path, format='parquet', overwrite = True)
        print(f'New merged catalog data written to {file_path}')

    # I don't know how else to do this logic 
    if 'euclid' in PrimaryData.survey or 'euclid' in SecondaryData.survey:
        if 'lsst' in PrimaryData.survey or 'lsst' in SecondaryData.survey:
            if validation_needed==True and preload==False:
                try: #if this code runs, it means lsst is primary data
                    match1Band, unmatch1Band, full1Band = matchesPrim.i, unmatchedPrim.i, PrimaryData.i
                    match2Band, unmatch2Band, full2Band = matchesSecun.VIS, unmatchedSecun.VIS, SecondaryData_masked.VIS
                    merged_df_coord1 = (merged_table['coord_ra'],merged_table['coord_dec'])
                    merged_df_coord2 = (merged_table['RIGHT_ASCENSION'],merged_table['DECLINATION'])
                except: #if this one runs, it means euclid is primary data
                    match1Band, unmatch1Band, full1Band = matchesPrim.VIS, unmatchedPrim.VIS, PrimaryData.VIS
                    match2Band, unmatch2Band, full2Band = matchesSecun.i, unmatchedSecun.i, SecondaryData_masked.i
                    merged_df_coord1 = (merged_table['RIGHT_ASCENSION'], merged_table['DECLINATION'])
                    merged_df_coord2 = (merged_table['coord_ra'], merged_table['coord_dec'])
                plotting_functions.match_validation_plots(match1Band, unmatch1Band, full1Band,
                                                          match2Band, unmatch2Band, full2Band,
                                                          merged_df_coord1, merged_df_coord2,
                                                          matchesPrim, matchesSecun,
                                                          SearchRegion, PrimaryData, SecondaryData_masked
                                                         )
                del matchesPrim, matchesSecun, unmatchedPrim, unmatchedSecun, SecondaryData_masked, ds
                gc.collect()
            elif validation_needed==True and preload==True:
                print('Automatic validation plots on preloaded data are not supported right now. Please run functions manually')
            mergedData = DataObjects.LSSTnEuclidData(merged_table,
                                                     f'{PrimaryData.survey}_{SecondaryData.survey}',
                                                     coord_choice='LSST')
            mergedData = SearchRegion.region_cut(mergedData, finer_nside = 4096)
            
        elif 'des' in PrimaryData.survey or 'des' in SecondaryData.survey:
            if validation_needed==True and preload==False:
                try: #if this code runs, it means des is primary data
                    match1Band, unmatch1Band, full1Band = matchesPrim.i, unmatchedPrim.i, PrimaryData.i
                    match2Band, unmatch2Band, full2Band = matchesSecun.VIS, unmatchedSecun.VIS, SecondaryData_masked.VIS
                    merged_df_coord1 = (merged_table['alphawin_j2000'],merged_table['deltawin_j2000'])
                    merged_df_coord2 = (merged_table['RIGHT_ASCENSION'],merged_table['DECLINATION'])
                except: #if this one runs, it means euclid is primary data
                    match1Band, unmatch1Band, full1Band = matchesPrim.VIS, unmatchedPrim.VIS, PrimaryData.VIS
                    match2Band, unmatch2Band, full2Band = matchesSecun.i, unmatchedSecun.i, SecondaryData_masked.i
                    merged_df_coord1 = (merged_table['RIGHT_ASCENSION'], merged_table['DECLINATION'])
                    merged_df_coord2 = (merged_table['alphawin_j2000'],merged_table['deltawin_j2000'])
                plotting_functions.match_validation_plots(match1Band, unmatch1Band, full1Band,
                                                          match2Band, unmatch2Band, full2Band,
                                                          merged_df_coord1, merged_df_coord2,
                                                          matchesPrim, matchesSecun,
                                                          SearchRegion, PrimaryData, SecondaryData_masked
                                                         )
                del matchesPrim, matchesSecun, unmatchedPrim, unmatchedSecun, SecondaryData_masked, ds
                gc.collect()
            elif validation_needed==True and preload==True:
                print('Automatic validation plots on preloaded data are not supported right now. Please run functions manually')
            mergedData = DataObjects.DESnEuclidData(merged_table,
                                                     f'{PrimaryData.survey}_{SecondaryData.survey}',
                                                     coord_choice='DES')
            mergedData = SearchRegion.region_cut(mergedData, finer_nside = 4096)
            
    # then more if statements for the other surveys...

    del merged_table, PrimaryData, SecondaryData
    gc.collect()

    return mergedData
