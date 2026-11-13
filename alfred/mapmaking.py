# standard
from matplotlib import pyplot as plt
import numpy as np
import yaml
import os
import pandas as pd
# astro specific
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import healsparse as hsp
import healpy as hp
# this file also requires hpgeom and Euclid from astroquery_updated
    # however, I decided to import within the functions when they're called
    # to avoid installing lsst packages when only doing a Euclid search (or vice versa)
# project specific
from ugali.utils import healpix
from alfred import utils, plotting_functions

with open('config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    #assuming that it's cool that the whole github repo is considered "home"
    where = cfg['setup']['where']
    home_dir = os.path.expandvars(cfg['setup']['home_dir'][where])
    #external data is gonna be in a directory above - subject to change
    data_dir = os.path.expandvars(cfg['setup']['data_dir'][where])
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    opt_survey = cfg['opt_survey']
    if 'lsst' in opt_survey:
        collection = cfg[opt_survey]['collection'][where]
        skymap = cfg[opt_survey]['skymap']
        

## which Rubin map do I want??

def rubin_fullmap(butler, 
               map_name = 'deepCoadd_psf_maglim_consolidated_map_weighted_mean', band = 'i', fracdet_nside=4096):
    '''
    Learning the healsparse interface and getting map names lives in healsparse-nersc.ipynb
    This function calls up the full Rubin map then restricts to a region, unless the user inputs None
    
    '''
    hspmap_cov = butler.get(map_name + '.coverage',
                        band=band, skymap=skymap)
    cov_pixels, = np.where(hspmap_cov.coverage_mask)
    print(f"querying {band} {map_name} survey map to nside {fracdet_nside}")
    hspmap =  butler.get(map_name, band = band,
                            collections = collection, skymap = skymap,
                            parameters={'pixels':list(cov_pixels), 'degrade_nside': fracdet_nside},)
    hpxmap = hspmap.generate_healpix_map()
    
    fracdet_map = (hpxmap > 0)
    
    fracdet_map = hp.reorder(fracdet_map, n2r = True)
    hpxmap = hp.reorder(hpxmap, n2r = True)

    #hspmap.write(data_dir+f'/maps/fullhealsparse_maglim_{band}_{opt_survey}.fits', clobber=True)
    
    return fracdet_map, hpxmap

def make_fits_file(fracdet_map, hpxmap, fracdet_nside, band, save=True):
    df = pd.DataFrame()

    df['PIXEL']=np.arange(0, len(fracdet_map))
    df['MAGLIM']= hpxmap
    df['FRACDET']= fracdet_map

    table = Table.from_pandas(df, index=False)
    hdu = fits.BinTableHDU(table)
    
    hdu.header['NSIDE'] = fracdet_nside  # e.g., 32 or 4096
    hdu.header['ORDERING'] = 'RING'
    hdu.header['INDXSCHM'] = 'EXPLICIT'

    if save==True:
        print('saving map fits file')
        hdu.writeto(data_dir+f'/maps/hdu_fullmap_{band}_{opt_survey}.fits', overwrite=True)
    return hdu

def euclid_map_query(map_name):
    '''
    map_name : string, the name of the table that I pulled from this code:
        tables = Euclid.load_tables(only_names=True, include_shared_tables=True)
        names = [t.name for t in tables if 'q1' in t.name.lower()]
        ex. q1.vmpz_healpix_coverage

    returns table of the maps of that type, sorted by file_path 
        (file_path is where you'd eventually download them from)
    *THIS DOESN'T SAVE THE MAP*
    '''
    from astroquery_updated.esa.euclid import Euclid

    query = f"SELECT * FROM {map_name}"
    results = Euclid.launch_job_async(query).get_results()
    results.sort('file_path')
    return results

def euclid_fullmap(map_name, band, simple_name, survey, preload=True):
    '''
    Opens or creates/saves the whole sky Euclid map as a Healsparse map
    
    map_name : str, ex. 'q1.vmpz_healpix_coverage', 'q1.vmpz_healpix_footprint_mask'
        code to get more Euclid map names is in scratch/euclid_astroquery.ipynb
    band : str, the filter of which map you want
    simple_name : str, name of the map for the file (i.e. coverage or footprint)
    preload = True means that I want to use the preloaded / saved data instead of querying again
    '''
    if not os.path.exists(data_dir+'/maps'):
        os.mkdir(data_dir+'/maps')
    full_map_path = data_dir + f'/maps/full_{simple_name}_{band.lower()}_{survey}.fits'
    query_check = utils.check_if_query(full_map_path, preload)
    if not query_check:
        print("Check tells me map exists and you don't want to overwrite. Opening map")
        full_map = hsp.HealSparseMap.read(full_map_path)
    if query_check:
        from astroquery.esa.euclid import Euclid
        print("Check tells me map doesn't exists or you want to overwrite. Making map now")
        map_table = euclid_map_query(map_name)

        tiles = [tile_list[0] for tile_list in np.unique(map_table['tile_index_list'])]
        map_list = []
        for tile in tiles:
            output_path = data_dir +'/maps/' + f'/{tile}_{simple_name}_{band}_{euclid_survey}.fits'
            i = np.where((map_table['tile_index_list']==[tile]) & (map_table['filter_name']==band))[0][0]
            path = Euclid.get_product(file_name=map_table['file_name_list'][i], product_id = map_table['product_id'][i], 
                                          output_file= output_path, verbose=False)
            # ! need to figure out if this nside_coverage is telling hsp which coverage to read in 
            # or if it makes an assumption about the properties of this map
            map_list.append(hsp.HealSparseMap.read(output_path, nside_coverage=32))
            # am I going to want to delete the individual tile map file?
            
        full_map = hsp.operations.sum_union(map_list)
        full_map.write(full_map_path, clobber=False)
        
    return full_map

def euclid_tilemap(tile_id, map_name, band, simple_name, preload=True):
    '''
    Opens/makes and saves just the single tile of Euclid map
    
    tile_id : int, Euclid tile index (not tract)
    map_name : str, ex. 'q1.vmpz_healpix_coverage', 'q1.vmpz_healpix_footprint_mask'
        code to get more Euclid map names is in scratch/euclid_astroquery.ipynb
    band : str, filter of which map you want
    simple_name : str, name of the map for the file (i.e. coverage or footprint)
    preload = True means that I want to use the preloaded / saved data instead of querying again
    '''
    if not os.path.exists(data_dir+'/maps'):
        os.mkdir(data_dir+'/maps')
    tile_map_path = data_dir + f'/maps/{tile}_{simple_name}_{band}_{euclid_survey}.fits'
    query_check = utils.check_if_query(tile_map_path,preload)
    if not query_check:
        print("Check tells me map exists and you don't want to overwrite. Opening map")
        tile_map = hsp.HealSparseMap.read(tile_map_path)
    if query_check:
        print("Check tells me map doesn't exists or you want to overwrite. Making map now")
        from astroquery_updated.esa.euclid import Euclid
        map_table = euclid_map_query(map_name)

        i_list = np.where((map_table['tile_index_list']==[tile_id])&(map_table['filter_name']==band))[0]
        if len(i_list) > 1:
            print('more than one map for this tile/filter choice')
        map_list = []
        # just in case there are multiple files for a tile/band (wouldn't think there would be?)
        for i in i_list:
            output_path = data_dir + f'/maps/{tile_id}_{euclid_survey}_{map_name}.fits'
            path = Euclid.get_product(file_name=covmap['file_name_list'][i], 
                                      product_id = covmap['product_id'][i], 
                                      output_file=output_path, verbose=False)
            # ! need to figure out if this nside_coverage is telling hsp which coverage to read in 
            # or if it makes an assumption about the properties of this map
            m = hp.read_map(output_path)
            nside = hp.get_nside(m)
            map_list.append(hsp.HealSparseMap.read(output_path, nside_coverage=32))
        tile_map = hsp.operations.sum_union(map_list)
        tile_map.write(tile_map_path, clobber=False)

    return tile_map

def match_map_polygon(fullmap, Region, step=1, nside=2048):
    '''
    takes in a healsparse map
    and restricts it to be just the polygon area defined by corners

    fullmap : Healsparse map
    '''
    from hpgeom import hpgeom

    polygon = Region.region_borders(return_type='list of tuples',step=step)
    ras = []
    decs =[]
    for coord in polygon:
        ras.append(coord[0])
        decs.append(coord[1])
    
    maskedmap = fullmap[hpgeom.query_polygon(nside, ras, decs, inclusive=True,)]

    return maskedmap
