import yaml
import os
from astropy.table import Table,vstack
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import gc
import scipy
import matplotlib.pyplot as plt

from alfred import utils, DataObjects

#I really need to fix that I'm importing a git repo of updated astroquery
    #maybe github submodules or enforcing a version of astroquery
    #I think it's version 0.4.11 or 10?
#idk if this is standard or astro specific
from pyvo.dal.tap import TAPService
#project specific
from ugali.utils import healpix
import ugali.utils.projector as projector

with open('config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    where = cfg['setup']['where']
    opt_survey = cfg['opt_survey']
    ir_survey = cfg['ir_survey']
    if 'lsst' in opt_survey:
        skymap = cfg[opt_survey]['skymap']
        repo_config = cfg[opt_survey]['repo_config'][where]
        collection = cfg[opt_survey]['collection'][where]
    home_dir = os.path.expandvars(cfg['setup']['home_dir'][where])
    data_dir = os.path.join(home_dir, cfg['setup']['data_dir'])
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    plots_dir = os.path.join(home_dir, cfg['output']['plots_dir'])
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

class Region():
    #for sharding data into large healpy pixel regions
    def __init__(self, nside, location):
        self.nside = nside
        if type(location)==int:
            pixel = location
        elif type(location)==tuple:
            ra_og,dec_og = location
            pixel = healpix.ang2pix(nside, ra_og, dec_og)
        ra_center,dec_center = healpix.pix2ang(nside, pixel)
        self.pixel = pixel
        self.pix_center = pixel
        self.coord_center = (ra_center, dec_center)
        phi = healpix.lon2phi(ra_center)
        theta = healpix.lat2theta(dec_center)
        self.pixel_neighbors = hp.pixelfunc.get_all_neighbours(nside, theta, phi=phi) 
                #returns 8 nearest pixel indices - not sure if it's necessary/helpful
        self.fracdet = None
        self.proj = projector.Projector(ra_center,dec_center)
        self.corners = hp.vec2ang(hp.boundaries(nside, pixel, step=1, nest=False).T, lonlat=True)
                #corners is of the format (ra_arr, dec_arr)
        self.buffer_corners = None

        #getting these overlapping regions for data querying purposes
        self.rubin_tracts = -1
        self.euclid_tiles = -1
        self.des_tiles = -1

        self.data_dict = {}

    def region_cut(self, Data, buffer=True, finer_nside = 4096):
        '''
        input a Data Object that has an ra and dec attribute, needs to be in degrees, and apply_mask method
        '''
        if buffer==True:
            try:
                #get pixels in buffer region
                vec_arr = healpix.ang2vec(self.buffer_corners[0], self.buffer_corners[1])
                ra_postvec, dec_postvec = hp.vec2ang(vec_arr)
                subpix = hp.query_polygon(finer_nside,vec_arr,inclusive=True)
            except ValueError:
                print("Buffer region probably not run yet, so it's still set to None")
        else:
            subpix = healpix.subpixel(self.pixel, self.nside, finer_nside)
        datapix = healpix.ang2pix(finer_nside, Data.ra, Data.dec)
        mask = np.isin(datapix, subpix)
        regionData = Data.apply_mask(mask)
        return regionData

    def buffer_region(self, ra_corners, dec_corners, center, scale=1.1):
        '''
        sets the buffer_corners attribute to be the corners of the wider borders, formatted as ra_arr,dec_arr
        also returns the ra and dec arrays
        '''
        ra_recentered, dec_recentered = ra_corners-center[0], dec_corners-center[1]
        ra_rescaled, dec_rescaled = scale*ra_recentered, scale*dec_recentered
        ra_new, dec_new = ra_rescaled+center[0], dec_rescaled+center[1]
        self.buffer_corners = ra_new, dec_new
        return ra_new, dec_new

    def region_borders(self, return_type = 'string', step = 1, buffer = True):
        '''
        Supported types to return are 
        1. "string" - returns a string of format "ra1,dec1,ra2,dec2..."
        2. "list of tuples" - returns what it sounds like, a list of tuples of format [(ra,dec)...]
            if the return_type is given as just "list", function will default to this
        3. "SkyCoord list" - returns list of SkyCoord objects
        4. "coord-separated list" - returns 2 lists, one for ras, one for decs

        all ra and dec are in degrees (hopefully)
        '''
        if buffer==True:
            try:
                ra_arr, dec_arr = self.buffer_corners
            except ValueError:
                print("Buffer region probably not run yet, so it's still set to None")
        else:
            ra_arr, dec_arr = self.corners
        if return_type == 'string':
            border_str = ''
            for ra, dec in zip(ra_arr, dec_arr):
                border_str += f'{ra}, {dec}, '
            border_str = border_str.removesuffix(', ')
            return border_str
        elif return_type == 'list of tuples' or return_type=='list':
            border_list = []
            for ra, dec in zip(ra_arr, dec_arr):
                border_list.append((ra, dec))
            return border_list
        elif return_type == 'SkyCoord list':
            border_list = []
            for ra, dec in zip(ra_arr, dec_arr):
                border_list.append(SkyCoord(ra*u.deg, dec*u.deg, frame='icrs'))
            return border_list
        elif return_type=='coord-separated list':
            return ra_arr, dec_arr
        else:
            print('Type not supported by region_borders function. Please input string, list of tuples, or SkyCoord list')
            return None        
        
    def get_rubin_tracts(self, butler, finer_nside=4096, buffer=True):
        '''
        SkyMap = skyMap object, generated from butler
        '''
        import lsst.geom as geom

        SkyMap =  butler.get('skyMap', skymap=skymap, collections=collection)
        if buffer==True:
            try:
                vec_arr = healpix.ang2vec(self.buffer_corners[0], self.buffer_corners[1])
                ra_postvec, dec_postvec = hp.vec2ang(vec_arr)
                subpix = hp.query_polygon(finer_nside,vec_arr,inclusive=True)
            except ValueError:
                print("Buffer region probably not run yet, so it's still set to None")
        else:
            subpix = healpix.subpixel(self.pixel, self.nside, finer_nside)
        tract_ids = []
        for pix in subpix:
            ra, dec = healpix.pix2ang(finer_nside, pix)
            TractsInfo = SkyMap.findTract(geom.SpherePoint(ra*geom.degrees,dec*geom.degrees))
            tract_ids.append(TractsInfo.tract_id)
        tract_ids = np.unique(tract_ids)
        
        self.rubin_tracts = tract_ids #maybe have them as Tract objects?

        return tract_ids

    def rubin_query(self, butler, INCOLS, tract_arr=None, buffer=True):
        '''
        I don't have a preload because I figured that we'd just be using the butler and not saving
        
        queries by tract but for a whole array of tracts, then restricts based on a healpix mask 
        of what is actually in that region
        '''
        if tract_arr is None:
            tract_arr = self.get_rubin_tracts(butler, finer_nside=4096, buffer=buffer)
        print('Querying Rubin tracts ', tract_arr)
        
        rubin_data_list = [] #these are a bunch of LSSTData objects -- need to think how to concatenate
        for tract in tract_arr:
            full_tract = butler.get('object', 
                                    dataId={'skymap': skymap, 'tract': tract}, 
                                    collections=collection, parameters={"columns":INCOLS})
            rubin_data_list.append(self.region_cut(DataObjects.LSSTData(full_tract,opt_survey)).data)
                #region_cut accepts and returns an LSSTData object, so append just the data of that object
            del full_tract
            gc.collect()
        rubinData = DataObjects.LSSTData(vstack(rubin_data_list), opt_survey)
            #shove all the tract data together, make it an LSST object
        #store it in the object as the LSST object but return it just as a table
        self.data_dict[opt_survey] = rubinData
        print('Rubin query done!')
        return rubinData

    def euclid_query(self, INCOLS, preload = True, nside_datashard=32, step=1, buffer=True):
        '''
        checks if the data already exists for that survey and nside/pixel. 
        if it does, we won't query again, if we query, this function saves the new queried data
            preload means "yes I want you to load the data if it exists"
            might be False if you want to overwrite the data
        either way, this assigns the data to self.euclid_data and self.data['euclid survey'] 
        '''
        if not os.path.exists(data_dir + f'/{ir_survey}'):
            print("no Euclid data folder, making one now")
            os.mkdir(data_dir + f'/{ir_survey}')

        pixel_datashard = healpix.superpixel(self.pixel, self.nside, nside_datashard)
        file_dir = data_dir + f'/{ir_survey}/{nside_datashard}_{pixel_datashard}_euclid.parquet'
        if not utils.check_if_query(file_dir, preload):
            print(f'''Check tells me Euclid data exists for nside = {nside_datashard}, pixel = {pixel_datashard}, 
                      and you don't want to overwrite. Opening existing file now''')
            results_table = Table.read(file_dir)
        else:
            print(f'''Check tells me Euclid data doesn't exist for nside = {nside_datashard}, pixel = {pixel_datashard} 
                      or you do want to overwrite, querying now''')
            from astroquery_updated.esa.euclid import Euclid
            Euclid.get_status_messages()
            
            query = f'SELECT {INCOLS} FROM mer_catalogue'
            radius = 1.7 #going for bigger than a tract
            #query += f''' WHERE DISTANCE({self.center.ra.value}, {self.center.dec.value},
            #                            right_ascension, declination) < {radius}'''
            #query += f''' WHERE CONTAINS(POINT('ICRS', RIGHT_ASCENSION, DECLINATION),
            #             POLYGON('ICRS', {self.region_borders(return_type='string',step=step)})) = 1'''
            ra_arr, dec_arr = self.region_borders(return_type='coord-separated list',step=step,buffer=buffer)
            query += f''' WHERE right_ascension BETWEEN {np.min(ra_arr)} AND {np.max(ra_arr)}
                        AND declination BETWEEN {np.min(dec_arr)} AND {np.max(dec_arr)}'''
    
            results_table = Euclid.launch_job_async(query, verbose=False).get_results()
            results_table.write(file_dir, format='parquet', overwrite = True)
        
        #storing it as the euclid object so we have access to attributes
        results = DataObjects.EuclidData(results_table, ir_survey)
        results = self.region_cut(results, finer_nside = 4096,buffer=buffer)
        self.data_dict[ir_survey] = results
        
        return results

    def des_query(self, INCOLS, preload = True, nside_datashard = 32, step=1, buffer=True):
        '''
        thinking that this would be really similar to the Euclid function in form
        taking in the columns, using the region borders, returning results/updating attributes

        self.data[DES survey] = DESData(query_result)
        return query_result
        '''
        if not os.path.exists(data_dir + f'/{opt_survey}'):
            print("no DES data folder, making one now")
            os.mkdir(data_dir + f'/{opt_survey}')
            
        pixel_datashard = healpix.superpixel(self.pixel, self.nside, nside_datashard)
        file_dir = data_dir + f'/{opt_survey}/{nside_datashard}_{pixel_datashard}_des.parquet'
        if not utils.check_if_query(file_dir, preload):
            print(f'''Check tells me DES data exists for nside = {nside_datashard}, pixel = {pixel_datashard}, 
                      and you don't want to overwrite. Opening existing file now''')
            resultsData = DataObjects.DESData(Table.read(file_dir), opt_survey)
        else:
            print(f'''Check tells me DES data doesn't exist for nside = {nside_datashard}, pixel = {pixel_datashard} 
                      or you do want to overwrite, querying now''')
            
            tap = TAPService("https://datalab.noirlab.edu/tap")
            ra_arr, dec_arr = self.region_borders(return_type='coord-separated list',step=step, buffer=buffer)
            query = f'''SELECT {INCOLS} FROM des_dr2.y6_gold
                        WHERE ra BETWEEN {np.min(ra_arr)} AND {np.max(ra_arr)}
                        AND dec BETWEEN {np.min(dec_arr)} AND {np.max(dec_arr)}'''
            job = tap.run_async(query)
            results_table = job.to_table()
            results_table.write(file_dir, format='parquet', overwrite = True)
            '''
            #des_dr2.y6_gold?
            from dl import queryClient as qc
            results_table = qc.query(query, fmt = 'table', qtype='adql', verbose=False)
            '''
            resultsData = DataObjects.DESData(results_table, opt_survey)
            
        desData = self.region_cut(resultsData, finer_nside = 4096, buffer=buffer)
        #storing it as the des object so we have access to attributes
        self.data_dict[opt_survey] = desData
        
        return desData

# BELOW METHODS ARE COPIED AND MODIFIED FROM SIMPLE_ADL TO MAKE REGION OBJECT MATCH THEIRS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # couldn't just directly use it because it's part of a Region object which I'm initializing differently...
    # and I had to change the self.data.survey.catalog['basis1'] bc I couldn't find where that was pointing (changed to self.data.basis1[iso_sel])
    def characteristic_density(self, iso_sel, verbose=True):
        """
        Compute the characteristic density of a region
        Convolve the field and find overdensity peaks

        iso_sel : mask, from cut_isochrone_path
        """

        x, y = self.proj.sphereToImage(self.data.basis1[iso_sel], self.data.basis2[iso_sel]) # Trimmed magnitude range for hotspot finding
        #x, y = self.proj.sphereToImage(self.data[self.survey.catalog['basis_1']], self.data[self.survey.catalog['basis_2']]) # If we want to use full magnitude range for significance evaluation (used to be x_full, y_full = proj.sphereToImage(data[basis_1], data[basis_2])
        delta_x = 0.01
        area = delta_x**2
        smoothing = 2. / 60. # Was 3 arcmin
        bins = np.arange(-8., 8. + 1.e-10, delta_x)
        centers = 0.5 * (bins[0: -1] + bins[1:])
        yy, xx = np.meshgrid(centers, centers)
    
        h = np.histogram2d(x, y, bins=[bins, bins])[0]
    
        h_g = scipy.ndimage.filters.gaussian_filter(h, smoothing / delta_x)
    
        delta_x_coverage = 0.1
        area_coverage = (delta_x_coverage)**2
        bins_coverage = np.arange(-5., 5. + 1.e-10, delta_x_coverage)
        h_coverage = np.histogram2d(x, y, bins=[bins_coverage, bins_coverage])[0]
        h_goodcoverage = np.histogram2d(x, y, bins=[bins_coverage, bins_coverage])[0]
    
        n_goodcoverage = h_coverage[h_goodcoverage > 0].flatten()
    
        characteristic_density = np.median(n_goodcoverage) / area_coverage # per square degree
        if verbose: print('Characteristic density = {:0.1f} deg^-2'.format(characteristic_density))
    
        # Use pixels with fracdet ~1.0 to estimate the characteristic density
        if self.fracdet is not None:
            fracdet_zero = np.tile(0., len(self.fracdet))
            cut = (self.fracdet != hp.UNSEEN)
            fracdet_zero[cut] = self.fracdet[cut]
    
            nside_fracdet = hp.npix2nside(len(self.fracdet))
            
            subpix_region_array = []
            for pix in np.unique(hp.ang2pix(self.nside,
                                            self.data.basis1[iso_sel],
                                            self.data.basis2[iso_sel],
                                            lonlat=True)):
                subpix_region_array.append(subpixel(self.pix_center, self.nside, nside_fracdet))
            subpix_region_array = np.concatenate(subpix_region_array)
    
            # Compute mean fracdet in the region so that this is available as a correction factor
            cut = (self.fracdet[subpix_region_array] != hp.UNSEEN)
            mean_fracdet = np.mean(self.fracdet[subpix_region_array[cut]])
    
            # Correct the characteristic density by the mean fracdet value
            characteristic_density_raw = 1. * characteristic_density
            characteristic_density /= mean_fracdet 
            if verbose: print('Characteristic density (fracdet corrected) = {:0.1f} deg^-2'.format(characteristic_density))

        return(characteristic_density)
    
    def characteristic_density_local(self, iso_sel, x_peak, y_peak, angsep_peak, verbose=True):
        """
        Compute the local characteristic density of a region
        """
        characteristic_density = self.density
    
        x, y = self.proj.sphereToImage(self.data.basis1[iso_sel], self.data.basis2[iso_sel]) # Trimmed magnitude range for hotspot finding
        #x, y = self.proj.sphereToImage(self.data[self.survey.catalog['basis_1']], self.data[self.survey.catalog['basis_2']]) # If we want to use full magnitude range for significance evaluation
    
        # If fracdet map is available, use that information to either compute local density,
        # or in regions of spotty coverage, use the typical density of the region
        if self.fracdet is not None:
            # The following is copied from how it's used in compute_char_density
            fracdet_zero = np.tile(0., len(self.fracdet))
            cut = (self.fracdet != hp.UNSEEN)
            fracdet_zero[cut] = self.fracdet[cut]
    
            nside_fracdet = hp.npix2nside(len(self.fracdet))
            
            subpix_region_array = []
            for pix in np.unique(hp.ang2pix(self.nside,
                                            self.data.basis1[iso_sel], self.data.basis2[iso_sel],
                                            lonlat=True)):
                subpix_region_array.append(subpixel(self.pix_center, self.nside, nside_fracdet))
            subpix_region_array = np.concatenate(subpix_region_array)
    
            # Compute mean fracdet in the region so that this is available as a correction factor
            cut = (self.fracdet[subpix_region_array] != hp.UNSEEN)
            mean_fracdet = np.mean(self.fracdet[subpix_region_array[cut]])
    
            subpix_region_array = subpix_region_array[self.fracdet[subpix_region_array] > 0.99]
            subpix = hp.ang2pix(nside_fracdet, 
                                self.data.basis1[cut_magnitude_threshold][iso_sel], 
                                self.data.basis2[cut_magnitude_threshold][iso_sel],
                                lonlat=True)
    
            # This is where the local computation begins
            ra_peak, dec_peak = self.proj.imageToSphere(x_peak, y_peak)
            subpix_all = hp.query_disc(nside_fracet, hp.ang2vec(ra_peak, dec_peak, lonlat=True), np.radians(0.5))
            subpix_inner = hp.query_disc(nside_fracet, hp.ang2vec(ra_peak, dec_peak, lonlat=True), np.radians(0.3))
            subpix_annulus = subpix_all[~np.in1d(subpix_all, subpix_inner)]
            mean_fracdet = np.mean(fracdet_zero[subpix_annulus])
            print('mean_fracdet {}'.format(mean_fracdet))
            if mean_fracdet < 0.5:
                characteristic_density_local = characteristic_density
                if verbose: print('characteristic_density_local baseline {}'.format(characteristic_density_local))
            else:
                # Check pixels in annulus with complete coverage
                subpix_annulus_region = np.intersect1d(subpix_region_array, subpix_annulus)
                if verbose: print('{} percent pixels with complete coverage'.format(float(len(subpix_annulus_region)) / len(subpix_annulus)))
                if (float(len(subpix_annulus_region)) / len(subpix_annulus)) < 0.25:
                    characteristic_density_local = characteristic_density
                    if verbose: print('characteristic_density_local spotty {}'.format(characteristic_density_local))
                else:
                    characteristic_density_local = float(np.sum(np.in1d(subpix, subpix_annulus_region))) \
                                                   / (hp.nside2pixarea(nside_fracdet, degrees=True) * len(subpix_annulus_region)) # deg^-2
                    if verbose: print('characteristic_density_local cleaned up {}'.format(characteristic_density_local))
        else:
            # Compute the local characteristic density
            area_field = np.pi * (0.5**2 - 0.3**2)
            n_field = np.sum((angsep_peak > 0.3) & (angsep_peak < 0.5))
            characteristic_density_local = n_field / area_field
    
            # If not good azimuthal coverage, revert
            cut_annulus = (angsep_peak > 0.3) & (angsep_peak < 0.5) 
            #phi = np.degrees(np.arctan2(y_full[cut_annulus] - y_peak, x_full[cut_annulus] - x_peak)) # Use full magnitude range, NOT TESTED!!!
            phi = np.degrees(np.arctan2(y[cut_annulus] - y_peak, x[cut_annulus] - x_peak)) # Impose magnitude threshold
            h = np.histogram(phi, bins=np.linspace(-180., 180., 13))[0]
            if np.sum(h > 0) < 10 or np.sum(h > 0.5 * np.median(h)) < 10:
                #angsep_peak = np.sqrt((x - x_peak)**2 + (y - y_peak)**2)
                characteristic_density_local = characteristic_density
    
        if verbose: print('Characteristic density local = {:0.1f} deg^-2 = {:0.3f} arcmin^-2'.format(characteristic_density_local, characteristic_density_local / 60.**2))
    
        return(characteristic_density_local)

    def find_peaks(self, iso_sel, distance_modulus, survey, plot=False):
        """
        Convolve field to find characteristic density and peaks within the selected pixel
        """

        #characteristic_density = self.characteristic_density(iso_sel)
        characteristic_density = self.density

        x, y = self.proj.sphereToImage(self.data.basis1[iso_sel], self.data.basis2[iso_sel]) # Trimmed magnitude range for hotspot finding
        #x, y = self.proj.sphereToImage(self.data[self.survey.catalog['basis_1']], self.data[self.survey.catalog['basis_2']]) # If we want to use full magnitude range for significance evaluation
        delta_x = 0.01
        area = delta_x**2
        distance = projector.distanceModulusToDistance(distance_modulus) #kpc
            #100 pc radius typical size -> 0.1 kpc*2
        smoothing = 1./60. #originally 2. arcmin / 60. (converting to deg)
        bins = np.arange(-8., 8. + 1.e-10, delta_x)
        #bins = np.arange(-4., 4. + 1.e-10, delta_x) # SM: not sure what to prefer here...
        centers = 0.5 * (bins[0: -1] + bins[1:])
        yy, xx = np.meshgrid(centers, centers)

        h = np.histogram2d(x, y, bins=[bins, bins])[0]

        h_g = scipy.ndimage.filters.gaussian_filter(h, smoothing / delta_x)

        # SM: If we can speed up this block that would be great
        factor_array = np.arange(1, 5., 0.05)
        rara, decdec = self.proj.imageToSphere(xx.flatten(), yy.flatten())
        cutcut = (hp.ang2pix(self.nside, rara, decdec, lonlat=True) == self.pix_center).reshape(xx.shape)
        threshold_density = 5 * characteristic_density * area
        h_region_l = []
        for factor in factor_array:
            # This is reducing the contrast against the background through the arbitrary measurement 'factor'
            # until there are fewer than 10 disconnected peaks
            h_region, n_region = scipy.ndimage.measurements.label((h_g * cutcut) > (area * characteristic_density * factor))
            h_region_l.append(h_region)
            if n_region < 10:
                threshold_density = area * characteristic_density * factor
                break

        h_region, n_region = scipy.ndimage.measurements.label((h_g * cutcut) > threshold_density)

        x_peak_array = []
        y_peak_array = []
        angsep_peak_array = []

        for index in range(1, n_region + 1): # loop over peaks
            index_peak = np.ravel_multi_index(scipy.ndimage.maximum_position(input=h_g, labels=h_region, index=index), h_g.shape)
            x_peak, y_peak = xx.flatten()[index_peak], yy.flatten()[index_peak]

            # SM: Could these numbers be useful?
            #index_max = scipy.ndimage.maximum(input=h_g, labels=h_region, index=index)
            #index_stddev = scipy.ndimage.standard_deviation(input=h_g, labels=h_region, index=index)
            #print('max: {}'.format(index_max))
            #print('stddev: {}'.format(index_stddev))
            
            #angsep_peak = np.sqrt((x_full - x_peak)**2 + (y_full - y_peak)**2) # Use full magnitude range, NOT TESTED!!!
            angsep_peak = np.sqrt((x-x_peak)**2 + (y-y_peak)**2)
    
            x_peak_array.append(x_peak)
            y_peak_array.append(y_peak)
            angsep_peak_array.append(angsep_peak)

        if plot==True:
            #print(np.shape(h_region), type(h_region[0][0]), h_region[0][0])
            fig, ax = plt.subplots(1,4, figsize=(20,5))
            #numpy documentation tells me to transpose it
            ax[0].imshow(h.T, extent=[bins[0], bins[-1], bins[0], bins[-1]], origin='lower')
            ax[1].imshow(h_g.T, extent=[bins[0], bins[-1], bins[0], bins[-1]], origin='lower')
            ax[2].imshow(h_region.T, extent=[bins[0], bins[-1], bins[0], bins[-1]], origin='lower')
            ax[3].imshow(h_region_l[0].T, extent=[bins[0], bins[-1], bins[0], bins[-1]], origin='lower')
            ax[0].set(xlim = (-.8, .8), ylim = (-.8, .8), title=f'Histogram')
            ax[1].set(xlim = (-.8, .8), ylim = (-.8, .8), title=f'Gaussian filter, {smoothing} deg smooth')
            ax[2].set(xlim = (-.8, .8), ylim = (-.8, .8), title=f'h region after loop')
            ax[3].set(xlim = (-.8, .8), ylim = (-.8, .8), title=f'h region before loop')
            fig.suptitle(f'{distance} kpc')
            for xy in zip(x_peak_array, y_peak_array):
                x, y = xy
                ax[0].scatter(x, y, marker = 'x', s = 10, label=f'{self.proj.imageToSphere(x, y)}')
                ax[1].scatter(x, y, marker = 'x', s = 10, label=f'{self.proj.imageToSphere(x, y)}')
                ax[2].scatter(x, y, marker = 'x', s = 10, label=f'{self.proj.imageToSphere(x, y)}')
                ax[3].scatter(x, y, marker = 'x', s = 10, label=f'{self.proj.imageToSphere(x, y)}')
            ax[1].legend(bbox_to_anchor=(0.5,1.5))
            fig.savefig(plots_dir + f'/{survey}_nside{self.nside}_pixel{self.pixel}/{round(distance)}kpc_region_histogram.png', bbox_inches='tight')
            plt.close()
        
        return x_peak_array, y_peak_array, angsep_peak_array, h_g
    
    def fit_aperture(self, iso_sel, x_peak, y_peak, angsep_peak, distance_modulus, survey, verbose=True, plot=False,extension=None):
        """
        Fit aperture by varying radius and computing the significance
        """
        dist = projector.distanceModulusToDistance(distance_modulus)

        characteristic_density_local = self.characteristic_density_local(iso_sel, x_peak, y_peak, angsep_peak, verbose=verbose)
    
        ra_peak_array = []
        dec_peak_array = []
        r_peak_array = []
        sig_peak_array = []
        n_obs_peak_array = []
        n_obs_half_peak_array = []
        n_model_peak_array = []
    
        
        if extension is not None:
            size_array = extension
        else:
            size_array = np.arange(0.01, 0.3, 0.01)
            
        sig_array = np.zeros(len(size_array))
        
        size_array_zero = np.concatenate([[0.], size_array])
        area_array = np.pi * (size_array_zero[1:]**2 - size_array_zero[0:-1]**2)

        n_obs_array = np.array([np.sum(angsep_peak < size) for size in size_array])
        n_model_array = np.array([characteristic_density_local * (np.pi * size**2) for size in size_array])
        
        sig_array = np.array([np.clip(scipy.stats.norm.isf(scipy.stats.poisson.sf(n_obs, n_model)), 0., 37.5) for (n_obs,n_model) in zip(n_obs_array,n_model_array)])
        sig_array = np.where(n_obs_array==0, 0 ,sig_array)
    
        ra_peak, dec_peak = self.proj.imageToSphere(x_peak, y_peak)
        
        index_peak = np.argmax(sig_array)
        r_peak = size_array[index_peak]
        n_obs_peak = n_obs_array[index_peak]
        n_model_peak = n_model_array[index_peak]
        n_obs_half_peak = np.sum(angsep_peak < (0.5 * r_peak))
        
        if plot==True:
            fig, ax = plt.subplots(1,2, figsize=(15,5))
            ax[0].plot(size_array, n_obs_array, label='N Observed')
            ax[0].plot(size_array, n_model_array, label='N Model = local density*(pi*r^2)')
            ax[0].scatter(r_peak, n_obs_peak, label='n obs peak v r peak', c='k',marker='o')
            ax[0].scatter(r_peak, n_model_peak, label='n model peak v r peak', c='k',marker='x')
            ax[0].set(xlabel='r [deg]', ylabel='number of stars')
            ax[1].plot(size_array, n_obs_array, label='N Observed')
            ax[1].plot(size_array, n_model_array, label='N Model = local density*(pi*r^2)')
            ax[1].scatter(r_peak, n_obs_peak, label='n obs peak v r peak', c='k',marker='o')
            ax[1].scatter(r_peak, n_model_peak, label='n model peak v r peak', c='k',marker='x')
            ax[1].set(xlim=(r_peak-0.001,r_peak+0.001), ylim=(n_model_peak-1,n_obs_peak+1),
                      xlabel='r [deg]', ylabel='number of stars',
                      title=f'n obs: {n_obs_array} \n n mod: {n_model_array} \n sig: {sig_array}')
            ax[0].legend()
            ax[1].legend()
            peak_id = f'{round(sig_array[index_peak],5)}_{round(ra_peak,5)}_{round(dec_peak,5)}_{int(dist)}'
            fig.savefig(plots_dir + f'/{survey}_nside{self.nside}_pixel{self.pixel}/aperture_{peak_id}.png',
                        bbox_inches='tight')
            plt.close()
    
        # Compile results
        if verbose: print('Candidate: x_peak: {:12.3f}, y_peak: {:12.3f}, r_peak: {:12.3f}, sig: {:12.3f}, ra_peak: {:12.3f}, dec_peak: {:12.3f}'.format(x_peak, y_peak, r_peak, np.max(sig_array), ra_peak, dec_peak))
        ra_peak_array.append(ra_peak)
        dec_peak_array.append(dec_peak)
        r_peak_array.append(r_peak)
        #sig_peak_array.append(np.max(sig_array))
        sig_peak_array.append(sig_array[index_peak])
        n_obs_peak_array.append(n_obs_peak)
        n_obs_half_peak_array.append(n_obs_half_peak)
        n_model_peak_array.append(n_model_peak)
    
        return ra_peak_array, dec_peak_array, r_peak_array, sig_peak_array, n_obs_peak_array, n_obs_half_peak_array, n_model_peak_array, characteristic_density_local


#~~~~ retiring these, I think ~~~~~
class Tract():
    def __init__(self, tract, butler):
        '''
        tract = int
        SkyMap = skyMap object, generated from butler
        '''
        self.tract = tract
        self.field = utils.get_field(tract)

        self.butler = butler
        SkyMap =  butler.get('skyMap', skymap=skymap, collections=collection)

        self.center = SkyCoord(SkyMap.generateTract(tract).getCtrCoord().getRa().asDegrees()*u.deg, 
                               SkyMap.generateTract(tract).getCtrCoord().getDec().asDegrees()*u.deg, 
                               frame='icrs')
        self.center_SpherePoint = SkyMap.generateTract(tract).getCtrCoord()
        ras = sorted([SkyMap.getRaDecRange(tract)[0].asDegrees()*u.deg, 
                      SkyMap.getRaDecRange(tract)[1].asDegrees()*u.deg])
        self.ra_range = ras
        decs = sorted([SkyMap.getRaDecRange(tract)[2].asDegrees()*u.deg, 
                       SkyMap.getRaDecRange(tract)[3].asDegrees()*u.deg])
        self.dec_range = decs
        self.corners = [SkyCoord(ra,decs[1],frame='icrs') for ra in ras] + [SkyCoord(ra,decs[0],frame='icrs') for ra in ras[::-1]]
        corners_str = ''
        for coord in self.corners:
            corners_str += f'{coord.ra.value}, {coord.dec.value}, '
        self.corners_str = corners_str.removesuffix(', ')
        self.corners_Angle = SkyMap.getRaDecRange(tract)


# I don't know if I'll want to do a patch class to make these smaller than tract
class Patch(Tract):
    def __init__(self, patch_num):
        self.patch = patch_num
        self.patch_data = self.tract_data[self.tract_data['patch']==patch_num]


