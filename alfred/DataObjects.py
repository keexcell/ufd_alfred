#standard
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
#astro specific
from astropy.table import Table
from astropy import units as u
#project specific
from alfred import utils, masks_and_filters, plotting_functions, search_tools
from ugali.utils import projector

class Data():
    def __init__(self, data, *args, **kwargs):
        self.data = data

    def apply_mask(self, mask):
        ## takes in a mask, applies it to the df, then returns another Data object
        new_data = self.data[mask]
        return Data(new_data)

class Band():
    def __init__(self, val, valerr, name, input_type='flux'):
        if input_type=='flux':
            self.mag = utils.flux2mag(val)
            self.magerr = utils.fluxerr2magerr(val, valerr)
        elif input_type=='mag':
            self.mag = val
            self.magerr = valerr
        self.str = name
        
    def apply_mask(self, mask):
        ## takes in a mask, applies it to the df, then returns another Data object
        new_mag = self.mag[mask]
        new_err = self.magerr[mask]
        return Band(new_mag, new_err, self.str, input_type='mag')

class LSSTData(Data):
    def __init__(self, data, survey='',**kwargs):
        super().__init__(data, **kwargs)
        self.release = survey
        self.survey = survey
        #self.tract = tract
        #self.field = utils.get_field(tract)

        ## coordinates
        self.ra_limits = (data['coord_ra'].min(), data['coord_ra'].max())
        self.dec_limits = (data['coord_dec'].min(), data['coord_dec'].max())
        self.ra = data['coord_ra']
        self.dec = data['coord_dec']
        self.basis1 = data['coord_ra']
        self.basis2 = data['coord_dec']

        ## Rubin bands
        self.g = Band(data['g_psfFlux'], data['g_psfFluxErr'], 'g')
        self.r = Band(data['r_psfFlux'], data['r_psfFluxErr'], 'r')
        self.i = Band(data['i_psfFlux'], data['i_psfFluxErr'], 'i')
        self.z = Band(data['z_psfFlux'], data['z_psfFluxErr'], 'z')
        #then because I have so many functions already defined, some retroactive definitions:
        self.g_mag = self.g.mag
        self.g_magerr = self.g.magerr
        self.r_mag = self.r.mag
        self.r_magerr = self.r.magerr
        self.i_mag = self.i.mag
        self.i_magerr = self.i.magerr
        self.z_mag = self.z.mag
        self.z_magerr = self.z.magerr

    ## morphology
    def band_psfmincmodel(self, band):
        psf_flux = utils.flux2mag(self.data[f'{band}_psfFlux'])
        cmodel_flux = utils.flux2mag(self.data[f'{band}_cModelFlux'])
        return psf_flux - cmodel_flux
    def band_psfdivcmodel(self, band):
        psf_flux = utils.flux2mag(self.data[f'{band}_psfFlux'])
        cmodel_flux = utils.flux2mag(self.data[f'{band}_cModelFlux'])
        return psf_flux / cmodel_flux

    def apply_mask(self, mask):
        ## takes in a mask, applies it to the df, then returns another Data object
        new_data = self.data[mask]
        return LSSTData(new_data, self.survey)

    def clean(self, bands):
        return masks_and_filters.clean_lsst(self.data, bands)        

class EuclidData(Data):
    def __init__(self, data, survey='', **kwargs):
        super().__init__(data, **kwargs)
        self.release = survey
        self.survey = survey #I realized release might be more confusing than survey? 
                                    #will try to fix where I use .release attribute

        ## coordinates
        self.ra = data['RIGHT_ASCENSION']
        self.dec = data['DECLINATION']
        self.basis1 = data['RIGHT_ASCENSION']
        self.basis2 = data['DECLINATION']

        ## Euclid bands (flux given in mu_Jy)
        num = 2 #as suggested in Zerjal et al
        #convert fluxes to nJy, that's what the flux -> mag functions assume
        self.VIS = Band(data[f'FLUX_VIS_2FWHM_APER']*(10**3),
                        data[f'FLUXERR_VIS_2FWHM_APER']*(10**3),
                        'VIS')
        self.H = Band(data[f'FLUX_H_2FWHM_APER']*(10**3),
                      data[f'FLUXERR_H_2FWHM_APER']*(10**3),
                      'H')
        self.Y = Band(data[f'FLUX_Y_2FWHM_APER']*(10**3),
                      data[f'FLUXERR_Y_2FWHM_APER']*(10**3),
                      'Y')
        self.J = Band(data[f'FLUX_J_2FWHM_APER']*(10**3),
                      data[f'FLUXERR_J_2FWHM_APER']*(10**3),
                      'J')
        #then because I have so many functions already defined, some retroactive definitions:
        self.VIS_mag = self.VIS.mag
        self.VIS_magerr = self.VIS.magerr
        self.H_mag = self.H.mag
        self.H_magerr = self.H.magerr
        self.Y_mag = self.Y.mag
        self.Y_magerr = self.Y.magerr
        self.J_mag = self.J.mag
        self.J_magerr = self.J.magerr

        ## morphology
        self.pointlikeprob = data['POINT_LIKE_PROB']
        self.ellipticity = data['ELLIPTICITY']
        self.mumax_minus_mag = self.data['MUMAX_MINUS_MAG']

    def apply_mask(self, mask):
        ## takes in a mask, applies it to the df, then returns another Data object
        new_data = self.data[mask]
        return EuclidData(new_data, self.survey)

    def clean(self, flags, bands = None, fwhm_limit = 1.5):
        return masks_and_filters.clean_euclid(self.data, flags, bands = None, fwhm_limit = 1.5)
        

class DESData(Data):
    def __init__(self, data, survey='',**kwargs):
        super(DESData, self).__init__(data,**kwargs)
        self.release = survey
        self.survey = survey
        
        ## coordinates
        self.ra_limits = (data['alphawin_j2000'].min(), data['alphawin_j2000'].max())
        self.dec_limits = (data['deltawin_j2000'].min(), data['deltawin_j2000'].max())
        self.ra = data['alphawin_j2000']
        self.dec = data['deltawin_j2000']
        self.basis1 = data['alphawin_j2000']
        self.basis2 = data['deltawin_j2000']

        ## DES bands
        self.g = Band(data['psf_mag_aper_8_g_corrected'], data['psf_mag_err_aper_8_g'], 'g', input_type='mag')
        self.r = Band(data['psf_mag_aper_8_r_corrected'], data['psf_mag_err_aper_8_r'], 'r', input_type='mag')
        self.i = Band(data['psf_mag_aper_8_i_corrected'], data['psf_mag_err_aper_8_i'], 'i', input_type='mag')
        self.z = Band(data['psf_mag_aper_8_z_corrected'], data['psf_mag_err_aper_8_z'], 'z', input_type='mag')
        self.y = Band(data['psf_mag_aper_8_y_corrected'], data['psf_mag_err_aper_8_y'], 'y', input_type='mag')
        #then because I have so many functions already defined, some retroactive definitions:
        self.g_mag = self.g.mag
        self.g_magerr = self.g.magerr
        self.r_mag = self.r.mag
        self.r_magerr = self.r.magerr
        self.i_mag = self.i.mag
        self.i_magerr = self.i.magerr
        self.z_mag = self.z.mag
        self.z_magerr = self.z.magerr
        
    def apply_mask(self, mask):
        ## takes in a mask, applies it to the df, then returns another Data object
        new_data = self.data[mask]
        return DESData(new_data, self.survey)

    def clean(self):
        return masks_and_filters.clean_des(self.data)


class LSSTnEuclidData(LSSTData, EuclidData):
    def __init__(self, merged_data, survey='', coord_choice='LSST', **kwargs):
        LSSTData.__init__(self, data=merged_data, survey=survey, **kwargs)
        EuclidData.__init__(self, data=merged_data, survey=survey, **kwargs)

        if coord_choice=='LSST':
            self.ra = merged_data['coord_ra']
            self.dec = merged_data['coord_dec']
        else:
            self.ra = merged_data['RIGHT_ASCENSION']
            self.dec = merged_data['DECLINATION']
        self.coord_choice = coord_choice
        self.release = survey
        self.survey = survey

    def apply_mask(self, mask):
        ## takes in a mask, applies it to the df, then returns another Data object
        new_data = self.data[mask]
        return LSSTnEuclidData(new_data, self.survey, coord_choice=self.coord_choice)

    def clean(self, lsst_bands, euclid_flags, euclid_bands=None, fwhm_limit=1.5, **kwargs):
        '''
        kwargs = lsst_bands, euclid_flags, euclid_bands, fwhm_limit
        '''
        mask = masks_and_filters.clean_lsst(self.data,bands=lsst_bands)
        mask &= masks_and_filters.clean_euclid(self.data,flags=euclid_flags,bands=euclid_bands,fwhm_limit=fwhm_limit)
        return mask


class DESnEuclidData(DESData, EuclidData):
    def __init__(self, merged_data, survey='', coord_choice='DES',**kwargs):
        DESData.__init__(self, data=merged_data, survey=survey,**kwargs)
        EuclidData.__init__(self, merged_data, survey=survey)

        if coord_choice=='DES':
            self.ra = merged_data['alphawin_j2000']
            self.dec = merged_data['deltawin_j2000']
        else:
            self.ra = merged_data['RIGHT_ASCENSION']
            self.dec = merged_data['DECLINATION']
        self.coord_choice = coord_choice
        self.release = survey
        self.survey = survey
  
    def apply_mask(self, mask):
        ## takes in a mask, applies it to the df, then returns another Data object
        new_data = self.data[mask]
        return DESnEuclidData(new_data, survey=self.survey, coord_choice=self.coord_choice)

    def clean(self, des_bands=None, euclid_flags=None, euclid_bands=None, fwhm_limit=1.5, **kwargs):
        '''
        kwargs = euclid_flags, euclid_bands, fwhm_limit
        '''
        mask = masks_and_filters.clean_des(self.data)
        mask &= masks_and_filters.clean_euclid(self.data,flags=euclid_flags,bands=euclid_bands,fwhm_limit=fwhm_limit)
        return mask

        

class Peak():
    def __init__(self, results_T, hist, hist_smooth, iso_starsData, iso, SearchRegion):
        #results_T = ra_peak, dec_peak, r_peak, sig_peak, distance_modulus, n_obs_peak, n_obs_half_peak, n_model_peak
        self.ra = results_T[0]
        self.dec = results_T[1]
        self.r = results_T[2]
        self.sig = results_T[3]
        self.distance_modulus = results_T[4]
        self.distance = projector.distanceModulusToDistance(results_T[4])
        self.samesig_duplicates = []
        self.n_obs = results_T[5]
        self.n_obs_half = results_T[6]
        self.n_model = results_T[7]
        self.isofiltered_hist = hist
        self.isofiltered_histsmooth = hist_smooth
        try:
            self.id = f'{round(self.sig,5)}_{round(self.ra,5)}_{round(self.dec,5)}_{int(self.distance)}'
        except:
            self.id = np.nan
        self.member_candidates = iso_starsData
        self.iso_stars = iso_starsData
        self.iso = iso
        self.region = SearchRegion

    def stars_within_the_radius(self, starsData=None, r=0, scale=0, set_attr = True):
        '''
        scale is to rescale the radius to allow to be more inclusive
        '''
        if starsData is None:
            starsData = self.iso_stars
        angsep = projector.angsep(starsData.ra, starsData.dec, self.ra, self.dec)
        if r != 0:
            sep = r
        elif scale != 0:
            sep = self.r*scale
        radius_mask = (angsep <= sep)
        try:
            member_candidates = starsData.apply_mask(radius_mask)
        except:
            member_candidates = Data(starsData.data[radius_mask])
        print(f'number of the stars within {sep} deg of center = ', len(member_candidates.data))
        if set_attr==True:
            print(f'set member candidates to be within {sep} deg of center')
            self.member_candidates = member_candidates
        return member_candidates
        
    def diagnostic_plots(self, iso_background_stars, all_background_stars, background_r, Region,
                         plots_dir, data_dir, 
                         n_plots = 6, preload=True, save=True, cutouts=True):
        '''
        stars_to_plot - list
        n plots has to be even with this logic I suppose (probably 4 or 6)
        will add axes with shape (2, n_plots/2)
        '''
        n_cols = int(n_plots/2)
        fig = plt.figure(figsize=(17,10))
        spec = gridspec.GridSpec(ncols=n_cols, nrows=2, figure=fig)
        # isochrone plot with just the member_candidates
        ax1 = fig.add_subplot(spec[0,0])
        mem_candidates = self.member_candidates
        if type(mem_candidates)==Data:
            plotting_functions.isochrone_plot([self.iso], {
                f'Stars fit isochrone + within {background_r} deg' : {'data':(iso_background_stars.VIS,iso_background_stars.Y),'markers':[5,'k']},
                f'All stars within {background_r} deg' : {'data':(all_background_stars.VIS,all_background_stars.Y),'markers':'hist'}
                                                          },
                                              title = "No member stars",
                                              legend=True, colorbar=True,
                                              x_lim = (-3, 3), y_lim = (26,16),
                                              save = False, ax = ax1)
        else:
            plotting_functions.isochrone_plot([self.iso], {
                f'All stars within {background_r} deg' : {'data':(all_background_stars.VIS,all_background_stars.Y),'markers':'hist'},
                f'Stars fit isochrone + within {background_r} deg' : {'data':(iso_background_stars.VIS,iso_background_stars.Y),'markers':[5,'k']},
                'Stars fit isochrone + within 1.2*radius deg' : {'data':(mem_candidates.VIS, mem_candidates.Y),'markers':[10,'r']}
                                                          },
                                              legend=True, colorbar=True,
                                              x_lim = (-3, 3), y_lim = (26,16),
                                              save = False, ax = ax1)

        # scatterplot of the stars, radius, center, etc
        #plotting_functions.candidate_scatterplot(self, ax = fig.add_subplot(spec[0,1]),legend=True)
        iso_background_stars.label = f'isochrone stars within {background_r} deg'
        all_background_stars.label = f'all stars within {background_r} deg'
        plotting_functions.candidates_v_background(self, iso_background_stars, ax=fig.add_subplot(spec[0,1]),legend=True)
        plotting_functions.candidates_v_smoothbackground(self, Region, ax=fig.add_subplot(spec[0,2]),legend=True)
        plotting_functions.density_v_r(self, [iso_background_stars, all_background_stars], ax=fig.add_subplot(spec[1,0]),legend=True, mode='circle')

        names = self.iso_stars.survey.lower()

        if cutouts==True:
            parts = names.split('_')
            try:
                euclid_name = parts[parts.index('euclid')] + '_' + parts[parts.index('euclid')+1]
                euclid_pixel_dir = data_dir + '/' + euclid_name + f'/nside{self.region.nside}_pixel{self.region.pixel}'
                if not os.path.exists(euclid_pixel_dir):
                    os.mkdir(euclid_pixel_dir)
            except:
                euclid_name = None
            try:
                des_name = parts[parts.index('des')] + ' ' + parts[parts.index('des')+1]
            except:
                des_name = None
            
            cutout_radius_deg = 0.05
            cutout_radius_arcsec = cutout_radius_deg*3600 #r in deg, convert to arcsec

            if euclid_name is not None:
                cutout_radii = [cutout_radius_arcsec, 120, 60] # this cutout radius is in arcsec
                euclid_file_base = euclid_pixel_dir + f'/{self.ra}_{self.dec}'
                for r in cutout_radii:
                    euclid_file = euclid_file_base + f'_{r}arcsec'
                    if utils.check_if_query(euclid_file, preload)==True:
                        print(f'Check tells me to download Euclid image for {r}arcsec')
                        try:
                            plotting_functions.saveEuclidCutout(euclid_file, self.ra, self.dec, 
                                                                cutout_radius = r)
                            plotting_functions.plotCutout(euclid_file, euclid_name.replace('_',' ').upper(), Peak=self,
                                                  legend=False,subplot=spec[1,1],fig=fig)
                            break
                        except:
                            continue
                        
                    else:
                        print(f'Check tells me to NOT download Euclid image for {r}arcsec')
                        plotting_functions.plotCutout(euclid_file, euclid_name.replace('_',' ').upper(), Peak=self,
                                                  legend=False,subplot=spec[1,1],fig=fig)
                        break
            try:
                #this cutout radius is in deg
                des_file_path = plotting_functions.saveDESCutout(self.ra, self.dec, radius = 0.1)
                plotting_functions.plotCutout(des_file_path, des_name.upper(), Peak=self,
                                              flip_x=True,subplot=spec[1,2],fig=fig)
            except:
                pass

            ## eventually add LSST cutout
        
        fig.suptitle(f"Peak at {round(self.ra,2)},{round(self.dec,2)} deg, r = {round(self.r,2)} deg, dist mod = {round(self.distance_modulus,2)} ({int(self.distance)} kpc), {round(self.sig,2)} sigma", y=1, fontsize=12)
        fig.tight_layout()
        if save == True:
            save_dir = plots_dir + f'/{names}_nside{self.region.nside}_pixel{self.region.pixel}'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            fig.savefig(save_dir + f'/diagnostic_{self.id}.png',dpi=300, bbox_inches = "tight")
        plt.close()

    
#Below methods are to help with formatting things~~~~~~~~~~~~~~~~~~~~~~~
    def make_list(self):
        if len(self.samesig_duplicates)!=0:
            distance = [peak['distance'] for peak in self.samesig_duplicates]
            dist_mod = [peak['distance modulus'] for peak in self.samesig_duplicates]
        else:
            distance = [self.distance]
            distance = [self.distance_modulus]
        values_list = [self.id, self.ra, self.dec, 
                       self.r, self.sig, 
                       distance, distance_modulus, 
                       self.n_obs, self.n_obs_half, self.n_model]
        return values_list

    def make_tuple(self):
        if len(self.samesig_duplicates)!=0:
            distance = [peak['distance'] for peak in self.samesig_duplicates]
            distance_modulus = [peak['distance modulus'] for peak in self.samesig_duplicates]
        else:
            distance = [self.distance]
            distance_modulus = [self.distance_modulus]
        return self.id, self.ra, self.dec, self.r, self.sig, f'{distance}', f'{distance_modulus}', self.n_obs, self.n_obs_half, self.n_model
        
    def list_labels(self, return_type = 'tuple'):
        if return_type == 'tuple':
            labels = ('peak_id', 'ra','dec','r','sig','distance','distance_modulus', 'n_obs', 'n_obs_half', 'n_model')
        elif return_type == 'list':
            labels = ['peak_id', 'ra','dec','r','sig','distance','distance_modulus', 'n_obs', 'n_obs_half', 'n_model']
        elif return_type == 'tuple units':
            labels = ('peak_id','ra_deg','dec_deg','r_deg','sig','distance_kpc','distance_modulus', 'n_obs', 'n_obs_half', 'n_model')
        elif return_type == 'list units':
            labels = ['peak_id','ra_deg','dec_deg','r_deg','sig','distance_kpc','distance_modulus', 'n_obs', 'n_obs_half', 'n_model']
        elif return_type == 'tuple units dtype':
            labels = (('peak_id',str),('ra_deg','f8'),('dec_deg','f8'),('r_deg','f8'),('sig','f8'),('distance_kpc','U'),('distance_modulus','U'), ('n_obs','f8'), ('n_obs_half','f8'), ('n_model','f8'))
        elif return_type == 'list units dtype':
            labels = [('peak_id',str),('ra_deg','f8'),('dec_deg','f8'),('r_deg','f8'),('sig','f8'),('distance_kpc','U'),('distance_modulus','U'), ('n_obs','f8'), ('n_obs_half','f8'), ('n_model','f8')]
        else:
            print('only tuple or list supported')
        return labels
        


        
'''
class LSSTnEuclidData(LSSTData):
    def __init__(self, data, lsst_survey, euclid_survey, field):
        super(LSSTnEuclidData, self).__init__(data, lsst_survey, field)
        self.euclid_survey = euclid_survey
        
        ## coordinates
        self.euclid_ra = data['right_ascension']
        self.euclid_dec = data['declination']
    
        ## Euclid bands (flux given in mu_Jy)
        num = 2 #as suggested in Zerjal et al
        #convert fluxes to nJy, that's what the flux -> mag functions assume
        self.VIS = Band(data[f'FLUX_VIS_{num}FWHM_APER'.lower()]*(10**3), 
                        data[f'FLUXERR_VIS_{num}FWHM_APER'.lower()]*(10**3),
                        'VIS')
        self.H = Band(data[f'FLUX_H_{num}FWHM_APER'.lower()]*(10**3), 
                      data[f'FLUXERR_H_{num}FWHM_APER'.lower()]*(10**3),
                      'H')
        self.Y = Band(data[f'FLUX_Y_{num}FWHM_APER'.lower()]*(10**3), 
                      data[f'FLUXERR_Y_{num}FWHM_APER'.lower()]*(10**3),
                      'Y')
        self.J = Band(data[f'FLUX_J_{num}FWHM_APER'.lower()]*(10**3), 
                      data[f'FLUXERR_J_{num}FWHM_APER'.lower()]*(10**3),
                      'J')
        #then because I have so many functions already defined, some retroactive definitions:
        self.VIS_mag = self.VIS.mag
        self.VIS_magerr = self.VIS.magerr
        self.H_mag = self.H.mag
        self.H_magerr = self.H.magerr
        self.Y_mag = self.Y.mag
        self.Y_magerr = self.Y.magerr
        self.J_mag = self.J.mag
        self.J_magerr = self.J.magerr
        
        ## morphology
        self.pointlikeprob = data['point_like_prob']
        self.ellipticity = data['ellipticity']
        self.mumax_minus_mag = self.data['mumax_minus_mag']
        
    def apply_mask(self, mask):
        ## takes in a mask, applies it to the df, then returns another Data object
        new_data = self.data[mask]
        return LSSTnEuclidData(new_data, self.lsst_survey, self.euclid_survey, self.tract)
'''


    

