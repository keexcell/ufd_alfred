# standard
import yaml
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Annulus
# astro specific
from pyvo.dal import sia
from astropy.utils.data import download_file
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import astropy_mpl_style, ImageNormalize, PercentileInterval, AsinhStretch,SqrtStretch, wcsaxes
#also requires astroquery_updated.esa.Euclid but importing that in the euclid cutout function to avoid unnecessary import if running not with euclid
# project specific
from alfred import utils, DataObjects
from ugali.utils import projector

## ~~~~~~~~~ PLOTS ~~~~~~~~~~~~

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
pad=20
size=1

try:
    with open('config.yaml', 'r') as ymlfile:
    # this is a bit hard coded too but idk another work around
    # if main.py stays same folder as the config then this should work
        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        # assuming that it's cool that the whole github repo is considered "home"
        where = cfg['setup']['where']
        home_dir = os.path.expandvars(cfg['setup']['home_dir'][where])
        output_dir =  os.path.expandvars(cfg['output']['output_dir'])
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        plots_dir = os.path.join(output_dir, cfg['output']['plots_dir'])
        if not os.path.exists(plots_dir):
            os.mkdir(plots_dir)
        if not os.path.exists(plots_dir+'/isochrones'):
            os.mkdir(plots_dir+'/isochrones')
except:
    print('No config file, this will mess up saving')

#~~~~~~~~~START CANDIDATE SCATTERPLOT/IMAGE FUNCTIONS~~~~~~~~~~~~~~~~~~~

def candidates_v_background(Peak, background_stars, ax = None, legend=True, colorbar=True, width=0.0002):
    '''
    Plots a scatterplot and histogram in RA v Dec space of:
        Peak's member candidate stars
        Peak's center (found by find_peaks function) and aperture (with radius found by fit_aperture function)
        2D histogram of background stars

    Parameters
    ----------
    Peak : Peak object (defined in DataObjects), 
        member_candidates attribute already needs to be updated with the desired radius cut
    background_stars : Data object with ra and dec attributes
        stars within the local area of the found peak, to be used for background histogram
    ax : optional, default None, else a matplotlib subplot axes on which this should be plotted
    legend : optional, default True
        whether to display legend describing which are the stars, center, and radius outline
    colorbar : optional, default True
        whether to display histogram colorbar
    width : optional, default 0.0002, else must be another float
        width of the aperture annulus line 
    
    Returns
    -------
    Pretty plot
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    x_lim = (Peak.ra-Peak.r*1.5, Peak.ra+Peak.r*1.5) #(np.min(background_stars.ra), np.max(background_stars.ra))
    y_lim = (Peak.dec-Peak.r*1.5, Peak.dec+Peak.r*1.5) #(np.min(background_stars.dec), np.max(background_stars.dec))
    h = ax.hist2d(background_stars.ra, background_stars.dec, label = background_stars.label, 
                    cmap='viridis', alpha=0.5, bins=50, 
                  range = [[x_lim[0],x_lim[1]],[y_lim[0], y_lim[1]]])
    if type(Peak.member_candidates)==type(background_stars):
        ax.scatter(Peak.member_candidates.ra, Peak.member_candidates.dec,
                   s=50, facecolors='none', edgecolors='r',
                   label='Candidate member stars')
    ax.scatter(Peak.ra, Peak.dec,
               marker='X',c='r',
               label=f'Peak center and radius outline')
    radius = Annulus((Peak.ra, Peak.dec),Peak.r, width, color='r')
    ax.add_artist(radius)
    ax.set(xlabel='RA (deg)', ylabel='Dec (deg)')
    ax.invert_xaxis()
    if legend==True:
        ax.legend(loc='upper right')
    if colorbar==True:
        cbar = plt.colorbar(h[3],ax=ax)
    if ax is None:
    	plt.close()

def candidates_v_smoothbackground(Peak, Region, ax = None, legend=True, colorbar = True, width=0.0002):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    x, y = Region.proj.sphereToImage(Peak.ra, Peak.dec)
    h_smooth = Peak.isofiltered_histsmooth
    delta_x = 0.01
    bins = np.arange(-8., 8. + 1.e-10, delta_x)
    h = ax.imshow(h_smooth.T, extent=[bins[0], bins[-1], bins[0], bins[-1]], origin='lower', label='Smoothed isochrone filtered stellar density')
    ax.scatter(x, y, marker='x', color='r', label='Peak center')
    if legend==True:
        ax.legend(loc='upper right')
    if colorbar==True:
        cbar = plt.colorbar(h[3],ax=ax)
    if ax is None:
    	plt.close()



def saveDESCutout(ra, dec, radius=0.1, band='i'):
    '''
    Fetches a DES DR2 cutout from Noirlab
    Modeled on example:  https://github.com/astro-datalab/notebooks-latest/blob/master/04_HowTos/SiaService/How_to_use_the_Simple_Image_Access_service.ipynb

    Parameters
    ----------
    ra, dec: floats
        center of cutout search
    radius : optional, float in decimal degrees, default 0.1
        radius of the search
    band : optional, string
        band of the desired cutout, default i to match Euclid VIS

    Returns
    -------
    Filename (str), which astropy automatically caches
    '''
    #https://datalab.noirlab.edu/data/dark-energy-survey#des-dr2
    DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia/des_dr2"
    service = sia.SIAService(DEF_ACCESS_URL)
    
    imgTable = service.search((ra,dec), (radius/np.cos(dec*np.pi/180), radius), verbosity=2).to_table()
    #trying to get the table down to just one row
    sel0 = np.char.startswith(imgTable['obs_bandpass'].astype(str),band)
    print("The full image list contains", len(imgTable[sel0]), "entries with bandpass="+band)
    sel = sel0 & ((imgTable['proctype'] == 'Stack') & (imgTable['prodtype'] == 'image')) # basic selection
    Table = imgTable[sel] # select
    
    if (len(Table)>0):
        row = Table[np.argmax(Table['exptime'].data.data.astype('float'))] # pick image with longest exposure time
        url = row['access_url'] # get the download URL
        print ('downloading deepest ' + band + ' image...')
        return download_file(url,cache=True,show_progress=False,timeout=120)
    else:
        return None

def saveEuclidCutout(output_file, ra, dec, cutout_radius=120):
    # https://astroquery.readthedocs.io/en/latest/esa/euclid/euclid.html#mer-cutouts
    """
    Retrieves a FITS image cutout from the Euclid Q1 VIS mosaic for a specific coordinate.

    This function queries the Euclid TAP server for 'VIS' instrument mosaic products 
    that overlap with the specified Right Ascension and Declination. It requires exactly 
    one overlapping file to proceed, and downloads the resulting spatial cutout.

    Parameters
    ----------
    output_file : str
        The local destination file path and name for the downloaded FITS cutout 
        (e.g., 'my_target_cutout.fits').
    ra : float
        Right Ascension of the target center in decimal degrees (ICRS).
    dec : float
        Declination of the target center in decimal degrees (ICRS).
    cutout_radius : float
        The requested radius of the image cutout in arcseconds.

    Raises
    ------
    ValueError
        If the spatial query returns zero files, or more than one overlapping file.

    Notes
    -----
    Requires `astroquery.esa.euclid`, `astropy.coordinates.SkyCoord`, and `astropy.units`.
    """
    from astroquery_updated.esa.euclid import Euclid

    coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
    
    # Step 1: Query file paths
    radius   = 0.5 / 60.
    query    = f"""
    SELECT 
    file_name, file_path, datalabs_path, instrument_name, filter_name, ra, dec, creation_date, product_type, patch_id_list, tile_index 
    FROM 
    dr1.mosaic_product 
    WHERE 
    instrument_name='VIS' AND INTERSECTS(CIRCLE({ra}, {dec}, {radius}), fov)=1 
    ORDER BY 
    file_name
    """
    print('starting query')
    results = Euclid.launch_job_async(query).get_results()
    print(len(results))
    if len(results) > 1:
        #breakpoint()
        #raise ValueError("Expected 1 file and got %i files"%(len(results)))
        print("Expected 1 file and got %i files"%(len(results)))

        # Select the closest file
        coord_results = SkyCoord(ra=results['ra'], dec=results['dec'], unit='deg', frame='icrs')
        results = results[np.argmin(coord.separation(coord_results))]
        file_path  = f"{results['file_path']}/{results['file_name']}"
    elif len(results) == 1:
        file_path  = f"{results['file_path'][0]}/{results['file_name'][0]}"
    else:
        print("Expected 1 file and got %i files"%(len(results)))
        return None

    # Step 2: Get cutout
    cutout_radius = cutout_radius * u.arcsec
    Euclid.get_cutout(
        file_path=file_path,
        coordinate=coord,
        radius=cutout_radius, 
        output_file=output_file,
        verbose=True,
	)

def plotCutout(in_filename, title, Peak=None,
               rotate_ax=False, flip_x=False, ralim=None, declim=None,
               legend=True, subplot=None, fig=None, save=False, out_filename=''):
    '''
    Plots cutout image

    Parameters
    ----------
    in_filename : string
        filepath of already-saved image to plot
    title : string
        title of image
    Peak : default None, else Peak object (defined in DataObjects)
        Peak of which to plot center and member candidates
    rotate_ax : default False
        whether to rotate image (some images have ra and dec axes flipped from our expected)
    flip_x : default False
        whether to invert x-axis (plot ra backwards as is convention)
    ralim : default None, else list
        limits for RA axis in deg, convention says have bigger RA first in list (inverting)
    declim : default None, else list
        limits for dec axis in deg
    legend : default True
        whether to display legend (only necessary if plotting a Peak)
    subplot : default None, else axes object (only tested using gridspec objects)
    fig : default None, else matplotlib figure
        if None will make a new figure with 1 subplot
        else using provided axes to plot a subplot within bigger figure outside of function
    save : default False
        whether to save image plot
    out_filename : default '', else str
        path where to save

    Returns
    -------
    Pretty plot, saves to out_filename
    '''
    
    print('starting to plot cutout')
    hdul = fits.open(in_filename)
    image = hdul[0].data
    hdr = hdul[0].header
    wcs = WCS(hdr)
    if rotate_ax == True:
        image=image.swapaxes(0,1)
        wcs=wcs.swapaxes(0,1)
    hdul.close()
    
    if subplot is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=wcs)
    else:
        ax = fig.add_subplot(subplot, projection=wcs)
        
    im = ax.imshow(image, cmap='gray_r', norm=ImageNormalize(image, interval=PercentileInterval(99.1), vmin=0))#, stretch=AsinhStretch()))

    if Peak is not None:
        center_coord = SkyCoord(ra=Peak.ra*u.deg,dec=Peak.dec*u.deg, frame='icrs')
        center_x, center_y = wcs.world_to_pixel(center_coord)
        ax.scatter(center_x, center_y, marker='x', c='r',label=f'Center of Peak \n at {round(Peak.ra,4)},{round(Peak.dec,4)} deg')
        starsData = Peak.member_candidates
        if type(starsData)==DataObjects.DESnEuclidData:
            catalog_coords = SkyCoord(ra=starsData.ra*u.deg,dec=starsData.dec*u.deg, frame='icrs')
            catalog_x, catalog_y = wcs.world_to_pixel(catalog_coords)
            ax.scatter(catalog_x, catalog_y, s=50, facecolors='none', edgecolors='r', label='Member candidate stars')

    wcsaxes.add_scalebar(ax, 0.01*u.deg, color='C2', label='0.01 deg')
    if legend==True:
        ax.legend(bbox_to_anchor=(1, -0.15))
    ax.set_title(title)
    ax.coords['ra'].set_format_unit(u.deg)
    if flip_x == True:
        ax.invert_xaxis()
    if ralim is not None and declim is not None:
        world_coords = SkyCoord(ra=ralim*u.deg, dec=declim*u.deg, frame='icrs')
        pixel_coords_x, pixel_coords_y = wcs.world_to_pixel(world_coords)
        ax.set_xlim(pixel_coords_x)
        ax.set_ylim(pixel_coords_y)

    #DES Code:   im = ax.imshow(image, cmap='gray',vmin=image.min(), vmax=image.min()+(image.max()-image.min())/100.)
    #ax.scatter(df['ra'], df['dec'], transform=ax.get_transform('icrs'), s=1000./(df['gmag']-12), 
    #           edgecolor='green', facecolor='none', linewidths=3)
    colorbar = plt.colorbar(im)
 
    if save==True:
        plt.savefig(out_filename)
        plt.close()

def density_v_r(Peak, stars_to_plot, mode = 'circle', ax=None, legend=True):
    '''
    Plots line plot of number density (# stars/ area) vs radii from peak center

    Parameters
    -----------
    Peak : Peak object (defined in DataObjects)
        Peak from which to take center coordinates and radius size
    stars_to_plot : list of DataObjects 
        stars of which you want to plot the density
        need to add a label attribute to the objects first
        want the peak stars to NOT be the last entry (preferably background is the last entry)
            (this is because the meshgrid limits for the uniform density test are taken from the last entry)
    mode : optional, default 'circle', else 'annulus'
        method for calculating density
        if 'circle' then the area is taken as everything within the given radius
        if 'annulus' then the area is taken as being within two radii with width of 0.005
    ax : optional, default None, else a matplotlib subplot axes on which this should be plotted
    legend : optional, default True
        whether to display legend
            
    '''
    c_ra, c_dec = Peak.ra, Peak.dec
    aper_r = Peak.r

    # uniform density test as a standard to compare
    mesh_ra = np.linspace(np.min(stars_to_plot[-1].ra),np.max(stars_to_plot[-1].ra))
    mesh_dec = np.linspace(np.min(stars_to_plot[-1].dec),np.max(stars_to_plot[-1].dec))
    unif_ra, unif_dec = np.meshgrid(mesh_ra, mesh_dec)
    uniform_angsep = projector.angsep(c_ra, c_dec, unif_ra, unif_dec)
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
        
    radii = np.linspace(0.015,0.5,50)
    y_max = 0
    for starsData in stars_to_plot+['uniform']:
        if starsData=='uniform':
            angsep = uniform_angsep
        else:
            angsep = projector.angsep(c_ra, c_dec, starsData.ra, starsData.dec)            
        density = []
        for r in radii:
            if mode == 'annulus':
                r2 = r-0.005
                area = np.pi*(r**2) - np.pi*(r2**2)
                mask = (angsep<=r) & (angsep>=r2)
                if radii.index(r)==0:
                    unif_mask = (uniform_angsep<=r) & (uniform_angsep>=r2)
            elif mode == 'circle':
                area = np.pi*(r**2)
                mask = (angsep<=r)
                if np.where(radii==r)[0][0]==0:
                    unif_mask = (uniform_angsep<=r)
            sum_within = np.sum(mask)
            density.append(sum_within / area)

        density_prob = utils.normalize_array(density,radii)
        if np.max(density_prob)>y_max:
            y_max=np.max(density_prob)
        if starsData=='uniform':
            ax.plot(radii, density_prob, label = 'uniform density test', c='k', ls='-')
        else:
            ax.plot(radii, density_prob, label = starsData.label)
        
    ax.plot(np.linspace(aper_r,aper_r,10), np.linspace(0,30,10),
            c='k', ls='--', alpha=0.5, label='radius of the peak aperature')
    ax.set(xlabel = 'radius (deg)', ylabel= 'normalized density = stars/area', xlim=(0,aper_r*3), ylim=(0,y_max))
    if legend==True:
        ax.legend()
    

#~~~~~~~~~~START MAPPING FUNCTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def map_plot(hsp_map, title, color_lims = (24,26),
	     save = True, filename = '', ax = None):
    if ax is None:
    	fig, ax = plt.subplots(figsize=(12, 8))
    sp = skyproj.MollweideSkyproj(ax=ax)
    sp.draw_hspmap(hsp_map, vmin = color_lims[0], vmax = color_lims[1])

    plt.title(title, pad=25)
    plt.colorbar(shrink=0.5)

    if save == True:
        if not os.path.exists(plots_dir + f'/maps'):
            os.mkdir(plots_dir + f'/maps')
        if filename == '':
            filename = title.lower.replace(' ','').replace('-','_').replace(',','_')
        plt.savefig(plots_dir + f'/maps/{filename}.png')
    if ax is None:
    	plt.close()

#~~~~~~~~~~START ISOCHRONE FUNCTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def isochrone_plot(iso_list, stars_band_dict,
                   title = '', legend = True, colorbar = True,
                   x_lim = (-1, 4), y_lim = (28,18), diagnostic=True,
                   save = True, filename = 'isochrone_cmd',
                   ax=None):
    '''
    Plots a CMD (any bands) with isochrone line on top

    Parameters
    ----------
    iso_list : list 
        Isochrone objects to plot, legend label will be distance
    stars_band_dict : dictionary
        formatted as {'legend label of stars data A' : (A band1, A band2), 
                      'legend label of stars data B' : (B band1, B band2),}
        currently only plots up to 3 sets with colors white, blue, red in that order
    title : default '', else other string
        title for the plot
    legend : default True
        whether to display legend
    colorbar : default True
        whether to display colorbar, only relevant if plot is histogram (one dataset plotted)
    xlim, ylim : default (-1,4) and (28,16) respectively, else other tuples of floats
        put larger y value first for flipped mag axes as is convention
    save : default True, decides whether to save the file or not
    filename : default isochrone_cmd, else string
        need to replace if you don't want it to overwrite the isochrone_cmd

    Returns
    -------
    Pretty plot, saves to plots_dir/isochrones/{filename}.png
    '''
    if ax is None:
    	fig, ax = plt.subplots(1,1, figsize=(6,6))
    
    if len(stars_band_dict.keys()) == 1:
        label = list(stars_band_dict.keys())[0]
        band1, band2 = stars_band_dict[label]['data']
        h = ax.hist2d(band1.mag - band2.mag, band1.mag, bins=200,
                      cmin=1, cmap = 'plasma', label = label, alpha = 0.7,
                      range = [[x_lim[0],x_lim[1]],[y_lim[1], y_lim[0]]])
        if colorbar==True:
            plt.colorbar(h[3], ax=ax, label= 'Stellar Sample Number Counts')
        ax.set(xlabel = f'{band1.str} - {band2.str}', ylabel = f'{band1.str}',
               xlim = x_lim, ylim = y_lim, title = title)
        
    else:
        for label in stars_band_dict.keys():
            band1, band2 = stars_band_dict[label]['data']
            markers = stars_band_dict[label]['markers']
            if markers=='hist':
                h = ax.hist2d(band1.mag - band2.mag, band1.mag, bins=200,
                      cmin=1, cmap = 'Greys', label = label, alpha = 0.5,
                      range = [[x_lim[0],x_lim[1]],[y_lim[1], y_lim[0]]])
                if colorbar==True:
                    plt.colorbar(h[3], ax=ax, label= 'Stellar Sample Number Counts')
            else:
                ax.scatter(band1.mag - band2.mag, band1.mag,
                        s=markers[0], c = markers[1], alpha = 1,
                        label = label)
                ax.set(xlabel = f'{band1.str} - {band2.str}', ylabel = f'{band1.str}',
                       xlim = x_lim, ylim = y_lim, title = title)
        
    ls = ['-', '--', ':']
    for i in range(len(iso_list)):
        iso = iso_list[i]
        dist_mod = iso.distance_modulus
        dist = projector.distanceModulusToDistance(dist_mod)
        #index = np.min(np.where(iso.stage == iso.hb_stage)[0]) + 1
        index = np.min(np.where(iso.stage >= iso.hb_stage)[0])
        ax.plot(iso.mag_1[0:index] - iso.mag_2[0:index], iso.mag_1[0:index] + dist_mod,
                color='k', label=f'{round(dist)} kpc', ls=ls[i])
        ax.plot(iso.mag_1[index:] - iso.mag_2[index:], iso.mag_1[index:] + dist_mod,
                color = 'k', ls=ls[i])
        
    if legend==True:
        ax.legend(loc='upper right')

    if save == True:
        if not os.path.exists(plots_dir + f'/isochrones'):
            os.mkdir(plots_dir + f'/isochrones')
        plt.savefig(plots_dir + f'/isochrones/{filename}.png')
    if ax is None: 	
    	plt.close()

#~~~~~~~~~~START MATCH VERIFICATION FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def oneD_hist_matches(match1Band, unmatch1Band, full1Band, match2Band, unmatch2Band, full2Band, SearchRegion, PrimaryData, SecondaryData):
    '''
    Plots 2 one-dimensional histograms of total sources, matched sources, and unmatched sources for Euclid and LSST

    Parameters - all of which come from the merging_catalogs function
    ----------
    match(1/2)Band, unmatch(1/2)Band : Band objects of the matched and unmatched sources from 1=Primary, 2=Secondary surveys
    full(1/2)Band : Band objects of the full data (matched+unmatched) from, again, 1=Primary, 2=Secondary surveys
                Band has attributes mag, magerr, flux, fluxerr, and str
    SearchRegion : Region object, has all the attributes to define where this data is looking
    Primary/SecondaryData : some Data objects (probably a child class), in this context just tells us which data release this is

    Returns
    --------
    1 pretty plot, 2 subplots
    Saves to plots_dir/merged_verification/1dhistmatches_{nside}_{pixel}_{primary survey}_{secondary survey}.png
    '''

    fig, ax = plt.subplots(1,2, figsize=(20,7))
    b = 60

    #one dataset...
    ax[0].hist(match1Band.mag, bins = b, histtype = 'step', color='b', label = 'Matched')
    ax[0].hist(unmatch1Band.mag, bins = b, histtype = 'step', color='r', label = 'Unmatched')
    ax[0].hist(full1Band.mag, bins = b, histtype = 'step', color='k', label = 'All Sources')
    ax[0].set(xlabel=f'{match1Band.str} mag', xlim=(16, 36), ylabel='Number counts', yscale='log',
              title=f'{PrimaryData.release.replace('_',' ').upper()} Sources Match/Unmatch')

    #... then the other
    ax[1].hist(match2Band.mag, bins = b, histtype = 'step', color='b', label = 'Matched')
    ax[1].hist(unmatch2Band.mag, bins = b, histtype = 'step', color='r', label = 'Unmatched')
    ax[1].hist(full2Band.mag, bins = b, histtype = 'step', color='k', label = 'All Sources')
    ax[1].set(xlabel=f'{match2Band.str} mag', xlim=(16, 36), ylabel='Number counts', yscale='log',
              title=f'{SecondaryData.release.replace('_',' ').upper()} Sources Match/Unmatch')

    plt.legend()

    plt.savefig(plots_dir + '/merged_verification/' + f'''1dhistmatches_
                                                          {SearchRegion.nside}_{SearchRegion.pixel}_
                                                          {PrimaryData.release}_{SecondaryData.release}.png''')
    plt.close()

def twoD_hist_matches(SearchRegion, matchPrimaryData, matchSecondaryData):
    '''
    Plots 2 two-dimensional histograms of matched sources for 2 surveys

    Parameters - all of which come from the merging_catalogs function
    ----------
    SearchRegion : Region object, has all the attributes to define where this data is looking
    Primary/SecondaryData : some Data objects (probably a child class) to map the locations of matched sources

    Returns
    --------
    Pretty plot with 2 subplots (one side is the matches in Euclid, other is the matches in LSST)
    Saves to plots_dir/merged_verification/2dhist_{nside}_{pixel}_{primary survey}_{secondary survey}.png
    '''
    fig, ax = plt.subplots(1,2, figsize=(13,5))
    _, _, _, im = ax[0].hist2d(matchPrimaryData.ra, matchPrimaryData.dec, bins=100)
    plt.colorbar(im, ax=ax[0])
    ax[0].set(title = f'Matches in {matchPrimaryData.release.replace('_',' ').upper()}', ylabel = "Dec (deg)", xlabel = "RA (deg)")
    ax[0].invert_xaxis()
    _, _, _, im = ax[1].hist2d(matchSecondaryData.ra, matchSecondaryData.dec, bins=100)
    plt.colorbar(im, ax=ax[1])
    ax[1].set(title = f'Matches in {matchSecondaryData.release.replace('_',' ').upper()}', ylabel = "Dec (deg)", xlabel = "RA (deg)")
    ax[1].invert_xaxis()
    plt.savefig(plots_dir + '/merged_verification/' + '2dhist_' + f'{SearchRegion.nside}_{SearchRegion.pixel}_{matchPrimaryData.release}_{matchSecondaryData.release}.png')
    plt.close()


def source_scatterplot(PrimaryData, SecondaryData, SearchRegion, mag_cut=None, mag_cut_label=None, ra_limits=None, dec_limits = None):
    '''
    Plots 2-D scatterplot of where *all* the source coordinates of the two surveys are, within 0.01x0.01 deg box

    Parameters - all of which come from the merging_catalogs function
    ----------
    Primary/SecondaryData : some Data objects (probably a child class) to map the locations of all sources, to see if they overlap
    SearchRegion : Region object, has all the attributes to define where this data is looking
    mag_cut (optional) : default None, else mask. A mask of a magnitude cut to be applied, e.g. (i.mag > 24)
    mag_cut_label (optional) : default None, else string. Labels what is the magnitude cut for the title
    ra_limits, dec_limits (optional) : default None, else a tuple of floats. The coordinate range you wish to plot (in degrees) (ra should be given backwards if a flipped x-axis is desired)

    Returns
    --------
    Pretty plot
    Saves to plots_dir/merged_verification/2dscatter_{nside}_{pixel}_{primary survey}_{secondary survey}_{mag_cut_label}.png
    '''
    title = f'{PrimaryData.release.replace('_',' ').upper()} and {SecondaryData.release.replace('_',' ').upper()} before matching'
    filename = '2dscatter_' + f'{SearchRegion.nside}_{SearchRegion.pixel}_{PrimaryData.release}_{SecondaryData.release}'
    if mag_cut is not None:
        primarydata = PrimaryData.apply_mask(mag_cut)
        secondarydata = SecondaryData.apply_mask(mag_cut)
        title += f' with magnitude cut {mag_cut_label}'
        filename += f'_{mag_cut_label.maketrans({'<': 'lt', '>': 'gt', ' ': ''})}.png'
    else:
        primarydata = PrimaryData
        secondarydata = SecondaryData
        filename += '.png'

    plt.scatter(primarydata.ra, primarydata.dec,
                marker = '+', label = f'{primarydata.release.replace('_',' ').upper()}'
               )
    plt.scatter(secondarydata.ra, secondarydata.dec,
                marker = 'x', label = f'{secondarydata.release.replace('_',' ').upper()}'
               )
    if ra_limits is None:
        ra_limits = (np.ma.median(primarydata.ra)+0.01, np.ma.median(primarydata.ra))
        dec_limits = (np.ma.median(primarydata.dec),np.ma.median(primarydata.dec)+0.01)
    plt.xlim(ra_limits[0], ra_limits[1])
    plt.xlabel('RA (deg)')
    plt.ylim(dec_limits[0], dec_limits[1])
    plt.ylabel('DEC (deg)')
    #plt.colorbar()
    plt.legend()
    plt.title(title)
    plt.savefig(plots_dir + '/merged_verification/' + filename)
    plt.close()

def coord_diff_hist(merged_df_coord1, merged_df_coord2, PrimaryData, SecondaryData, SearchRegion, ds=None):
    '''
    Plots 2 one-dimensional histograms of the RA and Dec differences (Euclid - LSST for both)

    Parameters - all of which come from the merging_catalogs function
    ----------
    merged_df : Table or other dataframe type, merged catalog (both Euclid and LSST data)
    tract : int, tract these datasets are querying
    Primary/SecondaryData : some Data objects (probably a child class) just need the releases
    SearchRegion : Region object, has all the attributes to define where this data is looking

    Returns
    --------
    1 pretty plot, 2 subplots
    Saves to plots_dir/merged_verification/{coorddiff}_{nside}_{pixel}_{primary survey}_{secondary survey}.png
    '''
    #commented out version of this below
    '''
    #histogram of separation
    ds = ds * 3600 #ds is in degrees, want to plot in arcsecs
    #I forced in the function that matches would be <1"
    plt.hist(ds, histtype='step', range=(0,1))
    plt.xlabel('separation [arcsec]')
    plt.title(f'Tract {tract}: Matched Source Separation')
    plt.tight_layout()
    plt.show()
    '''
    fig, ax = plt.subplots(1,2, figsize=(20,7))
    ra_diff = (merged_df_coord1[0] - merged_df_coord2[0])*3600
    dec_diff = (merged_df_coord1[1] - merged_df_coord2[1])*3600
    
    ax[0].hist(ra_diff, bins = 100)
    ax[0].set(title='Merged Catalog RA difference', xlabel=f'{PrimaryData.release} RA - {SecondaryData.release} RA (arcsec)', xlim=(-1,1))
    
    ax[1].hist(dec_diff, bins = 100)
    ax[1].set(title='Merged Catalog DEC difference', xlabel=f'{PrimaryData.release} DEC - {SecondaryData.release} DEC (arcsec)', xlim=(-1,1))
    
    plt.savefig(plots_dir + '/merged_verification/' + 'coorddiff_' + f'''{SearchRegion.nside}_{SearchRegion.pixel}
                                                                        _{PrimaryData.release}_{SecondaryData.release}.png''')
    plt.close()


def match_validation_plots(match1Band, unmatch1Band, full1Band,
                           match2Band, unmatch2Band, full2Band,
                           merged_df_coord1, merged_df_coord2,
                           matchPrim, matchSecun,
                           SearchRegion, PrimaryData, SecondaryData):
    '''
    All the outputs from the merging_catalogs function going into different validation plots

    Parameters - all of which come from the merging_catalogs function
    ----------
    tract : int, tract these datasets are querying
    lsst_survey : str, label of the survey release from which the data comes, e.g. 'dp2'
    euclid_survey : str, label of the survey release from which the data comes, e.g. 'q1'
    merged_df : Table or other dataframe type, merged catalog (both Euclid and LSST data)
    matches_euclid, unmatched_euclid : numpy arrays of the matched and unmatched sources from Euclid
    matches_lsst, unmatched_lsst : numpy arrays of the matched and unmatched sources from LSST
    lsst_table : Table or other dataframe type, all LSST data from that area 
    euclid_field : Table or other dataframe type, all Euclid data from that area
    ds : degree separation of each source, generated by ugali match function

    Returns
    --------
    Bunch of pretty plots
    Saves to plots_dir/merged_verification/
    '''
    if not os.path.exists(plots_dir + f'/merged_verification'):
        os.mkdir(plots_dir + f'/merged_verification')

    oneD_hist_matches(match1Band, unmatch1Band, full1Band, match2Band, unmatch2Band, full2Band,
                      SearchRegion, PrimaryData, SecondaryData)
    twoD_hist_matches(SearchRegion, matchPrim, matchSecun)
    source_scatterplot(PrimaryData, SecondaryData, SearchRegion, mag_cut=None, mag_cut_label=None)
    coord_diff_hist(merged_df_coord1, merged_df_coord2, PrimaryData, SecondaryData, SearchRegion, ds=None)
    print('Match validation plots ran and saved')


#~~~~~~~~~~START COLOR-MAG FUNCTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def color_magnitude(band1, band2, #mag defined in dataobjects file
                    color_data, title,
                    color_label = None, selection_label = '_nolegend',
                    colors = 'viridis',
                    histogram = False, ax = None,
                    colorbar_limits = (0,1),
                    x_lim = (-1, 4), y_lim = (30, 18),
                    save = False, filename = None):
    
    """
    Plots color-magnitude diagram, band1 vs band1-band2

    Parameters
    ----------
    band1_mag : array or column
        Band magnitude to be plotted on y-axis
    band1_str : string
        Label of band1
    band2_mag : array or column
        Band magnitude to be subtracted for the x-axis
    band2_str : string
        Label of band2
    color_data : data column OR None
        If plot is to be scatterplot or contour, then should be data for the mapping
        If plot is to be histogram, then None
    title : string
        Title of plot/subplot
    color_label (optional) : string, default ''
        Colorbar label
        If color_data is None, color_label won't be used
    selection_label (optional) : string, default '_nolegend'
        Label for legend if desired to annotate which selection was chosen for this plot
        If kept _nolegend, won't display a legend
    colors (optional) : string, default 'viridis'
        Color scheme to use
    histogram (optional) : boolean, default False
        If True, will plot either histogram or contour (depending on colors type)
        If False, will plot scatterplot
    ax (optional) : default None, else plt axes object
        Axes to plot subplots on
        If none, will set subplot axes (1,1)
    x_lim (optional) : tuple, default (-0.1, 0.6)
        Limits of x-axis
    y_lim (optional) : tuple, default (30, 18)
        Limits of y-axis
        Note: should be backwards because of weirdness of magnitude system
    save (optional) : default False
        File names have form 'colormag_{title}.png'
    plots_path (optional) : default None, else string
        Where to save
    filename (optional) : default None, else str
        Name of file to save as

    Returns
    -------
    Pretty plot, saves to plots_dir/colormag/{filename}.png
    """

    if ax == None: #to allow me to have this as a subplot
        fig, axes = plt.subplots(1,1, figsize=(7,5))
        ax = axes
  
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='.*colormapping.*')
        warnings.filterwarnings("ignore", message='.*labels.*')
        if histogram == True:
            if color_data is None:
                h = ax.hist2d(band1.mag - band2.mag, band1.mag, bins=200,
                          range = [[x_lim[0],x_lim[1]],[y_lim[1], y_lim[0]]],
                          cmin=1, cmap = colors)
                _ = h[3]
                ax.set_title(title + f'\n with selection {selection_label}', pad=pad)

            else:
                _ = ax.scatter(band1.mag - band2.mag, band1.mag, s = size,
                               cmap = colors, c = color_data,
                               vmin=colorbar_limits[0], vmax=colorbar_limits[1],
                               label = selection_label)
                x = band1.mag - band2.mag
                y = band1.mag
                sns.kdeplot(x=x, y=y, fill=False, color="k")
                ax.legend()
                ax.set_title(title, pad=pad)

        else:
            _ = ax.scatter(band1.mag - band2.mag, band1.mag, s = size,
                           cmap = colors, c = color_data,
                           vmin=colorbar_limits[0], vmax=colorbar_limits[1],
                           label = selection_label)
            ax.legend()
            ax.set_title(title, pad=pad)
            
        ax.set(xlabel = f"{band1.str} - {band2.str}", ylabel = f"{band1.str}", xlim = x_lim, ylim = y_lim)
        # cleaning up, not displaying everything if not needed
        if color_label is not None:
            plt.colorbar(_, label=color_label, ax=ax)
        plt.tight_layout()
        
        if save == True:
            if not os.path.exists(plots_dir + f'/colormag'):
                os.mkdir(plots_dir + f'/colormag')
            if filename is None:
                filename = title.replace(' ', '').replace('-', '_').lower()
            plt.savefig(plots_dir + f'/colormag/{filename}.png')
        #plt.show() #uncomment to quickly see CMD without making a dir
        plt.close()
        

#~~~~~~~~~~START COLOR-COLOR FUNCTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def color_color(band_list,
                color_data, title,
                color_label = '', selection_label = '_nolegend',
                colors = 'viridis',
                histogram = False, ax = None,
                y_lim = None, x_lim = None,
                save = False, filename = None):
    """
    Plots color-color for arbitrary bands

    Parameters
    ----------
    band_list : list of tuples
        1st entry of tuple is string labeling band
        2nd entries are the band data
    colors : data column OR None
        If plot is to be scatterplot or contour, then should be data for the mapping
        If plot is to be histogram, then None
    title : str
        Plot title
    color_label (optional) : str, default ''
        Colorbar label. If colors is None, color_label won't be used
    selection_label (optional) : str, default '_nolegend'
        Labels how the data was cut, to be displayed in legend
        If kept _nolegend, won't display a legend
    colors (optional) : str, default 'viridis'
        Color scheme to use
    histogram (optional) : boolean, default False
        If True, will plot either histogram or contour (depending on colors type)
        If False, will plot scatterplot
    ax (optional) : default None, else plt axes object
        Axes to plot subplots on
        If none, will set subplot axes (1,1)
    y_lim (optional) : default None, else tuple
        Limits of y-axis
        Found it necessary sometimes to zoom and enhance on LSST selector
    x_lim (optional) : default None, else tuple
        Limits of x-axis
    save (optional) : default False
        File names have form 'colorcolor_{title}.png'
    plots_path : default None, else string
        Where to save
    filename (optional) : default None, else str
        Name of file to save as

    Returns
    -------
    Pretty plot, saves to plots_dir/colorcolor/{filename}.png
    """

    if ax == None:
        fig, axes = plt.subplots(1,1, figsize=(7,6))
        ax = axes

   
    [band1_x, band2_x, band1_y, band2_y] = band_list
    #cleaning out nans because the hist2d doesn't like them
    nanmask = (~np.isnan(band1_x.mag[1])) & (~np.isnan(band2_x.mag[1])) & (~np.isnan(band1_y.mag[1])) & (~np.isnan(band2_y.mag[1]))
    x1_mag = band1_x.mag[1][nanmask]
    x2_mag = band2_x.mag[1][nanmask]
    y1_mag = band1_y.mag[1][nanmask]
    y2_mag = band2_y.mag[1][nanmask]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='.*colormapping.*')
        warnings.filterwarnings("ignore", message='.*labels.*')
        if histogram == True:
            if color_data is None:
            # 2D histograms require colormap which is str type
                ax.hist2d(x1_mag - x2_mag, y1_mag - y2_mag, bins=200,
                          cmin=1, cmap = colors, label = selection_label)
            # otherwise if color dimension is needed for data, it'll plot contours over scatter
            else:
                x=x1_mag - x2_mag
                y=y1_mag - y2_mag
                _ = ax.scatter(x, y, s = size,
                               c = color_data, cmap = colors,
                               label = selection_label)
                sns.kdeplot(x=x, y=y, fill=False, color="k")
        else:
            _ = ax.scatter(x1_mag - x2_mag, y1_mag - y2_mag,
                           c = color_data, vmin=0, vmax=1,
                           label = selection_label,
                           s = size, cmap = colors)

        ax.set_title(title, pad=pad)
        ax.set(ylabel = f'{band1_y.str} - {band2_y.str}', xlabel = f'{band1_x.str} - {band2_x.str}')
        # cleaning up the place, not displayin/changing everything if not needed
        if selection_label[0] != '_':
            ax.legend()
        if color_label != '':
            plt.colorbar(_,label=color_label)
        if y_lim is not None:
            ax.set(ylim=y_lim)
        if x_lim is not None:
            ax.set(xlim=x_lim)
        plt.tight_layout()

        if save == True:
            if not os.path.exists(plots_dir + f'/colorcolor'):
                os.mkdir(plots_dir + f'/colorcolor')
            if filename is None:
                title = title.replace(' ', '').replace('-', '_').lower()
                filename = title
            plt.savefig(plots_dir + f'/colorcolor/{filename}.png')

#~~~~~~~~~~START STAR-GAL MORPHOLOGY FUNCTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def star_gal_sep(xax, yax_sep, #y_label,
                 color_data, title, 
                 colors = 'viridis_r', #x_label = 'LSST i mag', 
                 selection_label = '_nolegend', color_label = None,
                 histogram = False, ax = None, line_plt = None,
                 colorbar_limits = (0,1), y_lim = None, x_lim = (18,28),
                 save = False, filename = None):
    '''
    Plots the morphology across magnitudes to show star-galaxy classifiers
    Shows magnitude where things start getting confused

    Parameters
    ----------
    xax_mag : array or column
        Data for the x-axis, typically an i magnitude
        Assumed to be LSST i mag; if not, change x_label
    yax_sep : array or column
        Data for the y-axis, some morphology measurement
    y_label : string
        Descriptor of the morphology measurement to label y-axis
    colors : data column OR None
        If plot is to be scatterplot or contour, then should be data for the mapping
        If plot is to be histogram, then None
    title : str
        Plot title
    colors (optional) : str, default 'viridis_r'
        Color scheme to use
    x_label : str, default 'LSST i mag'
        Descriptor to label x-axis
    selection_label (optional) : str, default '_nolegend'
        Description of how the data was cut to be displayed in legend
        If kept _nolegend, won't display a legend
    color_label (optional) : str, default None
        Colorbar label
        If None, won't display a colorbar
    histogram (optional) : boolean, default False
        If True, will plot either histogram or contour (depending on colors type)
        If False, will plot scatterplot
    ax (optional) : default None, else plt axes object
        Axes to plot subplots on
        If none, will set subplot axes (1,1)
    colorbar_limits (optional) : tuple, default (0, 1)
        Limits for the colorbar (changes vmin, vmax)
    y_lim (optional) : tuple, default None
        Limits of y-axis
        Found it necessary sometimes to zoom and enhance on LSST selector
    x_lim (optional) : tuple, default (18,28)
        Limits of x-axis
    save (optional) : default False
        If True the file will be saved
    plots_path (optional) : str, default empty
        Where to save
    filename (optional) : default None, else str
        Name of file to save as

    Returns
    -------
    Pretty plot, saves to plots_dir/stargalsep/{filename}.png
    '''
    if ax == None:
        fig, axes = plt.subplots(1,1, figsize=(7,5))
        ax = axes

    # the code will start yelling about the colorbar and labels depending on how you plot it
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='.*colormapping.*')
        warnings.filterwarnings("ignore", message='.*labels.*')
        # if you want to display by histogram/contour plot bc scatter gets oversaturated
        if histogram == True:
            # 2D histograms require colormap which is str type
            if color_data is None:
                ax.hist2d(xax.mag, yax_sep, bins=200,
                          range = [[x_lim[0],x_lim[1]],[y_lim[0], y_lim[1]]],
                          cmin=1, cmap = colors, label = selection_label)
            # otherwise if color dimension is needed for data, it'll plot contours over scatter
            else:
                _ = ax.scatter(xax.mag, yax_sep,
                               c = color_data, vmin=colorbar_limits[0], vmax=colorbar_limits[1],
                               label = selection_label,
                               s = size, cmap = colors)
                sns.kdeplot(x=xax_mag, y=yax_sep, fill=False, color="k")
        # normal scatter plot
        else:
            _ = ax.scatter(xax.mag, yax_sep,
                           c = color_data, vmin=colorbar_limits[0], vmax=colorbar_limits[1],
                           label = selection_label,
                           s = size, cmap = colors)

        ax.set(xlabel = f'{xax.str}', ylabel = 'Sep', xlim = x_lim)
        ax.set_title(title, pad=pad) #title is separate so I can have the pad
        # cleaning up the place, not displaying everything if not needed
        if y_lim is not None:
            ax.set(ylim = y_lim)
        if selection_label[0] != '_':
            ax.legend()
        if line_plt is not None:
            ax.plot(line_plt[0],line_plt[1],line_plt[2])
        if color_label is not None:
            plt.colorbar(_,label=color_label)
        plt.tight_layout()

        if save == True:
            if not os.path.exists(plots_dir + f'/stargalsep'):
                os.mkdir(plots_dir + f'/stargalsep')
            if filename is None:
                title.replace(' ', '').replace('-','_').lower()
                filename = title
            plt.savefig(plots_dir + f'/stargalsep/{filename}.png')


# ~~ slightly out-dated/useless Peak star scatterplot function ~~
def candidate_scatterplot(Peak, ax = None, legend=True):
    '''
    Plots a scatterplot in RA v Dec space of member candidates of a Peak
    Somewhat out-dated now that I wrote more useful candidates_v_background

    Parameters
    ----------
    Peak : a Peak object (defined in DataObjects), 
            member_candidates attribute already needs to be updated with the desired radius cut
    ax, optional : default None, else a matplotlib subplot axes on which this should be plotted
    legend, optional : default True, whether to display legend describing which are the stars, center, and radius outline
    
    Returns
    -------
    Pretty plot
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    stars = Peak.member_candidates
    ax.scatter(stars.ra,stars.dec,
               s=50, facecolors='none', edgecolors='C2',
               label='Candidate member stars')
    ax.scatter(Peak.ra, Peak.dec,
               marker='X',c='r',
               label=f'Peak center and radius outline')
    radius = Annulus((Peak.ra, Peak.dec),Peak.r, width, color='r')
    ax.add_artist(radius)
    ax.set(xlabel='RA (deg)', ylabel='Dec (deg)',
           xlim=(Peak.ra-Peak.r*1.2, Peak.ra+Peak.r*1.2),ylim=(Peak.dec-Peak.r*1.2, Peak.dec+Peak.r*1.2))
    if legend==True:
        ax.legend(loc='upper right')
    ax.invert_xaxis()
    if ax is None:
    	plt.close()

## ~~~~~ very outdated function trying to automate doing CMD subplots ~~~~
def color_magnitude2(star_df, pltL_dict, pltR_dict, title = None, save = False, filename = None):
    """
    Plots 2 subplots of color-magnitude diagrams, band1 vs band1-band2

    Parameters
    ----------
    star_df : Astropy Table
        Stellar catalog
    pltL/R_dict : dictionary
        Params for left/right subplot
        {band1 : , band2 : , colors : , color_label : , title : }
        See color_magnitude for definitions of above
    title (optional) : default None, else string,
        Super title for plot
    save (optional) : default False
        File names have form 'colormag_{title}.png'
    plots_path (optional) : default None, else string
        Where to save

    Returns
    -------
    Pretty plot
    """

    fig, axes = plt.subplots(1,2, figsize = (14,5))
    axes = axes.flatten()
    dict_list = [pltL_dict, pltR_dict]

    for i, ax in enumerate(axes):
        color_magnitude(star_df,
                        dict_list[i]['band1'], dict_list[i]['band2'],
                        dict_list[i]['colors'], dict_list[i]['color_label'], dict_list[i]['title'],
                        ax = ax)

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if save == True:
        if filename is None:
            title = title.replace(' ', '').replace('-', '_').lower()
            filename = title
        plt.savefig(plots_path + f'/colormag/{filename}.png')
