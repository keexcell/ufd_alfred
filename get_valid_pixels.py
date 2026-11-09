#standard
import gc
import numpy as np
import pandas as pd
#astro specific
from astropy.io import fits
import healpy as hp
#project specific
import ugali.utils.healpix as healpix


#load in Euclid DR1 map to know what's in the footprint
hdu = fits.open('external_data/maps/cgv_map_dr1input_o13.fits.gz')
info = hdu.info()
print(info)
data = hdu[1].data
hdu.close()
map_maybe = data['T'].T
del data
gc.collect()

nside_map = hp.pixelfunc.get_nside(map_maybe)
m = map_maybe[0]
valid_pixels_256, = np.where(m!=0)
valid_pixels_32 = np.unique(healpix.ud_grade_ipix(valid_pixels_256, 256, 32))
valid_pixels_64 = np.unique(healpix.ud_grade_ipix(valid_pixels_256, 256, 64))
valid_pixels_128 = np.unique(healpix.ud_grade_ipix(valid_pixels_256, 256, 128))
df_cols = ['nside32', 'nside64', 'nside128', 'nside256']
df_arrs = [valid_pixels_32, valid_pixels_64, valid_pixels_128, valid_pixels_256]
pix_df = pd.DataFrame(dict(zip(df_cols, map(pd.Series, df_arrs))))
pix_df.to_csv('euclid_dr1_valid_pixels.csv')
