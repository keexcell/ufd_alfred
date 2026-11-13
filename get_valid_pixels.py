import sys
import numpy as np
import pandas as pd

mode = int(sys.argv[1])
if mode==0:
    from astropy.io import fits
    import healpy as hp
    import ugali.utils.healpix as healpix
    import numpy as np
    import pandas as pd
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
    NESTvalid_pixels_256, = np.where(m!=0)
    valid_pixels_256 = hp.nest2ring(nside_map, NESTvalid_pixels_256)
    valid_pixels_32 = np.unique(healpix.ud_grade_ipix(valid_pixels_256, 256, 32))
    valid_pixels_64 = np.unique(healpix.ud_grade_ipix(valid_pixels_256, 256, 64))
    valid_pixels_128 = np.unique(healpix.ud_grade_ipix(valid_pixels_256, 256, 128))
    df_cols = ['nside32', 'nside64', 'nside128', 'nside256']
    df_arrs = [valid_pixels_32, valid_pixels_64, valid_pixels_128, valid_pixels_256]

    ring_df = pd.DataFrame(dict(zip(df_cols, map(pd.Series, df_arrs))))
    ring_df.to_csv('ufd_alfred/RINGeuclid_dr1_valid_pixels.csv')

elif mode==1:
    
    nest_ordering = pd.read_csv('ufd_alfred/NESTeuclid_dr1_valid_pixels.csv')
    nest_256 = nest_ordering['nside256'].to_numpy(dtype='int')
    ring_256 = hp.nest2ring(256,nest_256)
    nside_arr = [32,64,128]
    df_cols = ['nside256']
    df_arrs = [ring_256]
    for nside in nside_arr:
        ring_pix = np.unique(healpix.ud_grade_ipix(ring_256, 256, nside))
        df_cols.append(f'nside{nside}')
        df_arrs.append(ring_pix)

    ring_df = pd.DataFrame(dict(zip(df_cols, map(pd.Series, df_arrs))))
    ring_df.to_csv('ufd_alfred/RINGeuclid_dr1_valid_pixels.csv')

elif mode==2:
    pix_csv = pd.read_csv('RINGeuclid_dr1_valid_pixels.csv')
    nside32 = pix_csv['nside32'].dropna()
    new_df = pd.DataFrame({'nside' : np.linspace(32,32,len(nside32)), 'pix' : nside32})
    df_string = new_df.to_string(header=False, index=False)
    with open('ALL_euclid_dr1_ring_pixels.txt','w') as f:
        f.write(df_string)

