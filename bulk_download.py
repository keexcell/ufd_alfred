import pandas as pd
import numpy as np

valid_pixels = pd.read_csv('euclid_dr1_valid_pixels.csv')
nside_datashard = 32

pix_shard = valid_pixels[f'nside{nside_datashard}']
pix_shard = pix_shard[np.isfinite(pix_shard)]

# euclid columns to query

# can I avoid Region object and just use the euclid query for each pix
# then at runtime i open those files?
# need to create datashards for nside 32
# rewrite to use cone search - might be more efficient?