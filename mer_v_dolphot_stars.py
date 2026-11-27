from astropy.table import Table, join, hstack
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ugali.utils.projector as projector
from ugali.utils.projector import match
from astropy.io import fits
from dustmaps.config import config
config.reset()
config['data_dir'] = 'dustmaps'
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
from simple_adl.simple_adl import isochrone
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm


sfd = SFDQuery()
R_dict = {'VIS':1.83,'Y':0.98,'J':0.60,'H':0.31}

def flux2mag(flux):
    flux = flux*(10**3)
    zeropoint = 31.4
    flux = np.ma.filled(np.ma.asarray(flux, dtype=float), fill_value=np.nan)
    #flux = np.asarray(flux, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = -2.5 * np.log10(flux) + zeropoint
    mag[~np.isfinite(mag)] = np.nan
    #if index is not None:
    #    return pd.Series(mag, index=index)
    return mag

def fluxerr2magerr(flux, flux_err):
    flux = flux*(10**3)
    flux_err = flux_err*(10**3)
    flux = np.ma.filled(np.ma.asarray(flux, dtype=float), fill_value=np.nan)
    #index = flux.index if hasattr(flux, 'index') else None
    #flux = np.asarray(flux, dtype=float)
    #flux_err = np.asarray(flux_err, dtype=float)
    flux_err = np.ma.filled(np.ma.asarray(flux_err, dtype=float), fill_value=np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        magerr = (2.5 / np.log(10)) * (flux_err / flux)
    magerr[~np.isfinite(magerr)] = np.nan
    #if index is not None:
    #    return pd.Series(mag, index=index)
    return magerr

def flux2correctedmag(flux, ra, dec, band):
    coord = SkyCoord(ra, dec, frame='icrs')
    mag = flux2mag(flux)
    ebv = sfd(coord)
    R = R_dict[band]
    mag_0 = mag - R*ebv
    return mag_0
    
def radius_cut(data, ra_col, dec_col, r, center = (65.10, -32.83)):
    '''
    scale is to rescale the radius to allow to be more inclusive
    '''
    angsep = projector.angsep(data[ra_col], data[dec_col], center[0], center[1])
    radius_mask = (angsep <= r)
    return data[radius_mask]

def square_cut(data, ra_col, dec_col, length, center = (65.10, -32.83)):
    '''
    length being 4' if it's a 4'x4' cutout
    '''
    length_deg = length/60
    half_l_deg = length_deg/2
    ra_mask = (data[ra_col] <= center[0]+half_l_deg) & (data[ra_col] >= center[0]-half_l_deg)
    dec_mask = (data[dec_col] <= center[1]+half_l_deg) & (data[dec_col] >= center[1]-half_l_deg)
    square_mask = ra_mask & dec_mask
    return data[square_mask]

def iso_plot(isox, isoy, ax, label='_nolabel'):
    indexx = np.min(np.where(isox.stage >= isox.hb_stage)[0])
    indexy = np.min(np.where(isoy.stage >= isoy.hb_stage)[0])
    ax.plot(isox.mag_1[0:indexx] - isox.mag_2[0:indexx], isoy.mag_1[0:indexy] - isoy.mag_2[0:indexy],
            color='k', ls='-', label=label)

def merge_col(data1_mergeinto, data2_mergefrom, indices1, indices2, col_oldname, col_newname):
    old_shape = np.shape(data1_mergeinto)
    new_col = np.full(len(data1_mergeinto),np.nan)
    new_col[indices1] = data2_mergefrom[indices2][col_oldname]
    data1_mergeinto[col_newname] = new_col
    assert old_shape == np.shape(data1_mergeinto)
    return data1_mergeinto
    

# OPENING FILES
with fits.open('mer_catalog_v1.fits') as hdul:
    mer_data_before = Table.read(hdul)
dolphot_data_before = Table.read('eri6_1deg_2pass_dolphot.parquet', format='parquet')
with fits.open('ugali_euclid_stars_v2.fits') as hdul:
    ugali_data_before = Table.read(hdul)
# DEFINE ALL THE MAGS AND DUST CORRECT
for band in ['Y','H','J']:
    mer_data_before[f'MAG0_{band}'] = flux2correctedmag(mer_data_before[f'FLUX_{band}_TEMPLFIT'],
                            mer_data_before['RIGHT_ASCENSION'], mer_data_before['DECLINATION'], band)
    mer_data_before[f'MAGERR_{band}'] = fluxerr2magerr(mer_data_before[f'FLUX_{band}_TEMPLFIT'], mer_data_before[f'FLUXERR_{band}_TEMPLFIT'])
    mer_data_before[f'MAG0_VIS_TO_{band}'] = flux2correctedmag(mer_data_before[f'FLUX_VIS_TO_{band}_TEMPLFIT'], 
                            mer_data_before['RIGHT_ASCENSION'], mer_data_before['DECLINATION'], 'VIS')
    mer_data_before[f'MAGERR_VIS_TO_{band}'] = fluxerr2magerr(mer_data_before[f'FLUX_VIS_TO_{band}_TEMPLFIT'], mer_data_before[f'FLUXERR_VIS_TO_{band}_TEMPLFIT'])


# APPLY SQUARE CUT 4x4 ARCMIN
l = 4 #arcmin
#mer_data = radius_cut(mer_data_before, 'RIGHT_ASCENSION', 'DECLINATION', r/60)
mer_data = square_cut(mer_data_before, 'RIGHT_ASCENSION', 'DECLINATION', l)
#dolphot_data = radius_cut(dolphot_data_before, 'ra', 'dec', r/60)
dolphot_data = square_cut(dolphot_data_before, 'ra', 'dec', l)
ugali_data = square_cut(ugali_data_before, 'RA', 'DEC', l)
#print(f'objects within {r} arcmin: ', len(mer_data),len(dolphot_data))
print(f"objects within {l}'x{l}' cutout: mer = ", len(mer_data), ", dolphot = ", len(dolphot_data), ', ugali = ', len(ugali_data))
#print(dolphot_data.colnames)
#print(ugali_data.colnames)

# MATCH UGALI MEMBERSHIP, POSITIONS, & MAG DATA
mer_ugali_indices, ugali_mer_indices, ds = match(mer_data['RIGHT_ASCENSION'], mer_data['DECLINATION'],
                                                 ugali_data['RA'], ugali_data['DEC'],
                                                 tol = 0.0003)
dol_ugali_indices, ugali_dol_indices, ds = match(dolphot_data['ra'], dolphot_data['dec'],
                                                 ugali_data['RA'], ugali_data['DEC'],
                                                 tol = 0.0003)
mer_data = merge_col(mer_data, ugali_data, mer_ugali_indices, ugali_mer_indices, 'PROB', 'UGALI_PROB')
dolphot_data = merge_col(dolphot_data, ugali_data, dol_ugali_indices, ugali_dol_indices, 'PROB', 'UGALI_PROB')
dolphot_data = merge_col(dolphot_data, ugali_data, dol_ugali_indices, ugali_dol_indices, 'RA', 'UGALI_RA')
dolphot_data = merge_col(dolphot_data, ugali_data, dol_ugali_indices, ugali_dol_indices, 'DEC', 'UGALI_DEC')
ugali_data = merge_col(ugali_data, mer_data, ugali_mer_indices, mer_ugali_indices, 'MAG0_H', 'H0')
ugali_data = merge_col(ugali_data, mer_data, ugali_mer_indices, mer_ugali_indices, 'MAGERR_H', 'err_H')
'''
try:
    assert np.array_equal(dolphot_data['ra'], dolphot_data['UGALI_RA'], equal_nan=True)
    assert np.array_equal(dolphot_data['dec'], dolphot_data['UGALI_DEC'], equal_nan=True)
    print('Exactly the same')
except:
    try:
        assert np.allclose(dolphot_data['ra'], dolphot_data['UGALI_RA'], equal_nan=True)
        assert np.allclose(dolphot_data['dec'], dolphot_data['UGALI_DEC'], equal_nan=True)
        print('Close')
    except:
        fig, ax = plt.subplots(1,1)
        ax.scatter(dolphot_data['ra'],dolphot_data['dec'], marker='x', c='C0', label='All DOLPHOT')
        ax.scatter(dolphot_data['UGALI_RA'],dolphot_data['UGALI_DEC'], marker='x', c='C1', label='Matched UGALI-DOLPHOT')
        ax.scatter(ugali_data['RA'], ugali_data['DEC'], marker='o', c='C2', label='All UGALI')
        ax.legend()
        fig.savefig('check_dolphot_match')
'''
# CHECKING SPATIAL REGION
#this plot looks weird, but the MER data is a giant 2 deg circle around the pixel,
    #then it's cut down to 1 arcmin circle around center of Eri VI
fig, axes = plt.subplots(2,2)
axes[0,0].scatter(mer_data_before['RIGHT_ASCENSION'],mer_data_before['DECLINATION'], c='C1', s=5)
axes[0,0].set(xlabel='RA', ylabel='Dec', title='MER before')
axes[0,1].scatter(dolphot_data_before['ra'],dolphot_data_before['dec'], c='C0', s=5)
axes[0,1].set(xlabel='RA', ylabel='Dec', title='DOLPHOT before')
axes[1,0].scatter(mer_data['RIGHT_ASCENSION'],mer_data['DECLINATION'], c='C1', s=5)
axes[1,0].set(xlabel='RA', ylabel='Dec', title='MER after')
axes[1,1].scatter(dolphot_data['ra'],dolphot_data['dec'], c='C0', s=5)
axes[1,1].set(xlabel='RA', ylabel='Dec', title='DOLPHOT')
fig.suptitle('MER v DOLPHOT spatial cut')
fig.savefig('mer_dolphot_radec')
plt.close(fig=fig)


# VALIDATION PLOT DUST CORRECTED VS NOT
x_lim = (-1.5,1.5)
y_lim = (27,18)
fig, axes = plt.subplots(2,2)
no_dust_iminj = flux2mag(mer_data['FLUX_VIS_TO_J_TEMPLFIT']) - flux2mag(mer_data['FLUX_J_TEMPLFIT'])
no_dust_i = flux2mag(mer_data['FLUX_VIS_TO_J_TEMPLFIT'])
w_dust_iminj = mer_data['MAG0_VIS_TO_J'] - mer_data['MAG0_J']
w_dust_i = mer_data['MAG0_VIS_TO_J']
axes[0,0].hist2d(no_dust_iminj, no_dust_i,
                cmin=1, bins=100, range = [[x_lim[0],x_lim[1]],[y_lim[1], y_lim[0]]])
axes[0,1].hist2d(w_dust_iminj, w_dust_i,
               cmin=1, bins=100,range = [[x_lim[0],x_lim[1]],[y_lim[1], y_lim[0]]])
axes[0,0].set(xlabel = 'I - J', ylabel = 'I', title = f'No dust correction', ylim=y_lim)
axes[0,1].set(xlabel = 'I - J', ylabel = 'I', title = f'With dust correction', ylim=y_lim)
axes[1,0].hist(no_dust_iminj - w_dust_iminj,
               bins=100)
axes[1,1].hist(no_dust_i - w_dust_i,
                 bins=100)
axes[1,0].set(xlabel = '$(I - J) - (I - J)_0$', title = f'Color difference')
axes[1,1].set(xlabel = '$I - I_0$', title = f'Mag difference')
fig.savefig('mer_dust_correction')
plt.close(fig=fig)

# MER COLOR-COLOR STELLAR CUT
color_color_x = mer_data['MAG0_VIS_TO_H'] - mer_data['MAG0_H']
color_color_y = mer_data['MAG0_Y'] - mer_data['MAG0_H']
color_color_space = pd.DataFrame({'I-H' : color_color_x, 'Y-H' : color_color_y}).to_numpy()
poly = Path([(-1.5,-1.5),(-1.5,-0.4),(0.9,0.8),(1.4,0.8),(1.4,-0.4),(0.9,-0.4)],closed=True)
color_mask = poly.contains_points(color_color_space)
# POINTLIKE STELLAR CUT
point_like_mask = (mer_data['POINT_LIKE_PROB']>0.05)
# APPLY COLORCOLOR AND POINTLIKE STELLAR MASK
star_mask = point_like_mask & color_mask
# QUALITY MASK - enforcing detections
for col in ['FLUX_VIS_TO_J_TEMPLFIT', 'FLUX_H_TEMPLFIT', 'FLUX_J_TEMPLFIT']:
    star_mask &= (np.isfinite(mer_data[col]))
    star_mask &= (mer_data[col]>0)&(mer_data[col]<99.98)
    star_mask &= (mer_data[col]>0)&(mer_data[col]<99.98)
for band in ['VIS_TO_H','VIS_TO_J', 'J','H']:
    star_mask &= (~np.isnan(mer_data[f'MAG0_{band}']))
mer_stars = mer_data[star_mask]
# REDEFINE ALL MAGS AND ERRS (DUST CORRECTED)
mer_stars_Y, mer_stars_Y_err = mer_stars['MAG0_Y'], mer_stars['MAGERR_Y']
mer_stars_J, mer_stars_J_err = mer_stars['MAG0_J'], mer_stars['MAGERR_J']
mer_stars_H, mer_stars_H_err = mer_stars['MAG0_H'], mer_stars['MAGERR_H']
mer_stars_IH, mer_stars_IH_err = mer_stars['MAG0_VIS_TO_H'], mer_stars['MAGERR_VIS_TO_H']
mer_stars_IJ, mer_stars_IJ_err = mer_stars['MAG0_VIS_TO_J'], mer_stars['MAGERR_VIS_TO_J']
print(len(mer_stars['MAG0_J']), np.max(mer_stars['MAG0_J']), np.min(mer_stars['MAG0_J']))
print(mer_stars['MAG0_J'])
print(len(mer_stars['MAG0_H']), np.max(mer_stars['MAG0_H']), np.min(mer_stars['MAG0_H']))
print(mer_stars['MAG0_H'])
print(len(mer_stars['MAG0_VIS_TO_H']), np.max(mer_stars['MAG0_VIS_TO_H']), np.min(mer_stars['MAG0_VIS_TO_H']))
print(mer_stars['MAG0_VIS_TO_H'])
print(len(mer_stars['MAG0_VIS_TO_J']), np.max(mer_stars['MAG0_VIS_TO_J']), np.min(mer_stars['MAG0_VIS_TO_J']))
print(mer_stars['MAG0_VIS_TO_J'])
x, y = mer_stars['MAG0_J'], mer_stars['MAG0_H']   # whatever you're actually plotting
z, w = mer_stars['MAG0_VIS_TO_H'], mer_stars['MAG0_VIS_TO_J']
ok = np.isfinite(x) & np.isfinite(y)
print(ok.sum(), "finite of", len(mer_stars))
print("x nonfinite:", (~np.isfinite(x)).sum(), "| y nonfinite:", (~np.isfinite(y)).sum())
for col in ['FLUX_VIS_TO_J_TEMPLFIT', 'FLUX_H_TEMPLFIT', 'FLUX_J_TEMPLFIT']:
    print(f"{col} <= 0:", (mer_stars[col] <= 0).sum())

# VALIDATION PLOT OF MER STELLAR SELECTION
x_lim = (-2,2)
y_lim = (-2,2)
fig, axes = plt.subplots(1,2)
axes[0].hist2d(color_color_x, color_color_y, cmin=1,
                bins=100, range = [[x_lim[0],x_lim[1]],[y_lim[0], y_lim[1]]])
axes[1].hist2d(mer_stars_IH - mer_stars_H, mer_stars_Y - mer_stars_H, cmin=1,
                bins=100,range = [[x_lim[0],x_lim[1]],[y_lim[0], y_lim[1]]])
patch0 = patches.PathPatch(poly, fill=False, edgecolor ='k', lw=2)
patch1 = patches.PathPatch(poly, fill=False, edgecolor ='k', lw=2)
axes[0].add_patch(patch0)
axes[1].add_patch(patch1)
axes[0].set(xlabel = 'FLUX_VIS_TO_H_TEMPLFIT - FLUX_H_TEMPLFIT mags', ylabel = 'FLUX_Y_TEMPLFIT - FLUX_H_TEMPLFIT mags', 
            title = f"MER objects within {l}'x{l}' cutout\n{len(color_color_x)} objs")
axes[1].set(xlabel = 'FLUX_VIS_TO_H_TEMPLFIT - FLUX_H_TEMPLFIT mags', ylabel = 'FLUX_Y_TEMPLFIT - FLUX_H_TEMPLFIT mags', 
            title = f"MER stars within {l}'x{l}' cutout\n{len(mer_stars_IH)} objs")
fig.savefig('mer_colorcolor_stellar_selection')
plt.close(fig=fig)


# DOLPHOT STELLAR SELECTION
stars_mask = (dolphot_data['star_complete']==1)
for band in ['I','J','H']:
    stars_mask &= (dolphot_data[f'mag_{band}']<99)
dolphot_stars = dolphot_data[stars_mask]
# DEFINE DUST CORRECTED MAGNITUDES AND ERRS
dol_stars_I, dol_stars_I_err = dolphot_stars['mag0_I'], dolphot_stars['err_I']
dol_stars_H, dol_stars_H_err = dolphot_stars['mag0_H'], dolphot_stars['err_H']
dol_stars_J, dol_stars_J_err = dolphot_stars['mag0_J'], dolphot_stars['err_J']
# DOLPHOT STARS MATCHED TO MER
dol_matched_mer = dolphot_stars[dolphot_stars['mer_object_id']!=0]
print('Length of DOLPHOT with non-0 MER object ID = ', len(dolphot_stars[dolphot_stars['mer_object_id']!=0]))
dol_matched_mer = join(dolphot_stars, mer_stars, keys_left='mer_object_id', keys_right='OBJECT_ID', join_type='inner')
print('Length of matched on ID table', len(dol_matched_mer))

#print(dolphot_stars.colnames)
#print(mer_stars.colnames)
#print(dol_matched_mer.colnames)
dolmer_I, dolmer_I_err = dol_matched_mer['mag0_I'], dol_matched_mer['err_I']
dolmer_H, dolmer_H_err = dol_matched_mer['mag0_H'], dol_matched_mer['err_H']
dolmer_J, dolmer_J_err = dol_matched_mer['mag0_J'], dol_matched_mer['err_J']
assert np.allclose(dol_matched_mer['UGALI_PROB_1'], dol_matched_mer['UGALI_PROB_2'], equal_nan=True)
dol_matched_mer.rename_column('UGALI_PROB_1', 'UGALI_PROB')

print(f"Number of selected stars within {l}'x{l}' cutout: mer = ", len(mer_stars), ", dolphot = ", len(dolphot_stars))


# VIS - J ON X, VIS - H ON Y FOR ALL STARS WITHIN 4 ARCMIN CUTOUT
plt.style.use('plots.mplstyle')
fig, axes = plt.subplots(1,2, figsize=(15,10))
size = 50
mercolor = '#d84727'
paler_mercolor = '#deb49e'
pale_mercolor = '#df9877'
greyish = '#d6d1ca'
mer_cmap = LinearSegmentedColormap.from_list('mycmap', [greyish, paler_mercolor, pale_mercolor, '#ef7b45', '#ea6e3e', mercolor,'#a01800'])
dolcolor = '#042a2b'
pale_dolcolor = '#96cfdb'
paler_dolcolor = '#cdedf6'
dol_cmap = LinearSegmentedColormap.from_list('mycmap', [greyish, paler_dolcolor, pale_dolcolor, '#5eb1bf', '#48909a', dolcolor])
def norm_prob(data): return 0.2 + 0.8 * data['UGALI_PROB'] 

#axes 1: mer stars
ax0 = axes[0].scatter(mer_stars_IJ - mer_stars_J, mer_stars_IH - mer_stars_H,
                      c=mer_stars['UGALI_PROB'], cmap=mer_cmap, 
                      s=size)
axes[0].set(xlabel='$I_E - J_E$',ylabel='$I_E - H_E$', 
            title=f'MER stars ({len(mer_stars)})')#\ncolor selection, plp>0.05, detection I, J, H\n({len(mer_stars_IJ)} stars)')
#axes 2: dolphot stars with a mer match outline
axes[1].scatter(dol_stars_I - dol_stars_J, dol_stars_I - dol_stars_H,
                c=dolphot_stars['UGALI_PROB'], cmap=dol_cmap,
                s=size,
                label='All DOLPHOT stars')
ax1 = axes[1].scatter(dolmer_I - dolmer_J, dolmer_I - dolmer_H,
                      c = dol_matched_mer['UGALI_PROB'], cmap=dol_cmap, edgecolors=mercolor,
                      s=size+10, linewidths=1, label='DOLPHOT stars matched with MER')
axes[1].set(xlabel='$I_E - J_E$',ylabel='$I_E - H_E$',
            title = f'DOLPHOT ({len(dolphot_stars)})') #\nstar_complete, detection I, J, H\n({len(dol_stars_I)} stars)')
#add isochrone in vis-j, vis-h space
Z = 0.0001
age = 12.0
dist_mod = 21.63
iso1 = isochrone.Isochrone(age=age, metallicity=Z, distance_modulus=dist_mod,
                            survey='mixed',
                            band_1= 'VIS',
                            band_2= 'J')
iso2 = isochrone.Isochrone(age=age, metallicity=Z, distance_modulus=dist_mod,
                            survey='mixed',
                            band_1= 'VIS',
                            band_2= 'H')
iso_plot(iso1, iso2, axes[0], label=f'Marigo2017 isochrone\nage={age} Gyr, Z={Z}, $(m-M)_0$={dist_mod}')
iso_plot(iso1, iso2, axes[1])
axes[0].legend()
cbar = fig.colorbar(ax0, ax=axes[0], label='$p_{ugali}$')
cbar.set_label('$p_{ugali}$', fontsize=16)
axes[1].legend()
cbar = fig.colorbar(ax1, ax=axes[1], label='$p_{ugali}$')
cbar.set_label('$p_{ugali}$', fontsize=16)
#fig.suptitle(f"MER- vs DOLPHOT-selected stars within {l}'x{l}' cutout")
plt.tight_layout(pad=1.12)
fig.savefig('mer_dolphot_VIS_J_v_VIS_H')
plt.close(fig=fig)

#note: for the stellar selection used VIS_TO_H,
    #but for the color-color stellar locus plot used either VIS to J or H depending on axes
#note: used Marigo 2017
#question: ugali stars don't seem to match up 


# NEXT, VIS - J ON X, VIS - H ON Y FOR UGALI STARS
fig, axes = plt.subplots(1,2, figsize=(15,10))
ugali_inbox = square_cut(ugali_data, 'RA', 'DEC', l)

# first, add isochrone in vis-j, vis-h space
Z = 0.0001
age = 12.0
dist_mod = 21.63
iso1 = isochrone.Isochrone(age=age, metallicity=Z, distance_modulus=dist_mod,
                            survey='mixed',
                            band_1= 'VIS',
                            band_2= 'J')
iso2 = isochrone.Isochrone(age=age, metallicity=Z, distance_modulus=dist_mod,
                            survey='mixed',
                            band_1= 'VIS',
                            band_2= 'H')
iso_plot(iso1, iso2, axes[0], label=f'Marigo2017 isochrone\nage={age} Gyr, Z={Z}, $(m-M)_0$={dist_mod}')
iso_plot(iso1, iso2, axes[1])

#axes 1: mer stars
snr = 100
snrmask0 = (mer_stars_J/mer_stars_J_err > snr) & (mer_stars_H/mer_stars_H_err > snr) #(mer_stars_IJ/mer_stars_IJ_err > snr)
ax0 = axes[0].scatter(mer_stars_IJ[snrmask0] - mer_stars_J[snrmask0],
                      mer_stars_IH[snrmask0] - mer_stars_H[snrmask0],
                      c=mer_stars['UGALI_PROB'][snrmask0], cmap=mer_cmap, 
                      s=size, label='MER stars')
axes[0].scatter(ugali_inbox['I0'] - ugali_inbox['J0'],
                ugali_inbox['I0'] - ugali_inbox['H0'],
                c=ugali_data['PROB'], cmap=mer_cmap, 
                s=size, marker='x', label='Ugali objects')
axes[0].set(xlabel='$I_E - J_E$',ylabel='$I_E - H_E$', 
            title=f'MER stars I SNR > {snr} ({len(mer_stars[snrmask0])})')#\ncolor selection, plp>0.05, detection I, J, H, & I SNR > {snr}\n({len(mer_stars_IJ[snrmask0])} stars)')
#axes 2: dolphot stars with a mer match outline
snrmask1a = (dolphot_stars['snr_I'] > snr)
snrmask1b = (dol_matched_mer['snr_I'] > snr)
axes[1].scatter(dol_stars_I[snrmask1a] - dol_stars_J[snrmask1a],
                dol_stars_I[snrmask1a] - dol_stars_H[snrmask1a],
                c=dolphot_stars['UGALI_PROB'][snrmask1a], cmap=dol_cmap,
                s=size, label='All DOLPHOT stars')
ax1 = axes[1].scatter(dolmer_I[snrmask1b] - dolmer_J[snrmask1b],
                      dolmer_I[snrmask1b] - dolmer_H[snrmask1b],
                      c = dol_matched_mer['UGALI_PROB'][snrmask1b], cmap=dol_cmap, edgecolors=mercolor,
                      s=size+10, linewidths=1, label='DOLPHOT stars matched with MER')
axes[1].scatter(ugali_inbox['I0'] - ugali_inbox['J0'],
                ugali_inbox['I0'] - ugali_inbox['H0'],
                c=ugali_inbox['PROB'], cmap=dol_cmap, 
                s=size, marker='x', label='Ugali objects')
axes[1].set(xlabel='$I_E - J_E$',ylabel='$I_E - H_E$',
            title = f'DOLPHOT stars I SNR > {snr} ({len(dolphot_stars[snrmask1a])})') #\nstar_complete, detection I, J, H,  & I SNR > {snr}\n({len(dol_stars_I[snrmask1a])} stars)')

leg = axes[0].legend()
for h, text in zip(leg.legend_handles, leg.get_texts()):
    if text.get_text() == 'Ugali objects':
        h.set_facecolor(mercolor)      # pick any visible color
        #h.set_edgecolor(mercolor)
        h.set_sizes([size])
    if text.get_text() == 'MER stars':
        h.set_facecolor(mercolor)      # pick any visible color
        #h.set_edgecolor(mercolor)
        h.set_sizes([size])
cbar = fig.colorbar(ax0, ax=axes[0], label='$p_{ugali}$')
cbar.set_label('$p_{ugali}$', fontsize=16)

leg = axes[1].legend()
for h, text in zip(leg.legend_handles, leg.get_texts()):
    h.set_facecolor(dolcolor)      # pick any visible color
    #h.set_edgecolor(dolcolor)
    h.set_sizes([size])
cbar = fig.colorbar(ax1, ax=axes[1], label='$p_{ugali}$')
cbar.set_label('$p_{ugali}$', fontsize=16)

#fig.suptitle(f"MER- vs DOLPHOT-selected stars within {l}'x{l}' cutout")
plt.tight_layout(pad=1.12)
fig.savefig('mer_dolphot_highsnr_ugali_VIS_J_v_VIS_H')
plt.close(fig=fig)

#note: for the stellar selection used VIS_TO_H,
    #but for the color-color stellar locus plot used either VIS to J or H depending on axes
#note: used Marigo 2017


# 2D HISTOGRAMS FOR PETER
dol_vis_mag = dol_matched_mer['mag0_I']
dol_j_mag = dol_matched_mer['mag0_J']
dol_h_mag = dol_matched_mer['mag0_H']
dol_v_min_j = dol_vis_mag - dol_j_mag
mer_vis_mag = dol_matched_mer['MAG0_VIS_TO_H']
mer_j_mag = dol_matched_mer['MAG0_J']
mer_h_mag = dol_matched_mer['MAG0_H']
mer_v_min_j = mer_vis_mag - mer_j_mag

b = 20
# CHOICE 1: MESH
fig, axes = plt.subplots(2,2, figsize=(15,10), layout="constrained")
vmax = 3 #int(counts.max())
n = vmax + 1                                   # number of distinct values: 0..vmax
cmap = plt.get_cmap("plasma", n)              # resample cmap into n discrete colors
norm = BoundaryNorm(np.arange(-0.5, vmax + 1.5), cmap.N)
y_list = [mer_vis_mag - dol_vis_mag, mer_j_mag - dol_j_mag, mer_h_mag - dol_h_mag, mer_v_min_j - dol_v_min_j]
y_label_list = ['MER $I_E$ (VIS_TO_H) - DOLPHOT $I_E$', 'MER - DOLPHOT $J_E$', 'MER - DOLPHOT $H_E$', 'MER - DOLPHOT $I_E$ - $J_E$']
for i, ax in enumerate(axes.flatten()):
    counts, xedges, yedges = np.histogram2d(dol_vis_mag, y_list[i], bins=b)
    h = ax.pcolormesh(xedges, yedges, counts.T, cmap=cmap, norm=norm)
    ax.set(xlabel = 'DOLPHOT $I_E$', ylabel=y_label_list[i])

cbar = fig.colorbar(h, ax=axes, ticks=np.arange(0, vmax + 1), label='Stellar Sample Number Counts')
fig.savefig('try_mer_dolphot_stars_photometrymesh')
plt.close(fig=fig)

# CHOICE 2: HISTOGRAM
fig, axes = plt.subplots(2,2, figsize=(15,10), layout="constrained")
h = axes[0,0].hist2d(dol_vis_mag, mer_vis_mag - dol_vis_mag, bins=b)
axes[0,0].set(xlabel = 'DOLPHOT $I_E$', ylabel='MER $I_E$ (VIS_TO_H) - DOLPHOT $I_E$')
axes[0,1].hist2d(dol_vis_mag, mer_j_mag - dol_j_mag, bins=b)
axes[0,1].set(xlabel = 'DOLPHOT $I_E$', ylabel='MER - DOLPHOT $J_E$')
axes[1,0].hist2d(dol_vis_mag, mer_h_mag - dol_h_mag, bins=b)
axes[1,0].set(xlabel = 'DOLPHOT $I_E$', ylabel='MER - DOLPHOT $H_E$')
axes[1,1].hist2d(dol_vis_mag, mer_v_min_j - dol_v_min_j, bins=b)
axes[1,1].set(xlabel = 'DOLPHOT $I_E$', ylabel='MER - DOLPHOT $I_E$ - $J_E$')
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(h[3], ax=axes, shrink=0.8, pad=0.02, label='Stellar Sample Number Counts')
#plt.tight_layout()
#fig.suptitle(f"Photometry of MER- + DOLPHOT-matched stars within {l}'x{l}' cutout")
fig.savefig('mer_dolphot_stars_photometryhist')
plt.close(fig=fig)