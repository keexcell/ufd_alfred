from ugali.utils import projector
import pandas as pd
import numpy as np
import sys

n_jobs = int(sys.argv[1])
job_id = int(sys.argv[2])
all_simple_candidates = pd.read_csv(f'{job_id}_all_candidates.csv')
known_objects = pd.read_csv('all_mw_candidates.csv')

all_simple_candidates.sort_values('sig', ascending=False) # so now the first one is the most significant
all_simple_candidates['duplicate_id'] = np.arange(0,len(all_simple_candidates),1)
print(all_simple_candidates['duplicate_id'])
#overlaps = []
for i in range(len(all_simple_candidates)):
    for j in range(i+1, len(all_simple_candidates)):
        angsep = projector.angsep(all_simple_candidates.at[i,'ra_deg'], all_simple_candidates.at[i,'dec_deg'],
                                  all_simple_candidates.at[j,'ra_deg'], all_simple_candidates.at[j,'dec_deg'],)
        if (angsep<=all_simple_candidates.at[i,'r_deg']) | (angsep<=all_simple_candidates.at[j,'r_deg']):
            # this is considered an overlap
            # check if they have the same significance
            # if they do then keep both distance moduli in the first one
            # then change their duplicate index column to the same number
            all_simple_candidates.at[j, 'duplicate_id'] = all_simple_candidates.at[i, 'duplicate_id']
            #overlaps.append((all_simple_candidates.iloc[i],all_simple_candidates.iloc[j]))
all_simple_candidates.drop_duplicates(subset='duplicate_id')

#for pair in overlaps:
#    more_sig = pair[0] #because in the pair, i always comes first. i < j and this list is sorted by significance
#    less_sig = pair[1]
#    if more_sig['sig']==less_sig['sig']: #I don't know what to do if they have the same significance yet, this should save both
#        more_sig.samesig_duplicates.append({'distance' : less_sig['distance_kpc'], 'r' : less_sig['r_deg'],
#                                            'ra' : less_sig['ra_deg'],  'dec' : less_sig['dec_deg']
#                                            })
#    try:
#        del Peaks[Peaks.index(less_sig)] #this (hopefully) works because they're all different objects 
#                                #therefore none are exactly alike and they won't be deleted 2x
#    #    print('successfully deleted above')
#    except:
#        continue
# as far as I can tell the above code works for duplicates (commented out code is for debugging)
