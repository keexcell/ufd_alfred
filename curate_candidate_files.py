import os
import shutil
import sys
import re

# run with python2 curate_candidate_files.py job_id
#n_jobs = int(sys.argv[1])
try:
    job_id = sys.argv[1]
except:
    print('forgot job id')
    raise(ValueError)
    exit()
dir_list = os.listdir('outputs')
empty_pix = []
for file in dir_list:
    if (str(job_id) in file and 'tar.gz' in str(file)):
        try:
            os.system(f'tar -xzf outputs/{file}') #unpacks the tarred file will open in home/output
            if not os.path.exists(f'outputs/{job_id}_candidates'):
                os.mkdir(f'outputs/{job_id}_candidates')
            if not os.path.exists(f'outputs/{job_id}_plots'):
                os.mkdir(f'outputs/{job_id}_plots')
            output_file_str = ','.join(os.listdir('output'))
            if 'EMPTY' in output_file_str:
                #EMPTY{SearchRegion.nside}_{SearchRegion.pixel}_{Stars.survey}
                empty_start = output_file_str.find('EMPTY')
                first_underscore = output_file_str.find('_',empty_start)
                second_underscore = output_file_str.find('_',first_underscore+1)
                pix = output_file_str[first_underscore:second_underscore]
                empty_pix.append(pix)
                continue
            os.system(f'mv output/simplecandidates_* outputs/{job_id}_candidates') #move the csvs to the candidates folder
            try:
                os.system(f'mv output/plots/* outputs/{job_id}_plots') # move the plots to the plots folder
            except:
                os.system(f'mv output/plots/*/* outputs/{job_id}_plots') # move the subfolders to the plots folder
            os.system(f'mv outputs/{file} /staging/k/kexcell/tar_files/{file}') # shove the tar files into staging
        except:
            pass

print(f'Empty pixels: {empty_pix}')

dir_path = f'outputs/{job_id}_candidates'
dir_list = os.listdir(dir_path)
dir_list.sort()
outfile_name = f'{job_id}_all_candidates.csv'
# read all the candidates into one giant csv
with open(outfile_name, 'w') as outfile:
    for i, filename in enumerate(dir_list):
        inpath = os.path.join(dir_path,filename)
        with open(inpath, 'r') as infile:
            if i!=0:
                infile.readline()
            shutil.copyfileobj(infile, outfile)

os.system('rm -r output')
