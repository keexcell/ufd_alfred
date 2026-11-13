#$1 needs to be job id (cluster)
python3 curate_candidate_files.py $1

#run curate data files for next euclid_dr1.tar.gz job run
cd /staging/k/kexcell/euclid_dr1 && for f in *.tar.gz; do tar -xzf "$f"; done
mv /staging/k/kexcell/euclid_dr1/*.tar.gz /staging/k/kexcell/tar_files
cd /staging/k/kexcell/ && tar -czf euclid_dr1_new.tar.gz euclid_dr1

#apptainer shell -e post_processing.sif
#python3 drop_duplicate_candidates.py $1 $2
