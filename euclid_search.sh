#!/bin/bash
#
#test_main.sh
# My CHTC job
#
#untar the ufd_alfred zip
tar -xzf ufd_alfred.tar.gz
#untar ugali
#tar -xzf ugali.tar.gz
#print pixel numbers
cd ufd_alfred
python3 euclid_main.py $1 $2
#tar -czf data_output.tar.gz *.parquet
#tar results folder
cd ..
tar -czf output.tar.gz output
#mv ufd_alfred/data_output.tar.gz data_output.tar.gz