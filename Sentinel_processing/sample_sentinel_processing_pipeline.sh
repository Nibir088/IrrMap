#!/bin/bash

# Activate virtual environment (if applicable)
# source /path/to/venv/bin/activate  

# Run the Processing script
python find_available_data_sentinel.py -s AZ -y 2016 --start_date 2016-03-01 --end_date 2016-09-30 -o sentinel_AZ_2016.csv

python download_available_data_sentinel.py -s AZ -y 2016 -u "username" -p "password"

python extract_files.py -s AZ -y 2016

python extract_files.py -s AZ -y 2016

python create_qa_band.py -p "Sentinel-UnZip/AZ"

python create_patches_sentinel.py -s AZ -i "Sentinel-Unzip"

python create_label_sentinel.py --state AZ --year 2016