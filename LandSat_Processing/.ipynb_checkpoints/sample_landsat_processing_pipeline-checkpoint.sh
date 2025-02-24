#!/bin/bash

# Activate virtual environment (if applicable)
# source /path/to/venv/bin/activate  

# Run the Processing script
python find_available_data_landsat.py -s AZ -y 2016

python download_available_data_landsat.py -s AZ -y 2016 -u "username" -p "password"

python extract_files.py -s AZ

python create_patches_landsat.py -s AZ

python create_label_landsat.py --state AZ --year 2016