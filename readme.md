# IrrMap

## Overview
This is official git repository for our work IrrMap, published in **KDD 2025**. The **IrrMap** is designed to process satellite imagery data from LandSat and Sentinel missions for developing and training machine learning models for irrigation classification. This repository includes tools for data preprocessing, feature extraction, model training, and evaluation. Paper link: (https://arxiv.org/abs/2505.08273)

## Data Repository

Labeled data has been collected from different sources as follows:
- Arizona: https://catalog.data.gov/dataset/verified-irrigated-agricultural-lands-for-the-united-states-200217
- Utah: https: //dwre-utahdnr.opendata.arcgis.com/pages/wrlu-data 
- Washington: https://agr.wa.gov/departments/land-and-water/natural-resources/agricultural-land-use
- Colorado: https://dwr.colorado.gov/services/data-information/gis

We processed this data and created patches following **IrrMap** paper (given in the repository). Processed data can be downloaded from [huggingface](https://huggingface.co/Nibir/IrrMap/tree/main). Data Structure of the given repository:

```
IrrMap/
│── LandSat_Label.zip     # dataset contains labels for LandSat patches of shape 224x224 
│── Sentinel_Label.zip    # dataset contains labels for Sentinel patches of shape 224x224 
│── shape-file.zip        # Raw shape-file collected from different sources 
│── LandSat/              # LandSat imagery patches divided into two zip files. ``readme.md'' contains meta data information.
|── |── LandSat_part1.zip
|── |── LandSat_part2.zip
|── |── readme.md
│── Sentinel/             # Sentinel imagery patches for each states. ``readme.md'' contains meta data information.
|── |── AZ
|── |── CO
|── |── UT
|── |── WA
|── |── readme.md      
```


## Repository Structure
```
KDD-2025-Data-Track/
│── LandSat_Processing/    # Scripts for downloading and processing LandSat imagery
│── Sentinel_Processing/   # Scripts for retrieving and processing Sentinel imagery
│── Model_Training/        # Machine learning model training scripts and configurations
│── train_test_split/      # Scripts for splitting data into training and test sets
│── Crop-Groups.csv        # Crop classification mapping file
│── requirements.txt       # List of dependencies
|── Manual_verification.zip # Patches that has been manually verified (0 indicates noisy and 1 indicates clean patch). 
│── README.md              # Documentation
```

## Data Processing Workflow
### LandSat Data Processing
- **Data Acquisition**: Downloading LandSat data.
- **Preprocessing**:
  - Resampling and reprojection
  - Cloud masking
  - Quality checking through QA_Band
- **Feature Extraction**:
  - Vegetation indices (NDVI, EVI)
  - Spectral band combinations (stacking)

### Sentinel Data Processing
- **Data Retrieval**: Fetching Sentinel-1 (SAR) and Sentinel-2 (multispectral) imagery.
- **Preprocessing**:
  - Geometric and radiometric corrections
  - Normalization and resampling
  - Cloud removal
- **Feature Engineering**:
  - Polarimetric decomposition (SAR)
  - Spectral indices calculation
  - Temporal feature stacking
 
### Data Labeling
- Preprocessing: Reprojection (EPSG:4326)
- Label Merging: Shapely Library for overlay
### Train Test Split
- There are three zip file inside train_test_split.
  - LandSat.zip contains train test split for LandSat patches for each states.
  - Sentinel.zip contains train test split for Sentinel patches for each states.
  - Subset_Training_Set.zip contains a small training dataset of 234861 patches for Sentinel and 26316 patches for LandSat.

## Model Training Pipeline
1. **Data Preparation**:
   - Load and merge LandSat and Sentinel data
   - Normalize and scale features
2. **Model Selection**:
   - Deep Learning: `Resnet34`
3. **Training Process**:
   - Hyperparameter tuning
   - Loss function optimization
4. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-Score, IoU


## Crop Classification Mapping
The `Crop-Groups.csv` file maps crop types into broader classifications:
- Specific crops (e.g., Wheat, Maize, Rice)
- Crop groups (e.g., Cereal Grains, Legumes, Fruits)

## Installation and Setup
### Clone the Repository
```bash
git clone https://github.com/Nibir088/KDD-2025-Data-Track.git
cd KDD-2025-Data-Track
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Process Data
```bash
cd LandSat_Processing
./sample_landsat_processing_pipeline.sh
cd Sentinel_Processing
./sample_sentinel_processing_pipeline.sh
```
or 
```bash
cd LandSat_Processing
python find_available_data_landsat.py -s AZ -y 2016
python download_available_data_landsat.py -s AZ -y 2016 -u "username" -p "password"
python extract_files.py -s AZ
python create_patches_landsat.py -s AZ
python create_label_landsat.py --state AZ --year 2016
cd Sentinel_Processing
python find_available_data_sentinel.py -s AZ -y 2016 --start_date 2016-03-01 --end_date 2016-09-30 -o sentinel_AZ_2016.csv
python download_available_data_sentinel.py -s AZ -y 2016 -u "username" -p "password"
python extract_files.py -s AZ -y 2016
python create_qa_band.py -p "Sentinel-UnZip/AZ"
python create_patches_sentinel.py -s AZ -i "Sentinel-Unzip"
python create_label_sentinel.py --state AZ --year 2016
```
### Train the Model
Unzip train_test_split/LandSat.zip and train_test_split/Sentinel.zip and run

```bash
cd Model_Training
python train.py --data_path ../train_test_split/LandSat/IrrMap_combined.yaml \
                --source landsat \
                --input_types "image,ndvi" \
                --epochs 5 \
                --batch_size 32 \
                --num_classes 4 \
                --lr 0.0005 \
                --devices -1 \
                --strategy ddp
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Make changes and commit with clear messages.
4. Submit a pull request.

## License

## Contact & Support
For issues or inquiries, open an issue on [GitHub Issues](https://github.com/Nibir088/KDD-2025-Data-Track/issues).

