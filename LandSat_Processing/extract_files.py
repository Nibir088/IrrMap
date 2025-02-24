import tarfile
import shutil
from tqdm import tqdm
import glob
import os
import logging
import argparse
from datetime import datetime
import sys

def setup_logging(state):
    """Setup logging with timestamp in filename"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'{log_dir}/extraction_failures_{state}_{timestamp}.log'
    
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return log_file

def get_valid_states():
    """Return list of valid state codes"""
    return ['AZ', 'TX', 'FL', 'GA', 'IL', 'MT', 'NM', 'WA', 'CO','UT']

def validate_state(state):
    """Validate state input"""
    valid_states = get_valid_states()
    if state.upper() not in valid_states:
        raise ValueError(f"Invalid state code. Valid states are: {', '.join(valid_states)}")
    return state.upper()

def extract_landsat_files(state, base_directory='LandSat-8'):
    """
    Extract Landsat tar files for a specific state
    
    Parameters:
    state (str): Two-letter state code
    base_directory (str): Base directory for temporary extraction
    """
    try:
        # Validate state
        state = validate_state(state)
        
        # Setup logging
        log_file = setup_logging(state)
        print(f"Logging errors to: {log_file}")
        
        # Create base directory
        # os.makedirs(base_directory, exist_ok=True)
        
        # Find all tar files for the state
        tar_pattern = f'Downloaded-Tar-File/Landsat-8_{state}_file/*/*'
        tar_files = glob.glob(tar_pattern)
        
        if not tar_files:
            print(f"No tar files found for state {state} using pattern: {tar_pattern}")
            return
        
        print(f"Found {len(tar_files)} tar files to extract for {state}")
        
        # Counter for statistics
        stats = {
            'success': 0,
            'skipped': 0,
            'failed': 0
        }
        
        # Process each tar file with progress bar
        for tar_file_path in tqdm(tar_files, desc=f"Extracting files for {state}"):
            try:
                # Parse file information
                fname = os.path.splitext(os.path.basename(tar_file_path))[0]
                year = os.path.basename(os.path.dirname(tar_file_path))
                extract_to_dir = f'Extracted-File/{state}/{year}/{fname}'
                
                # Skip if already extracted
                if os.path.exists(extract_to_dir):
                    stats['skipped'] += 1
                    continue
                
                # Create extraction directory
                os.makedirs(os.path.dirname(extract_to_dir), exist_ok=True)
                
                # Extract tar file
                with tarfile.open(tar_file_path, 'r') as tar:
                    tar.extractall(path=extract_to_dir)
                
                stats['success'] += 1
                
            except Exception as e:
                stats['failed'] += 1
                # Log the error
                error_msg = f"{state}: Error extracting {tar_file_path}: {str(e)}"
                logging.error(error_msg)
                print(f"\nError: {error_msg}")
                
                # Clean up failed extraction
                if os.path.exists(extract_to_dir):
                    shutil.rmtree(extract_to_dir)
                
                # Remove corrupted tar file
                if os.path.isfile(tar_file_path):
                    os.remove(tar_file_path)
                    print(f"Removed corrupted tar file: {tar_file_path}")
        
        # Print summary
        print("\nExtraction Summary:")
        print(f"Successfully extracted: {stats['success']} files")
        print(f"Skipped (already extracted): {stats['skipped']} files")
        print(f"Failed: {stats['failed']} files")
        print(f"Details of failed extractions can be found in: {log_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"General error: {str(e)}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract Landsat tar files for a specific state')
    parser.add_argument('-s', '--state', type=str, required=True,
                       help='Two-letter state code (e.g., AZ, TX, FL)')
    # parser.add_argument('-d', '--directory', type=str, default='LandSat-8',
    #                    help='Base directory for extraction (default: LandSat-8)')
    
    args = parser.parse_args()
    
    # Run extraction
    extract_landsat_files(args.state)#, args.directory
