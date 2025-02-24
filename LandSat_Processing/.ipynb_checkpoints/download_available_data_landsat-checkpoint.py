import json
from landsatxplore.earthexplorer import EarthExplorer
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from datetime import datetime
import argparse

def get_valid_states():
    return ['AZ', 'TX', 'FL', 'GA', 'IL', 'MT', 'NM', 'WA', 'CO','UT']

def validate_inputs(state, year):
    valid_states = get_valid_states()
    if state not in valid_states:
        raise ValueError(f"Invalid state code. Valid states are: {', '.join(valid_states)}")
    
    current_year = datetime.now().year
    if not (2013 <= year <= current_year):
        raise ValueError(f"Year must be between 2013 and {current_year}")
    
    return True

def save_failed_downloads(failed_files, state, year):
    """Save failed downloads to a text file"""
    error_dir = 'error_logs'
    os.makedirs(error_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    error_file = f'{error_dir}/failed_downloads_{state}_{year}_{timestamp}.txt'
    
    with open(error_file, 'w') as f:
        f.write(','.join(failed_files))
    
    return error_file

def setup_logging(state, year):
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'{log_dir}/download_failures_{state}_{year}_{timestamp}.log'
    
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return log_file

def download_file(args):
    display_id, index, total, state, year, ee = args
    output_dir = f'Downloaded-Tar-File/Landsat-8_{state}_file/{year}'
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, f'{display_id}.tar')
    status_msg = f'[{index}/{total}] {display_id}'
    
    if os.path.isfile(file_path):
        return f'{status_msg} - Already exists', None
    
    try:
        ee.download(display_id, output_dir=output_dir)
        return f'{status_msg} - Download completed', None
    except Exception as e:
        error_msg = f'{state}: Error downloading {display_id}: {str(e)}'
        logging.error(error_msg)
        return f'{status_msg} - {error_msg}', display_id

def read_filenames_from_text(text_file):
    """Read display IDs from the given text file"""
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Text file not found: {text_file}")
    
    with open(text_file, 'r') as f:
        content = f.read().strip()
    return content.split(',')

def download_landsat_data(state, year, username='ncm088', password='', text_file=None):
    try:
        validate_inputs(state.upper(), year)
        state = state.upper()
        
        print("Initializing EarthExplorer...")
        ee = EarthExplorer(username, password)
        print("EarthExplorer initialized successfully")
        
        log_file = setup_logging(state, year)
        print(f"Logging errors to: {log_file}")
        
        if text_file:
            print(f"Reading display IDs from: {text_file}")
            display_ids = read_filenames_from_text(text_file)
            df = pd.DataFrame(display_ids, columns=['display_id'])
        else:
            csv_path = f'available_files/landsat_8_Y_{year}_S_{state}_cloud_5.csv'
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            df = pd.read_csv(csv_path, index_col=0)
            df = df.sample(frac=1).reset_index(drop=True)

        total_files = len(df)
        print(f"Found {total_files} files to download for {state} {year}")
        
        download_args = [
            (row['display_id'], idx + 1, total_files, state, year, ee) 
            for idx, (_, row) in enumerate(df.iterrows())
        ]
        
        start_time = time.time()
        completed = 0
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(download_file, args) for args in download_args]
            
            for future in as_completed(futures):
                completed += 1
                result, failed_file = future.result()
                
                if failed_file:
                    failed_files.append(failed_file)
                
                elapsed_time = time.time() - start_time
                avg_time_per_file = elapsed_time / completed
                remaining_files = total_files - completed
                estimated_time_remaining = avg_time_per_file * remaining_files
                
                print(f"{result} | Progress: {completed}/{total_files} "
                      f"({(completed/total_files*100):.1f}%) | "
                      f"Est. remaining time: {estimated_time_remaining/60:.1f} minutes")
        
        if failed_files:
            error_file = save_failed_downloads(failed_files, state, year)
            print(f"\nFailed downloads ({len(failed_files)} files) saved to: {error_file}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"General error: {str(e)}")
    
    finally:
        try:
            ee.logout()
            print("Successfully logged out from EarthExplorer")
        except:
            print("Error logging out from EarthExplorer")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Landsat data for a specific state and year')
    parser.add_argument('-s', '--state', type=str, required=True, 
                       help='Two-letter state code (e.g., AZ, TX, FL)')
    parser.add_argument('-y', '--year', type=int, required=True,
                       help='Year to download (2013 or later)')
    parser.add_argument('-u', '--username', type=str, default='',
                       help='EarthExplorer username')
    parser.add_argument('-p', '--password', type=str, default='',
                       help='EarthExplorer password')
    parser.add_argument('-t', '--text_file', type=str, default=None,
                       help='Optional: Path to a text file with filenames (comma-separated)')
    
    args = parser.parse_args()
    
    download_landsat_data(args.state, args.year, args.username, args.password, args.text_file)
