import json
from landsatxplore.api import API
import pandas as pd
import argparse
from datetime import datetime
import os

def get_state_bbox():
    """Return dictionary of state bounding boxes"""
    return {
        'AZ': (-114.818269, 31.332177, -109.045223, 37.004260),  # Arizona
        'TX': (-106.645646, 25.837377, -93.508292, 36.500704),   # Texas
        'FL': (-87.634938, 24.523096, -80.031362, 31.000888),    # Florida
        'GA': (-85.605165, 30.357851, -80.839729, 35.000659),    # Georgia
        'IL': (-91.513079, 36.970298, -87.494756, 42.508481),    # Illinois
        'MT': (-116.050003, 44.358221, -104.039138, 49.001146),  # Montana
        'NM': (-109.050173, 31.332177, -103.001964, 37.000232),  # New Mexico
        'WA': (-124.763068, 45.543541, -116.915989, 49.002494),  # Washington
        'CO': (-109.060253, 36.992426, -102.041524, 41.003444),  # Colorado
        'UT': (-114.052962, 36.997968, -109.041058, 42.001567)   # Utah
    }


def validate_year(year):
    """Validate if the year is within reasonable range"""
    current_year = datetime.now().year
    if not (2013 <= year <= current_year):  # Landsat 8 launched in 2013
        raise ValueError(f"Year must be between 2013 and {current_year}")
    return year

def validate_state(state, state_dict):
    """Validate if the state code exists in our dictionary"""
    if state not in state_dict:
        valid_states = ', '.join(sorted(state_dict.keys()))
        raise ValueError(f"Invalid state code. Valid states are: {valid_states}")
    return state

def search_landsat_data(year, state, username='nibir088', password='bakul1147130BUET'):
    """
    Search Landsat data for given year and state
    
    Parameters:
    year (int): Year to search for
    state (str): Two-letter state code
    username (str): Landsat API username
    password (str): Landsat API password
    """
    try:
        # Get state bounding boxes
        state_dict = get_state_bbox()
        
        # Validate inputs
        year = validate_year(year)
        state = validate_state(state.upper(), state_dict)
        
        # Initialize API
        print(f"Connecting to Landsat API...")
        api = API(username, password)
        
        # Search for scenes
        print(f"Searching for Landsat scenes in {state} for year {year}...")
        scenes = api.search(
            dataset='landsat_ot_c2_l2',
            bbox=state_dict[state],
            start_date=f'{year}-03-01',
            end_date=f'{year}-09-31',
            max_cloud_cover=5,
            max_results=10000
        )
        
        # Convert to DataFrame and save
        df = pd.DataFrame(scenes)
        os.makedirs('available_files/', exist_ok=True)
        output_file = f'available_files/landsat_8_Y_{year}_S_{state}_cloud_5.csv'
        df.to_csv(output_file)
        print(f"Found {len(scenes)} scenes")
        print(f"Results saved to: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
        
    finally:
        try:
            api.logout()
            print("Logged out from Landsat API")
        except:
            pass

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Search Landsat data for a specific year and state')
    parser.add_argument('-y', '--year', type=int, required=True, help='Year to search (2013 or later)')
    parser.add_argument('-s', '--state', type=str, required=True, help='Two-letter state code')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run search
    search_landsat_data(args.year, args.state)