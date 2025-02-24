import os
import requests
import pandas as pd
from tqdm import tqdm
import argparse

def get_keycloak(username: str, password: str) -> tuple:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token", data=data)
    r.raise_for_status()
    return r.json()["access_token"], r.json()["refresh_token"]

def refresh_token(r_token: str) -> tuple:
    data = {
        "client_id": "cdse-public",
        "refresh_token": r_token,
        "grant_type": "refresh_token",
    }
    r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token", data=data)
    r.raise_for_status()
    return r.json()["access_token"], r.json()["refresh_token"]

def download_sentinel_data(state, year, username, password, output_dir):
    # Read CSV file
    data_df = pd.read_csv(f'available-file/sentinel_{state}_{year}.csv')
    if 'Id' not in data_df.columns:
        raise ValueError("CSV file does not contain an 'Id' column.")
    
    output_dir = output_dir+f'/{state}/{year}'
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Authenticate user
    a_token, r_token = get_keycloak(username, password)
    headers = {"Authorization": f"Bearer {a_token}"}

    # Create session
    session = requests.Session()
    session.headers.update(headers)

    for i, data_id in enumerate(data_df['Id']):
        file_path = os.path.join(output_dir, f"{data_id}.zip")
        if os.path.isfile(file_path):
            print(f"File {file_path} already exists, skipping...")
            continue

        url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({data_id})/$value"
        response = session.get(url, stream=True)

        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(file_path, 'wb') as file, tqdm(
                desc=f"Downloading {data_id}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                ascii=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    pbar.update(len(chunk))

        elif response.status_code == 401:
            if 'Expired signature' in response.text:
                a_token, r_token = refresh_token(r_token)
                session.headers.update({"Authorization": f"Bearer {a_token}"})
                print(f'Expired token. Refreshing and retrying for {data_id}...')
            else:
                print('Authorization problem. Could not download file.')

        else:
            print(f"Failed to download {data_id}. Status code: {response.status_code}")
            print(response.text)
            break

        print(f"Downloaded {i+1}/{len(data_df)}: {data_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Sentinel-2 data from a CSV file")
    parser.add_argument('-s', '--state', type=str, required=True, help="state-name")
    parser.add_argument('-y', '--year', type=str, required=True, help="year")
    parser.add_argument('-u', '--username', type=str, required=True, help="Copernicus Dataspace username")
    parser.add_argument('-p', '--password', type=str, required=True, help="Copernicus Dataspace password")
    parser.add_argument('-o', '--output_dir', type=str, default="Sentinel-Zip", help="Directory to save downloaded files")

    args = parser.parse_args()

    download_sentinel_data(args.state, args.year, args.username, args.password, args.output_dir)
