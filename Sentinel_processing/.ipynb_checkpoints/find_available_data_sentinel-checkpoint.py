import requests
import pandas as pd
import argparse

def fetch_sentinel_data(state, start_date, end_date, output_file):
    data_collection = "SENTINEL-2"

    aoi = {
        'UT': "POLYGON((-114.039648 36.993076,-114.039648 41.003444,-109.045223 41.003444,-109.045223 36.993076,-114.039648 36.993076))",
        'AZ': "POLYGON((-114.818269 31.332177,-114.818269 37.00426,-109.045223 37.00426,-109.045223 31.332177,-114.818269 31.332177))",
        'WA': "POLYGON((-124.848974 45.543541,-124.848974 49.002494,-116.916556 49.002494,-116.916556 45.543541,-124.848974 45.543541))",
        'CO': "POLYGON((-109.060257 36.993076,-109.060257 41.003444,-102.041524 41.003444,-102.041524 36.993076,-109.060257 36.993076))",
        'FL': "POLYGON((-87.634938 24.396308,-87.634938 31.000888,-80.031362 31.000888,-80.031362 24.396308,-87.634938 24.396308))"
    }

    if state not in aoi:
        raise ValueError(f"Invalid state code: {state}. Choose from {list(aoi.keys())}")

    query_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{aoi[state]}') and ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T00:00:00.000Z and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le 5.00)&$orderby=ContentDate/Start desc&$top=1000"
    print(query_url)
    response = requests.get(query_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch Sentinel-2 data. Status code: {response.status_code}")

    json_data = response.json()
    if 'value' not in json_data:
        raise Exception("Invalid response format from Sentinel-2 API")

    data_df = pd.DataFrame(json_data['value'])

    if data_df.empty:
        print("No data found for the given parameters.")
    else:
        data_df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Sentinel-2 metadata and save as CSV")
    parser.add_argument('-s', '--state', type=str, required=True, help="Two-letter state code (e.g., AZ, TX, FL)")
    parser.add_argument('--start_date', type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end_date', type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument('-o', '--output_file', type=str, default="sentinel_data.csv", help="Output CSV file")

    args = parser.parse_args()

    fetch_sentinel_data(args.state, args.start_date, args.end_date, args.output_file)
