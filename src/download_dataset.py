import pandas as pd
import os
import urllib.request

def download_dataset():
    url = "https://raw.githubusercontent.com/dcpedrelli/hotel-bookings/master/hotel_bookings.csv"
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, "hotel_bookings.csv")
    
    if os.path.exists(filepath):
        print(f"Dataset already exists at {filepath}")
        return
        
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"Dataset successfully downloaded and saved to {filepath}")
        
        # Verify it loads
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Failed to download dataset: {e}")

if __name__ == "__main__":
    download_dataset()
