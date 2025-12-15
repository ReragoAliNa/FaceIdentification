
import io
import os
import urllib.request
import zipfile

def download_and_extract():
    url = "https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"
    print(f"Downloading from {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
            print(f"Downloaded {len(data)} bytes.")
            
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                # The zip usually contains 'orl_faces' directory or just s1..s40
                # Let's inspect names
                names = z.namelist()
                print(f"Zip contains {len(names)} files.")
                
                # Check if it has a root folder or not
                # Standard att_faces.zip has a readme and s1..s40 at root
                # We want to put them in 'orl_faces'
                
                extract_path = "orl_faces"
                if not os.path.exists(extract_path):
                    os.makedirs(extract_path)
                
                print(f"Extracting to {extract_path}...")
                z.extractall(extract_path)
                print("Done.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_and_extract()
