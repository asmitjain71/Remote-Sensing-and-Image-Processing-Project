# scripts/1_check_viability.py
import rasterio
import numpy as np

VIIRS_PATH = '../data/project_area/viirs_nightlight.tif'
THRESHOLD = 5.0 # nW/cm2/sr

def check_viability():
    try:
        with rasterio.open(VIIRS_PATH) as src:
            data = src.read(1)
            # Filter negatives (sensor noise)
            data = np.maximum(data, 0)
            avg_radiance = np.mean(data)
            
            print(f"📊 Average Night Light Radiance: {avg_radiance:.2f}")
            
            if avg_radiance > THRESHOLD:
                print("✅ Area is URBAN. Proceeding with High-Res Analysis.")
                return True
            else:
                print("❌ Area is RURAL. Solar ROI too low. Aborting.")
                return False
    except FileNotFoundError:
        print("⚠️ No VIIRS file found. Skipping check (Assumed Urban).")
        return True

if __name__ == "__main__":
    check_viability()
