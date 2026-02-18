# 🌞 Urban Solar Energy Mapping Project

Complete implementation of Theme 4: Urban Solar Energy Mapping with AI-powered building detection and solar potential analysis.

## Project Structure

```
/Solar_Project
    /scripts
        0_slice_images.py          # Prepares data
        1_check_viability.py       # (Student 3) Night Light Filter
        2_train_detector.py        # (Student 1) Building Detection AI (U-Net)
        3_train_classifier.py      # (Student 2) Roof Type AI (ResNet) - *For No DSM*
        4_main_pipeline.py         # (Student 4/5) The Master Script (Detection + Analysis)
    /data
        /training_buildings        # INRIA/SpaceNet Dataset (For Script 2)
            /images
            /masks
        /training_roofs            # (For Script 3 - If No DSM)
            /flat                  # Put 50 crop images of flat roofs here
            /gable                 # Put 50 crop images of slanted roofs here
        /project_area
            my_city.tif            # Your high-res map
            my_city_dsm.tif        # (Optional) Height map
            viirs_nightlight.tif   # Night light data
    /models
        unet_detector.pth
        resnet_classifier.pth
    /output
        final_solar_map.shp
```

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

- Download Inria dataset images → `data/training_buildings/images/` and masks → `data/training_buildings/masks/`
- Download your city map → `data/project_area/my_city.tif`
- **(If No DSM)**: Manually create folders `data/training_roofs/flat` and `/gable` and fill them with 50 small images each.

### 2. Run Preprocessing

```bash
cd scripts
python 0_slice_images.py
```

### 3. Train Models

```bash
python 2_train_detector.py  # Wait ~1 hour
```

**(If No DSM)**:
```bash
python 3_train_classifier.py  # Wait ~10 mins
```

### 4. Execute Pipeline

```bash
python 4_main_pipeline.py
```

### 5. Visualize

Open `output/final_solar_map.shp` in QGIS.
- Change color style to categorize by `roof_type` or graduate by `annual_savings_inr`.

## Script Descriptions

- **0_slice_images.py**: Slices large satellite images into 512x512 patches for training
- **1_check_viability.py**: Checks night light data to confirm urban area viability
- **2_train_detector.py**: Trains U-Net model for building detection
- **3_train_classifier.py**: Trains ResNet model for roof type classification (fallback when DSM unavailable)
- **4_main_pipeline.py**: Master pipeline that detects buildings, classifies roofs, and calculates solar potential

## Output

The final output (`output/final_solar_map.shp`) contains:
- Building footprints (geometry)
- Roof type (Flat/Gable)
- Total area (m²)
- Usable area (m²)
- Annual energy potential (kWh)
- Annual savings (INR)
- CO2 offset (kg)

## Notes

- The project supports both DSM-based (geometric) and AI-based (visual) roof classification
- All paths in scripts are relative to the `scripts/` directory
- Make sure to have sufficient training data before running the training scripts
