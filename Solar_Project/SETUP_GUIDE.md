# Setup and Execution Guide

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Git** (optional, for version control)

## Step 1: Install Dependencies

Open a terminal/command prompt in the `Solar_Project` directory and run:

```bash
pip install -r requirements.txt
```

This will install all required libraries:
- PyTorch (for deep learning)
- Segmentation Models PyTorch (for U-Net)
- Rasterio (for geospatial data)
- GeoPandas (for shapefiles)
- OpenCV (for image processing)
- And other dependencies

## Step 2: Prepare Training Data

### For Building Detection (Required):

1. Download the **Inria Aerial Image Labeling Dataset**:
   - Visit: https://project.inria.fr/aerialimagelabeling/
   - Register and download the training set
   - Extract images to: `data/training_buildings/images/`
   - Extract masks to: `data/training_buildings/masks/`
   - **Note**: You only need 10-20 images for a student project, not the full dataset

### For Roof Classification (If No DSM Available):

1. Manually crop roof images from your city map:
   - Create 50+ images of **flat roofs** → save to `data/training_roofs/flat/`
   - Create 50+ images of **gable/slanted roofs** → save to `data/training_roofs/gable/`
   - Images can be in JPG or PNG format
   - Recommended size: 224x224 pixels or larger

## Step 3: Prepare Project Area Data

Place your city data in `data/project_area/`:

1. **Satellite Image** (`my_city.tif`):
   - High-resolution RGB GeoTIFF of your study area
   - Can be exported from SASPlanet, Google Earth Engine, or similar tools

2. **DSM (Optional)** (`my_city_dsm.tif`):
   - Digital Surface Model (height data)
   - Download from JAXA ALOS World 3D or OpenTopography
   - If unavailable, the system will use AI-based classification

3. **Night Light Data** (`viirs_nightlight.tif`):
   - VIIRS VNP46A1 from NASA Earthdata
   - Used for urban/rural viability check

## Step 4: Execute the Pipeline

### 4.1 Preprocess Training Data

```bash
cd scripts
python 0_slice_images.py
```

This will slice large images into 512x512 patches and save them to `data/training_sliced/`.

### 4.2 Check Viability (Optional)

```bash
python 1_check_viability.py
```

This checks if your area is urban enough for solar analysis.

### 4.3 Train Building Detector

```bash
python 2_train_detector.py
```

**Time**: ~1-2 hours on CPU, ~10 minutes on GPU

This trains the U-Net model to detect buildings. The model will be saved to `models/unet_detector.pth`.

### 4.4 Train Roof Classifier (Only if No DSM)

```bash
python 3_train_classifier.py
```

**Time**: ~10 minutes

This trains the ResNet model to classify roof types. Only needed if you don't have DSM data.

### 4.5 Run Main Pipeline

```bash
python 4_main_pipeline.py
```

**Time**: Depends on image size (typically 10-30 minutes)

This is the master script that:
1. Detects all buildings in your city map
2. Classifies roof types (using DSM or AI)
3. Calculates solar potential, savings, and CO2 offset
4. Generates the final shapefile

## Step 5: Visualize Results

1. Open `output/final_solar_map.shp` in **QGIS** (free GIS software)
2. Right-click the layer → Properties → Symbology
3. Choose **Graduated** style
4. Select column: `annual_energy_kwh` or `annual_savings_inr`
5. Choose a color ramp (e.g., Red to Blue)
6. Click OK to see your solar potential map!

## Troubleshooting

### Issue: "FileNotFoundError" when running scripts
- **Solution**: Make sure you're running scripts from the `scripts/` directory
- All paths in scripts are relative to the `scripts/` folder

### Issue: "No module named 'patchify'"
- **Solution**: Run `pip install patchify`

### Issue: Training takes too long
- **Solution**: Reduce the number of training images or use GPU acceleration

### Issue: "CUDA out of memory"
- **Solution**: Reduce batch size in training scripts (change `batch_size=8` to `batch_size=4`)

### Issue: Model not found when running pipeline
- **Solution**: Make sure you've completed Step 4.3 (train detector) before running the pipeline

## Expected Output

The final shapefile (`output/final_solar_map.shp`) contains:
- **geometry**: Building footprints (polygons)
- **total_area**: Total roof area in m²
- **roof_type**: "Flat" or "Gable"
- **usable_area**: Usable area for solar panels in m²
- **annual_energy_kwh**: Annual energy generation potential
- **annual_savings_inr**: Annual cost savings in Indian Rupees
- **co2_saved_kg**: Annual CO2 offset in kilograms

## Next Steps

- Validate results by manually checking 10-20 buildings in Google Earth
- Create visualizations and charts for your presentation
- Calculate total city potential (sum of all buildings)
- Compare different areas of the city
