# YOLO for Building Detection & Footprint Extraction (Remote Sensing)

This document is **class-ready**: it explains YOLO from the ground up, then shows how to use **YOLOv8 segmentation** to extract **building footprints** from satellite/UAV imagery, and finally how to export results into **GIS formats** (GeoJSON/Shapefile).

---

## 1) What problem are we solving?

Given an overhead image (satellite/UAV), we want to find buildings.

There are three closely-related CV tasks:

- **Object detection**: outputs **bounding boxes** around each building.
- **Semantic segmentation**: outputs a **pixel mask** where each pixel is “building” or “not building” (buildings can merge together in dense areas).
- **Instance segmentation**: outputs a **separate mask per building instance** (best match for *building footprint extraction*).

**Building footprints** require *shape boundaries*, so most modern “YOLO for footprints” work uses **YOLO segmentation (YOLOv8-seg)** rather than box-only detection.

---

## 2) YOLO in one sentence

**YOLO (You Only Look Once)** is a *single-stage* model that predicts objects **in one forward pass**, turning detection into a direct regression/classification problem instead of running a classifier on many proposed regions.

---

## 3) How YOLO works (from the ground up)

### A) “Single-stage” vs “two-stage”

- **Two-stage (older style)**: propose candidate regions → refine/classify each region (accurate but slower).
- **YOLO (single-stage)**: predict boxes/classes (and optionally masks) directly from feature maps (fast and scalable).

### B) The classic YOLO mental model (grid responsibility)

A simple way to explain YOLO:

1. **Grid division**: the image is conceptually divided into grid cells (implementation uses multi-scale feature maps).
2. **Responsibility**: cells (or anchor points) are responsible for predicting objects whose centers fall there.
3. **Box regression**: predict \(x, y, w, h\) (box) plus an **objectness** score (how likely an object exists).
4. **Class probabilities**: predict class scores (e.g., “building”).
5. **Non-Maximum Suppression (NMS)**: keep the best prediction among many overlapping boxes.

### C) What the model outputs (detection)

For each predicted object:

- **Box**: center + width/height (or corner coordinates depending on representation)
- **Objectness**: confidence that a real object is present
- **Class score(s)**: probability per class

The final confidence is often *objectness × class score*.

### D) Why NMS is needed

Multiple predictions may describe the same building. **NMS** removes duplicates by:

- Sorting by confidence
- Keeping the highest-confidence box
- Removing boxes with high overlap (IoU) with the kept box

**IoU (Intersection over Union)**:

\[
IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}}
\]

---

## 4) YOLO for building footprints (instance segmentation)

### Why detection boxes are not enough

A bounding box is not a footprint. A footprint is a **polygon boundary** (or pixel mask).

### What YOLOv8-seg adds

YOLOv8 segmentation predicts:

- **Boxes + classes** (like detection)
- **A mask for each detected object** (instance segmentation)

So you can go from:

**Image → (per-building mask) → polygon footprints → GIS file**

### Why YOLO is popular in remote sensing

- **Speed at scale**: large areas can be processed quickly (tiling + fast inference).
- **Instance separation**: dense urban scenes benefit from instance masks.
- **Practical tooling**: Ultralytics makes training/inference/metrics easy.

---

## 5) Remote sensing specifics (important in class!)

These details explain why remote sensing building detection is “harder than normal photos”:

- **Very large images**: GeoTIFFs can be 10k×10k+ pixels → you must **tile/chip**.
- **Different scales**: buildings vary by size; zoom level matters.
- **Dense layouts**: roofs touch/overlap visually; shadows confuse boundaries.
- **Sensor & lighting variation**: off-nadir angles, haze, seasonal changes.
- **Georeferencing**: results must map back to real-world coordinates (CRS + transform).

---

## 6) Practical implementation (Ultralytics YOLOv8-seg)

### A) Install

```bash
pip install ultralytics opencv-python rasterio geopandas shapely numpy
```

### B) Quick inference (segmentation)

This is a **demo** using a generic segmentation model. For building footprints, you should train on a building dataset.

```python
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  # demo checkpoint; not building-specialized
results = model("path/to/image.jpg", conf=0.25, iou=0.5)

# results[0] contains boxes, classes, and masks (if any)
print(results[0].boxes)
print(results[0].masks)
```

---

## 7) Training YOLOv8-seg on a building footprint dataset

### A) Dataset choices (good for research/class demos)

- **Inria Aerial Image Labeling** (aerial imagery + building labels)
- **WHU Building Dataset**
- **SpaceNet** building footprints

### B) Label format (YOLO segmentation)

Ultralytics supports YOLO-format segmentation labels:

- One `.txt` per image
- Each line:  
  `class_id x1 y1 x2 y2 ... xn yn`  
  where polygon points are **normalized** to \([0,1]\) relative to image width/height.

Example (one building polygon, class 0 = building):

```text
0 0.10 0.20 0.18 0.20 0.18 0.30 0.10 0.30
```

### C) Minimal `data.yaml`

```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: building
```

### D) Train command

```bash
yolo task=segment mode=train model=yolov8n-seg.pt data=data.yaml imgsz=1024 epochs=100 batch=8
```

### E) Validate + metrics

```bash
yolo task=segment mode=val model=runs/segment/train/weights/best.pt data=data.yaml
```

Common metrics you can mention in class:

- **IoU** (overlap quality)
- **AP50** (Average Precision at IoU=0.50)
- **mAP50-95** (COCO-style average across IoU thresholds)

### F) Step-by-step training pipeline (detailed example)

This section walks through the **full training pipeline** for building footprint extraction using YOLOv8 instance segmentation.

#### 1. Data preparation (the most important step)

Remote sensing datasets often come with annotations in **GeoJSON** or **COCO** format. For YOLO segmentation, you must convert them into the **YOLO Segmentation Format**.

For segmentation, YOLO requires a corresponding `.txt` file for every image.  
Each line in the text file represents **one building** and contains:

```text
<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

- **class-index**: `0` for building (if you only have one class).
- **x, y**: polygon vertex coordinates **normalized** to \([0,1]\) (divide by image width/height).

**Directory structure (must match this pattern):**

```text
building_dataset/
├── images/
│   ├── train/   # .jpg or .png tiles for training
│   └── val/     # tiles for validation
└── labels/
    ├── train/   # YOLO .txt labels for train images
    └── val/     # YOLO .txt labels for val images
```

Tools like **Roboflow** are very useful: upload your labeled remote sensing dataset and export directly in **YOLOv8 segmentation format** with this structure.

#### 2. Configuration file (`data.yaml`)

YOLO needs a simple YAML file that tells it **where the images/labels are** and **what classes** you have.

```yaml
# data.yaml
path: /absolute/path/to/your/building_dataset  # Root directory of dataset
train: images/train                            # Relative path to training images
val: images/val                                # Relative path to validation images

names:
  0: building
```

#### 3. Training script (Python)

With the data and YAML ready, training is just a few lines using Ultralytics:

```python
from ultralytics import YOLO

def train_building_model():
    # 1. Load a pre-trained YOLO instance segmentation model
    # 'yolov8n-seg.pt' = nano (fastest)
    # 'yolov8m-seg.pt' or 'yolov8x-seg.pt' = larger, usually more accurate
    model = YOLO("yolov8n-seg.pt")

    # 2. Train the model
    results = model.train(
        data="data.yaml",                 # Path to your dataset configuration
        epochs=100,                       # Number of epochs (tune for your dataset)
        imgsz=640,                        # Input size (remote sensing often benefits from 1024)
        batch=16,                         # Batch size (reduce if GPU OOM)
        project="Building_Extraction",    # Folder for saving results
        name="yolov8_run1",               # Run name
        device=0                          # GPU 0; use 'cpu' if no GPU
    )

    print("Training complete!")

if __name__ == "__main__":
    train_building_model()
```

#### 4. Evaluating the results

During training, YOLO will log metrics for **boxes** and **masks**:

- **Box mAP**: how well it draws bounding boxes around buildings.
- **Mask mAP**: how well it traces the true **outline/footprint** of each building.

For building footprint extraction, **mask mAP is the critical metric** to mention in class.

After training finishes, the best weights are saved automatically as:

```text
Building_Extraction/yolov8_run1/weights/best.pt
```

You can then plug `best.pt` into the **inference / GIS export** scripts earlier in this README to test on new satellite images and create GeoJSON/Shapefiles of building footprints.

---

## 8) Export footprints to GIS (GeoJSON/Shapefile)

The most “real” remote-sensing step is turning masks into **georeferenced polygons**.

### Option A (recommended): Use the predicted mask polygons directly

Ultralytics provides per-instance mask outlines in pixel coordinates via `results[0].masks.xy`.

This script:

1. Runs YOLOv8-seg on a **GeoTIFF**
2. Converts each predicted mask outline (pixel coords) to a **Shapely polygon**
3. Uses the GeoTIFF **affine transform** to convert pixel \((x,y)\) → map coordinates
4. Writes **GeoJSON**

```python
from ultralytics import YOLO
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon

geotiff_path = "path/to/ortho.tif"
model_path = "runs/segment/train/weights/best.pt"  # your trained building model

model = YOLO(model_path)

with rasterio.open(geotiff_path) as src:
    # Read as RGB if available (assumes bands 1-3 are RGB)
    img = np.stack([src.read(1), src.read(2), src.read(3)], axis=-1)
    transform = src.transform
    crs = src.crs

results = model(img)[0]
if results.masks is None:
    raise RuntimeError("No masks predicted. Try lowering conf or use a trained building model.")

polys_map = []
scores = []

for i, xy in enumerate(results.masks.xy):
    # xy is an (N,2) array of pixel coordinates (x=col, y=row)
    if xy.shape[0] < 3:
        continue

    # Convert pixel coords -> map coords using affine transform
    xs = xy[:, 0]
    ys = xy[:, 1]
    x_map, y_map = rasterio.transform.xy(transform, ys, xs, offset="center")
    poly = Polygon(zip(x_map, y_map))

    if not poly.is_valid or poly.area == 0:
        continue

    polys_map.append(poly)
    scores.append(float(results.boxes.conf[i]) if results.boxes is not None else None)

gdf = gpd.GeoDataFrame({"score": scores}, geometry=polys_map, crs=crs)
gdf.to_file("predicted_buildings.geojson", driver="GeoJSON")
print("Wrote predicted_buildings.geojson")
```

### Option B: Raster mask → vector polygons

If you instead have a binary raster mask (0/1), you can vectorize it using `rasterio.features.shapes`. This is closer to a semantic segmentation workflow and is useful to mention as an alternative.

---

## 9) A simple talk track (5–8 minutes)

Use this as your presentation script:

- **Problem**: “We want building footprints from overhead imagery for GIS/urban analysis.”
- **Traditional approach**: “Older detectors propose regions then classify each; accurate but slow.”
- **YOLO idea**: “Single-stage—predict boxes/classes in one pass.”
- **Key steps**: “Grid/anchors → box regression + class scores → NMS.”
- **Why segmentation**: “Footprints need shape, not just boxes.”
- **YOLOv8-seg**: “Adds per-object masks, so each building becomes its own polygon.”
- **Remote sensing challenges**: “Huge images → tiling; dense buildings; shadows; georeferencing.”
- **Pipeline**: “Train on building dataset → run inference → export polygons to GeoJSON/Shapefile.”
- **Metrics**: “IoU + mAP measure overlap and detection quality.”

---

## 10) Common Q&A (good to prepare)

- **Q: Why not U-Net?**  
  **A**: U-Net is great for semantic segmentation, but **instance separation** is harder. YOLOv8-seg directly outputs instances.

- **Q: Why tile the image?**  
  **A**: GeoTIFFs are too large for GPU memory; tiling also improves batch training and keeps objects at consistent scale.

- **Q: What makes results “geospatial”?**  
  **A**: Using the image’s **CRS + affine transform** to convert pixel coordinates into real-world coordinates.

---

## 11) Where this fits in this repository

Your main project guide focuses on a segmentation workflow for **Urban Solar Energy Mapping**.

This YOLO guide is an **alternative approach** for the *“Student 1: Building Detection”* phase if you want:

- Faster inference at scale
- Instance-level building separation
- Direct footprint export to GIS formats

