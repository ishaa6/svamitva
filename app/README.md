# 🛰️ SVAMITVA Drone Feature Extractor — MVP

AI-based multiclass segmentation of SVAMITVA scheme drone orthophotos.  
Detects **buildings, roads, water bodies, railways, and utility infrastructure**.

---

## Quick Start

```bash
# 1 — Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2 — Launch the web app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## Using the App

1. Upload a GeoTIFF or PNG/JPEG orthophoto via the sidebar.
2. (Optional) Enter the path to your trained `.pth` model file.
3. Click **Run Inference**.
4. View the colour-coded segmentation map and per-class statistics.
5. Download the prediction as a PNG or `.npy` class mask.

### Demo mode
If no model file is provided, the app runs a **demo mode** that generates
illustrative blobs so you can verify the UI without a trained model.

---

## Command-Line Inference

```bash
python inference.py path/to/village.tif \
    --model best_unetpp_b4.pth \
    --out   prediction.png \
    --geojson ./geojson_output   # optional GeoJSON export
```

---

## Model

| Component      | Choice                       |
|----------------|------------------------------|
| Architecture   | UNet++ (segmentation-models-pytorch) |
| Encoder        | EfficientNet-B4 (ImageNet weights) |
| Decoder        | scSE attention               |
| Loss           | Combined CE + Dice + Focal   |
| Classes        | 6 (background + 5 features)  |
| Tile size      | 256 × 256 px (50% overlap)   |

---

## Project Structure

```
svamitva_app/
├── app.py            ← Streamlit web app (main entry point)
├── inference.py      ← Reusable inference + CLI
├── requirements.txt  ← Python dependencies
└── README.md
```

---

## Classes & Colours

| ID | Class      | Colour  |
|----|------------|---------|
| 0  | Background | ⬛ dark  |
| 1  | Building   | 🟠 orange |
| 2  | Road       | 🟣 purple |
| 3  | Water      | 🔵 blue  |
| 4  | Railway    | 🔴 red   |
| 5  | Utility    | 🟢 green |

---

## Expected Deliverables (Problem Statement alignment)

| Deliverable | Covered by |
|---|---|
| Trained + optimised AI model | `inference.py → load_model()` |
| Feature-extracted datasets   | `inference.py → save_geojson()` |
| Technical documentation      | This README |
| Final report (accuracy)      | Evaluation cells in `Final.ipynb` |
