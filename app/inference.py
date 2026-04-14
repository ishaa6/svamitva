import os
import numpy as np

NUM_CLASSES  = 6
CLASS_NAMES  = ["background", "building", "road", "water", "railway", "utility"]
CLASS_COLORS = [
    [30,  30,  30 ],
    [255, 127, 14 ],
    [148, 103, 189],
    [31,  119, 180],
    [214, 39,  40 ],
    [44,  160, 44 ],
]

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, device: str = "auto"):
    """
    Load UNet++ / EfficientNet-B4 from a saved state-dict.

    Parameters
    ----------
    model_path : str
        Path to .pth file (state-dict or {'model': state-dict}).
    device : str
        'cpu', 'cuda', or 'auto' (default).

    Returns
    -------
    torch.nn.Module  (eval mode)
    """
    import torch
    import segmentation_models_pytorch as smp

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
        decoder_attention_type="scse",
    )

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Model loaded from {model_path!r} on {device}")
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_rgb(path: str) -> tuple[np.ndarray, dict]:
    """
    Load a GeoTIFF or standard image as uint8 RGB (H, W, 3).

    Returns (rgb_array, meta_dict).
    """
    if path.lower().endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(path) as src:
            rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
            meta = {
                "crs":       str(src.crs),
                "transform": src.transform,
                "width":     src.width,
                "height":    src.height,
                "res":       src.res,
            }
    else:
        from PIL import Image
        rgb = np.array(Image.open(path).convert("RGB"))
        meta = {"crs": "N/A", "transform": None, "width": rgb.shape[1], "height": rgb.shape[0]}

    if rgb.dtype != np.uint8:
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb = np.clip(
            (rgb.astype(np.float32) - p2) / (p98 - p2 + 1e-8) * 255, 0, 255
        ).astype(np.uint8)

    return rgb, meta


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_rgb(
    rgb: np.ndarray,
    model,
    device: str,
    tile_size: int  = 256,
    stride:    int  = 128,
    batch_size: int = 4,
) -> np.ndarray:
  
    import torch

    H, W = rgb.shape[:2]
    logit_sum = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    tiles, coords = [], []

    def flush():
        if not tiles:
            return
        arr    = np.stack(tiles).astype(np.float32)
        tensor = torch.from_numpy(arr).to(device)
        with torch.no_grad():
            try:
                with torch.cuda.amp.autocast():
                    logits = model(tensor).float().cpu().numpy()
            except Exception:
                logits = model(tensor).cpu().numpy()
        for k, (y, x, ph, pw) in enumerate(coords):
            logit_sum[:, y:y+ph, x:x+pw] += logits[k, :, :ph, :pw]
            count_map[y:y+ph, x:x+pw]    += 1
        tiles.clear()
        coords.clear()

    for y in range(0, max(H - tile_size + 1, 1), stride):
        for x in range(0, max(W - tile_size + 1, 1), stride):
            y2, x2 = min(y + tile_size, H), min(x + tile_size, W)
            ph, pw  = y2 - y, x2 - x
            patch   = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            patch[:ph, :pw] = rgb[y:y2, x:x2].astype(np.float32) / 255.0
            tiles.append(patch.transpose(2, 0, 1))  # CHW
            coords.append((y, x, ph, pw))
            if len(tiles) == batch_size:
                flush()

    flush()

    count_map = np.maximum(count_map, 1)
    return (logit_sum / count_map).argmax(axis=0).astype(np.uint8)


def predict_geotiff(
    tif_path:   str,
    model,
    device:     str = "cpu",
    tile_size:  int = 256,
    stride:     int = 128,
    batch_size: int = 4,
) -> np.ndarray:
    """Convenience wrapper: load GeoTIFF and run inference."""
    rgb, _ = load_rgb(tif_path)
    return predict_rgb(rgb, model, device, tile_size, stride, batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def colorise_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class-id mask to RGB uint8 image."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[mask == cls_id] = color
    return rgb


def class_stats(mask: np.ndarray) -> dict:
    """Return pixel count and percentage for each class."""
    total = mask.size
    return {
        CLASS_NAMES[c]: {
            "pixels": int((mask == c).sum()),
            "pct":    float((mask == c).sum() / total * 100),
        }
        for c in range(NUM_CLASSES)
    }


def save_geojson(mask: np.ndarray, transform, crs, out_dir: str):
    """
    Vectorise each foreground class and write one GeoJSON per class.

    Requires rasterio, shapely, geopandas, scipy.
    """
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape
    from scipy.ndimage import binary_opening

    os.makedirs(out_dir, exist_ok=True)
    MIN_AREA_M2 = 20

    for c in range(1, NUM_CLASSES):
        binary  = (mask == c).astype("uint8")
        cleaned = binary_opening(binary, structure=np.ones((3, 3))).astype("uint8")
        polys   = [
            shape(geom)
            for geom, val in shapes(cleaned, transform=transform)
            if val == 1
        ]
        polys = [p for p in polys if p.area >= MIN_AREA_M2]
        if not polys:
            continue
        gdf = gpd.GeoDataFrame(geometry=polys, crs=crs)
        out = os.path.join(out_dir, f"{CLASS_NAMES[c]}.geojson")
        gdf.to_file(out, driver="GeoJSON")
        print(f"  Saved {len(polys):,} {CLASS_NAMES[c]} polygons → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    from PIL import Image as PILImage

    parser = argparse.ArgumentParser(description="SVAMITVA inference CLI")
    parser.add_argument("image",      help="Path to orthophoto (.tif or .png)")
    parser.add_argument("--model",    default="best_unetpp_b4.pth")
    parser.add_argument("--out",      default="output_mask.png")
    parser.add_argument("--geojson",  default="", help="Dir for GeoJSON output")
    parser.add_argument("--tile",     type=int, default=256)
    parser.add_argument("--stride",   type=int, default=128)
    parser.add_argument("--batch",    type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model}")

    model, device = load_model(args.model)
    rgb, meta     = load_rgb(args.image)
    print(f"Image: {meta['width']}×{meta['height']}  CRS: {meta['crs']}")

    print("Running inference…")
    mask     = predict_rgb(rgb, model, device, args.tile, args.stride, args.batch)
    pred_rgb = colorise_mask(mask)

    PILImage.fromarray(pred_rgb).save(args.out)
    print(f"Saved prediction → {args.out}")

    for name, s in class_stats(mask).items():
        print(f"  {name:<12}: {s['pct']:6.2f}%  ({s['pixels']:,} px)")

    if args.geojson and meta.get("transform"):
        save_geojson(mask, meta["transform"], meta["crs"], args.geojson)
