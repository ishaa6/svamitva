"""
SVAMITVA Drone Feature Extraction — MVP v3
==========================================
Run:  streamlit run app.py
"""

import os, io, random, zipfile, tempfile, time
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="SVAMITVA · Feature Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES  = ["background", "building", "road", "water", "railway", "utility"]
CLASS_COLORS = [
    [22,  26,  34 ],
    [230, 138,  50],
    [130,  90, 210],
    [45,  140, 200],
    [200,  60,  60],
    [60,  185,  90],
]
CLASS_ICONS = ["—", "Building", "Road", "Water", "Railway", "Utility"]
NUM_CLASSES  = len(CLASS_NAMES)

MODEL_URL  = "https://drive.google.com/uc?id=1ddjM68Q8UIQ1ujxkx5KtVxTpTBT3G49P"
MODEL_PATH = "model.pth"

# ═══════════════════════════════════════════════════════════════════
#  CSS — Precision Instrument / Mission-Brief aesthetic
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@300;400;500&family=Lato:wght@300;400;700&display=swap');

/* ── Reset & base ── */
:root {
  --ink:       #1a1e28;
  --ink-soft:  #2c3342;
  --paper:     #f4f1eb;
  --paper-dim: #e8e4db;
  --rule:      #d0cab8;
  --amber:     #c97d2a;
  --amber-lt:  #f0a44a;
  --teal:      #2a8a8a;
  --teal-lt:   #3db8b8;
  --warn:      #c04a2a;
  --ok:        #2a8a5a;
  --data-bg:   #1a1e28;
  --data-fg:   #a8c4d8;
  --mono:      'JetBrains Mono', monospace;
  --display:   'Syne', sans-serif;
  --body:      'Lato', sans-serif;
}

html, body, [class*="css"], .stApp {
  background-color: var(--paper) !important;
  color: var(--ink) !important;
  font-family: var(--body) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--ink) !important;
  border-right: 2px solid var(--ink-soft) !important;
}
[data-testid="stSidebar"] * { color: var(--data-fg) !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label { color: var(--data-fg) !important; }

/* Sidebar slider track */
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
  background: var(--amber) !important;
}

/* ── Typography ── */
h1 {
  font-family: var(--display) !important;
  font-size: 2rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.5px !important;
  color: var(--ink) !important;
  line-height: 1.1 !important;
}
h2, h3 {
  font-family: var(--display) !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
}

/* ── Primary action button ── */
.stButton > button[kind="primary"] {
  background: var(--ink) !important;
  color: var(--paper) !important;
  border: 2px solid var(--ink) !important;
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  border-radius: 2px !important;
  padding: 0.7rem 2rem !important;
  transition: background 0.15s, color 0.15s !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--amber) !important;
  border-color: var(--amber) !important;
  color: var(--paper) !important;
}

/* ── Secondary buttons ── */
.stButton > button {
  background: transparent !important;
  color: var(--ink) !important;
  border: 1.5px solid var(--rule) !important;
  font-family: var(--mono) !important;
  font-size: 0.73rem !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  border-radius: 2px !important;
  padding: 0.55rem 1.2rem !important;
  transition: border-color 0.15s, background 0.15s !important;
}
.stButton > button:hover {
  border-color: var(--amber) !important;
  background: rgba(201,125,42,0.07) !important;
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
  background: var(--paper-dim) !important;
  color: var(--ink) !important;
  border: 1.5px solid var(--rule) !important;
  font-family: var(--mono) !important;
  font-size: 0.7rem !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  border-radius: 2px !important;
  width: 100% !important;
  transition: border-color 0.15s !important;
}
[data-testid="stDownloadButton"] > button:hover {
  border-color: var(--amber) !important;
  background: rgba(201,125,42,0.06) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: white !important;
  border: 1.5px dashed var(--rule) !important;
  border-radius: 3px !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--amber) !important;
}
[data-testid="stFileUploader"] * { color: var(--ink) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  color: var(--ink-soft) !important;
  border-bottom: 2px solid transparent !important;
  padding: 8px 16px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--amber) !important;
  border-bottom-color: var(--amber) !important;
  font-weight: 500 !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
  background: var(--amber) !important;
}
[data-testid="stProgressBar"] > div {
  background: var(--paper-dim) !important;
  border-radius: 0 !important;
  height: 3px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--ink) !important; }

/* ── Checkbox ── */
[data-testid="stCheckbox"] span { color: var(--data-fg) !important; }
[data-testid="stCheckbox"] input:checked + span::before {
  background: var(--amber) !important;
  border-color: var(--amber) !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
  border-radius: 2px !important;
  border: 1px solid var(--rule) !important;
}

/* ── Hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Matplotlib figure background ── */
.stPlotlyChart, [data-testid="stPyplotUserWarning"] { display: none; }

/* ── Custom components ── */

.top-rule {
  width: 100%;
  height: 3px;
  background: linear-gradient(90deg, var(--amber) 0%, var(--ink) 40%, var(--paper) 100%);
  margin-bottom: 0;
}

.page-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  padding: 20px 0 16px;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 28px;
}

.page-title {
  font-family: var(--display);
  font-size: 1.85rem;
  font-weight: 800;
  color: var(--ink);
  letter-spacing: -0.5px;
  line-height: 1;
  margin: 0;
}

.page-sub {
  font-family: var(--mono);
  font-size: 0.68rem;
  color: #888;
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-top: 6px;
}

.page-meta {
  font-family: var(--mono);
  font-size: 0.62rem;
  color: #aaa;
  text-align: right;
  line-height: 2;
  letter-spacing: 0.5px;
}

.section-label {
  font-family: var(--mono);
  font-size: 0.62rem;
  font-weight: 500;
  color: var(--amber);
  letter-spacing: 3px;
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 28px 0 14px;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--rule);
}

.meta-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--rule);
  border: 1px solid var(--rule);
  border-radius: 2px;
  margin-bottom: 16px;
  overflow: hidden;
}
.meta-cell {
  background: white;
  padding: 12px 16px;
}
.meta-cell-label {
  font-family: var(--mono);
  font-size: 0.58rem;
  color: #aaa;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.meta-cell-value {
  font-family: var(--mono);
  font-size: 0.88rem;
  font-weight: 500;
  color: var(--ink);
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 1px;
  background: var(--rule);
  border: 1px solid var(--rule);
  border-radius: 2px;
  overflow: hidden;
}
.stat-cell {
  background: white;
  padding: 16px 14px;
  position: relative;
}
.stat-cell::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
}
.stat-class {
  font-family: var(--mono);
  font-size: 0.6rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #999;
  margin-bottom: 8px;
}
.stat-pct {
  font-family: var(--display);
  font-size: 1.6rem;
  font-weight: 700;
  line-height: 1;
  color: var(--ink);
}
.stat-px {
  font-family: var(--mono);
  font-size: 0.62rem;
  color: #bbb;
  margin-top: 4px;
}

.step-panel {
  background: var(--ink);
  border-radius: 2px;
  padding: 20px 24px;
  margin: 12px 0;
}
.step-item {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 9px 0;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  font-family: var(--mono);
  font-size: 0.75rem;
}
.step-item:last-child { border-bottom: none; }
.step-num {
  font-size: 0.6rem;
  letter-spacing: 1px;
  color: #444;
  width: 20px;
  flex-shrink: 0;
}
.step-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.dot-done { background: #2a8a5a; }
.dot-run  { background: var(--amber-lt); animation: blink 1.1s ease-in-out infinite; }
.dot-wait { background: #333; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

.infobar {
  background: white;
  border: 1px solid var(--rule);
  border-left: 3px solid var(--amber);
  border-radius: 0 2px 2px 0;
  padding: 10px 16px;
  font-family: var(--mono);
  font-size: 0.73rem;
  color: #555;
  margin: 10px 0;
  line-height: 1.6;
}
.warnbar {
  background: #fffbf5;
  border: 1px solid #f0c080;
  border-left: 3px solid var(--amber);
  border-radius: 0 2px 2px 0;
  padding: 10px 16px;
  font-family: var(--mono);
  font-size: 0.73rem;
  color: #886030;
  margin: 10px 0;
}
.okbar {
  background: #f5fff8;
  border: 1px solid #80c09a;
  border-left: 3px solid var(--ok);
  border-radius: 0 2px 2px 0;
  padding: 10px 16px;
  font-family: var(--mono);
  font-size: 0.73rem;
  color: #1a5c38;
  margin: 10px 0;
}
.errbar {
  background: #fff5f5;
  border: 1px solid #e08080;
  border-left: 3px solid var(--warn);
  border-radius: 0 2px 2px 0;
  padding: 10px 16px;
  font-family: var(--mono);
  font-size: 0.73rem;
  color: #8c2020;
  margin: 10px 0;
}

.geo-count-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid var(--paper-dim);
  font-family: var(--mono);
  font-size: 0.73rem;
}
.geo-swatch {
  width: 10px; height: 10px;
  border-radius: 1px;
  flex-shrink: 0;
}
.geo-name { color: var(--ink); min-width: 80px; }
.geo-n { color: #aaa; }

.footer-rule {
  height: 1px;
  background: var(--rule);
  margin: 40px 0 16px;
}
.footer-text {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: #ccc;
  letter-spacing: 2px;
  text-transform: uppercase;
  text-align: center;
}

/* Sidebar section label */
.sb-label {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: #555;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  margin: 20px 0 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid #2a2e3a;
}
.sb-title {
  font-family: var(--display);
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--paper);
  letter-spacing: 1px;
  padding: 16px 0 4px;
}
.legend-row {
  display: flex; align-items: center; gap: 10px;
  padding: 4px 0;
  font-family: var(--mono);
  font-size: 0.72rem;
  color: #8a9ab0;
}
.legend-swatch {
  width: 8px; height: 8px;
  border-radius: 1px;
  flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  Pure-Python helpers (no UI)
# ═══════════════════════════════════════════════════════════════════
def colorise_mask(mask_np):
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for c, col in enumerate(CLASS_COLORS):
        rgb[mask_np == c] = col
    return rgb


def compute_stats(mask):
    total = mask.size
    return {
        CLASS_NAMES[c]: {
            "pixels": int((mask == c).sum()),
            "pct":    float((mask == c).sum() / total * 100),
        }
        for c in range(NUM_CLASSES)
    }


def legend_figure():
    fig, ax = plt.subplots(figsize=(5.5, 1.4))
    ax.axis("off")
    patches = [
        mpatches.Patch(facecolor=[v/255 for v in CLASS_COLORS[i]],
                       edgecolor="#d0cab8", linewidth=0.6, label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    ax.legend(handles=patches, loc="center", fontsize=9,
              framealpha=0, ncol=6, labelcolor="#2c3342",
              handlelength=1.2, handleheight=1.0,
              borderpad=0, labelspacing=0.4, columnspacing=1.2)
    fig.patch.set_facecolor("#f4f1eb")
    fig.tight_layout(pad=0.2)
    return fig


def ensure_model(url, path, ph):
    if os.path.exists(path):
        return True
    try:
        import gdown
        ph.markdown("<div class='infobar'>↓ Fetching model weights from Google Drive…</div>",
                    unsafe_allow_html=True)
        gdown.download(url, path, quiet=True)
        ph.markdown("<div class='okbar'>✓ Model weights downloaded.</div>",
                    unsafe_allow_html=True)
        return True
    except Exception as e:
        ph.markdown(f"<div class='errbar'>✗ Download failed — {e}</div>",
                    unsafe_allow_html=True)
        return False


@st.cache_resource(show_spinner=False)
def load_model(path):
    import torch
    import segmentation_models_pytorch as smp
    m = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4", encoder_weights=None,
        in_channels=3, classes=NUM_CLASSES,
        activation=None, decoder_attention_type="scse",
    )
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    m.load_state_dict(state)
    m.eval()
    return m


def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _flush_batch(tiles, coords, logit_sum, count_map, model, device):
    import torch
    arr = np.stack(tiles).astype(np.float32)
    t   = torch.from_numpy(arr).to(device)
    with torch.no_grad():
        try:
            with torch.cuda.amp.autocast():
                logits = model(t).float().cpu().numpy()
        except Exception:
            logits = model(t).cpu().numpy()
    for k, (y, x, ph, pw) in enumerate(coords):
        logit_sum[:, y:y+ph, x:x+pw] += logits[k, :, :ph, :pw]
        count_map[y:y+ph, x:x+pw]    += 1


def predict_image(rgb_np, model, device, tile_sz, stride_sz, bs, pbar=None):
    import torch
    H, W = rgb_np.shape[:2]
    logit_sum = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    tiles, coords = [], []
    ys = list(range(0, max(H - tile_sz + 1, 1), stride_sz))
    xs = list(range(0, max(W - tile_sz + 1, 1), stride_sz))
    total = max(len(ys) * len(xs), 1)
    done  = 0
    for y in ys:
        for x in xs:
            y2, x2 = min(y+tile_sz, H), min(x+tile_sz, W)
            ph, pw = y2-y, x2-x
            p = np.zeros((tile_sz, tile_sz, 3), dtype=np.float32)
            p[:ph, :pw] = rgb_np[y:y2, x:x2].astype(np.float32) / 255.0
            tiles.append(p.transpose(2,0,1)); coords.append((y,x,ph,pw))
            if len(tiles) == bs:
                _flush_batch(tiles, coords, logit_sum, count_map, model, device)
                done += len(tiles); tiles.clear(); coords.clear()
                if pbar: pbar.progress(min(done/total, 1.0))
    if tiles:
        _flush_batch(tiles, coords, logit_sum, count_map, model, device)
        if pbar: pbar.progress(1.0)
    return (logit_sum / np.maximum(count_map, 1)).argmax(0).astype(np.uint8)


def demo_predict(rgb_np):
    H, W = rgb_np.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    rng  = np.random.default_rng(42)
    for c in range(1, NUM_CLASSES):
        for _ in range(rng.integers(2, 6)):
            cy = rng.integers(H//4, 3*H//4); cx = rng.integers(W//4, 3*W//4)
            ry = rng.integers(H//16, H//6);  rx = rng.integers(W//16, W//6)
            yy, xx = np.ogrid[-cy:H-cy, -cx:W-cx]
            mask[(yy/ry)**2 + (xx/rx)**2 <= 1] = c
    return mask


def vectorise_to_geojson(mask, transform, crs, min_area_m2=20):
    try:
        import geopandas as gpd
        from rasterio.features import shapes
        from shapely.geometry import shape
        from scipy.ndimage import binary_opening
    except ImportError as e:
        return None, str(e)
    results = {}
    for c in range(1, NUM_CLASSES):
        binary  = (mask == c).astype("uint8")
        cleaned = binary_opening(binary, structure=np.ones((3,3))).astype("uint8")
        polys   = [shape(g) for g, v in shapes(cleaned, transform=transform) if v == 1]
        polys   = [p for p in polys if p.area >= min_area_m2]
        if polys:
            import geopandas as gpd
            results[CLASS_NAMES[c]] = gpd.GeoDataFrame(
                {"class": [CLASS_NAMES[c]]*len(polys), "class_id": [c]*len(polys)},
                geometry=polys, crs=crs,
            )
    return results, None


def _gdf_to_bytes(gdf):
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
        tmp = f.name
    try:
        gdf.to_file(tmp, driver="GeoJSON")
        return open(tmp, "rb").read()
    finally:
        os.unlink(tmp)


def build_geojson_zip(d):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, gdf in d.items():
            zf.writestr(f"{name}.geojson", _gdf_to_bytes(gdf))
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div class='sb-title'>SVAMITVA</div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-label'>Inference Parameters</div>", unsafe_allow_html=True)

    tile_size  = st.slider("Tile size (px)",        128, 512, 256, 64)
    stride     = st.slider("Stride (px)",            64, tile_size, tile_size//2, 64)
    batch_size = st.slider("Batch size",              1, 16, 4)
    min_area   = st.slider("Min polygon area (m²)",   5, 200, 20, 5)
    use_demo   = st.checkbox("Demo mode if no model", value=True)

    st.markdown("<div class='sb-label' style='margin-top:24px'>Class Legend</div>",
                unsafe_allow_html=True)
    for i, name in enumerate(CLASS_NAMES):
        r, g, b = CLASS_COLORS[i]
        st.markdown(
            f"<div class='legend-row'>"
            f"<span class='legend-swatch' style='background:rgb({r},{g},{b})'></span>"
            f"{name}</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════
#  Page header
# ═══════════════════════════════════════════════════════════════════
st.markdown("<div class='top-rule'></div>", unsafe_allow_html=True)
st.markdown("""
<div class='page-header'>
  <div>
    <div class='page-title'>Orthophoto Feature Intelligence</div>
    <div class='page-sub'>SVAMITVA Scheme · AI-Based Land Feature Extraction</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  01 — Upload
# ═══════════════════════════════════════════════════════════════════
st.markdown("<div class='section-label'>01 — Upload Orthophoto</div>",
            unsafe_allow_html=True)

uploaded = st.file_uploader(
    "GeoTIFF / PNG / JPEG  ·  3-band RGB",
    type=["tif","tiff","png","jpg","jpeg"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.stop()


# ═══════════════════════════════════════════════════════════════════
#  Read image
# ═══════════════════════════════════════════════════════════════════
fname         = uploaded.name.lower()
geo_transform = None
geo_crs       = None

with st.spinner("Reading image…"):
    if fname.endswith((".tif",".tiff")):
        try:
            import rasterio
            with rasterio.open(uploaded) as src:
                rgb           = src.read([1,2,3]).transpose(1,2,0)
                geo_crs       = src.crs
                geo_transform = src.transform
                crs_str       = str(src.crs)
                res_str       = f"{src.res[0]:.5f} m"
        except Exception as e:
            st.warning(f"rasterio: {e} — falling back to PIL.")
            rgb = np.array(Image.open(uploaded).convert("RGB"))
            crs_str = "—"; res_str = "—"
    else:
        rgb = np.array(Image.open(uploaded).convert("RGB"))
        crs_str = "—"; res_str = "—"

if rgb.dtype != np.uint8:
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb.astype(np.float32)-p2)/(p98-p2+1e-8)*255, 0, 255).astype(np.uint8)

H, W = rgb.shape[:2]

# Metadata strip
st.markdown(f"""
<div class='meta-grid'>
  <div class='meta-cell'>
    <div class='meta-cell-label'>Width</div>
    <div class='meta-cell-value'>{W} px</div>
  </div>
  <div class='meta-cell'>
    <div class='meta-cell-label'>Height</div>
    <div class='meta-cell-value'>{H} px</div>
  </div>
  <div class='meta-cell'>
    <div class='meta-cell-label'>CRS</div>
    <div class='meta-cell-value' style='font-size:0.75rem'>{crs_str[:22] if crs_str != "—" else "—"}</div>
  </div>
  <div class='meta-cell'>
    <div class='meta-cell-label'>Resolution</div>
    <div class='meta-cell-value'>{res_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.image(rgb, caption=uploaded.name, use_column_width=True)


# ═══════════════════════════════════════════════════════════════════
#  02 — Analyse
# ═══════════════════════════════════════════════════════════════════
st.markdown("<div class='section-label'>02 — Analyse</div>", unsafe_allow_html=True)

run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

if run_btn:
    steps_ph  = st.empty()
    status_ph = st.empty()
    prog_ph   = st.empty()

    def render_steps(cur):
        labels = [
            "Verify model weights",
            "Initialise network  (UNet++ / EfficientNet-B4)",
            "Tile & infer",
            "Assemble prediction",
        ]
        rows = ""
        for i, lbl in enumerate(labels):
            if   i < cur:  cls, col = "dot-done", "#2a8a5a"
            elif i == cur: cls, col = "dot-run",  "#f0a44a"
            else:          cls, col = "dot-wait", "#333"
            rows += (
                f"<div class='step-item'>"
                f"<span class='step-num'>{i+1:02d}</span>"
                f"<span class='step-dot {cls}'></span>"
                f"<span style='color:{col}'>{lbl}</span>"
                f"</div>"
            )
        steps_ph.markdown(
            f"<div class='step-panel'>{rows}</div>",
            unsafe_allow_html=True,
        )

    render_steps(0)
    model_ready = ensure_model(MODEL_URL, MODEL_PATH, status_ph)

    render_steps(1)
    model_obj  = None
    device_str = "cpu"

    if model_ready and os.path.exists(MODEL_PATH):
        try:
            model_obj  = load_model(MODEL_PATH)
            device_str = get_device()
            status_ph.markdown(
                f"<div class='okbar'>✓ Model loaded on {device_str.upper()}</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            status_ph.markdown(
                f"<div class='errbar'>✗ Model failed to load — {e}</div>",
                unsafe_allow_html=True,
            )

    if model_obj is None and use_demo:
        status_ph.markdown(
            "<div class='warnbar'>⚠ Demo mode active — predictions are illustrative only</div>",
            unsafe_allow_html=True,
        )

    render_steps(2)
    pbar = prog_ph.progress(0.0)

    if model_obj is not None:
        mask = predict_image(rgb, model_obj, device_str,
                             tile_size, stride, batch_size, pbar)
    else:
        for p in np.linspace(0, 1, 40):
            pbar.progress(float(p)); time.sleep(0.015)
        mask = demo_predict(rgb)

    render_steps(3); time.sleep(0.25); render_steps(4)
    prog_ph.empty()

    st.session_state.update({
        "mask": mask, "rgb": rgb,
        "geo_transform": geo_transform, "geo_crs": geo_crs,
        "demo_mode": (model_obj is None),
    })


# ═══════════════════════════════════════════════════════════════════
#  03 — Results
# ═══════════════════════════════════════════════════════════════════
if "mask" not in st.session_state:
    st.stop()

mask          = st.session_state["mask"]
rgb           = st.session_state["rgb"]
geo_transform = st.session_state["geo_transform"]
geo_crs       = st.session_state["geo_crs"]
is_demo       = st.session_state.get("demo_mode", False)
pred_rgb      = colorise_mask(mask)

st.markdown("<div class='section-label'>03 — Segmentation Result</div>",
            unsafe_allow_html=True)

if is_demo:
    st.markdown("<div class='warnbar'>Demo mode — results are illustrative. "
                "Provide a trained model for real predictions.</div>",
                unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="small")
with c1:
    st.image(rgb,      caption="Input", use_column_width=True)
with c2:
    st.image(pred_rgb, caption="Predicted Segmentation", use_column_width=True)

# Legend (matplotlib — light background)
st.pyplot(legend_figure(), use_container_width=False)


# ═══════════════════════════════════════════════════════════════════
#  04 — Statistics
# ═══════════════════════════════════════════════════════════════════
st.markdown("<div class='section-label'>04 — Class Statistics</div>",
            unsafe_allow_html=True)

stats = compute_stats(mask)
cells = ""
for i, (name, s) in enumerate(stats.items()):
    r, g, b  = CLASS_COLORS[i]
    hex_col  = "#{:02x}{:02x}{:02x}".format(r, g, b)
    cells += f"""
    <div class='stat-cell' style='border-top-color:{hex_col}'>
      <div class='stat-class'>{name}</div>
      <div class='stat-pct'>{s['pct']:.1f}<span style='font-family:var(--mono);font-size:0.9rem;font-weight:400;color:#999'>%</span></div>
      <div class='stat-px'>{s['pixels']:,} px</div>
    </div>"""

st.markdown(f"<div class='stat-grid'>{cells}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  05 — Export
# ═══════════════════════════════════════════════════════════════════
st.markdown("<div class='section-label'>05 — Export</div>", unsafe_allow_html=True)

tab_img, tab_geo = st.tabs(["Image Exports", "GeoJSON / Vectors"])

with tab_img:
    st.markdown("<br>", unsafe_allow_html=True)
    d1, d2 = st.columns(2, gap="small")
    with d1:
        buf = io.BytesIO()
        Image.fromarray(pred_rgb).save(buf, format="PNG")
        st.download_button("Download  ·  Segmentation RGB (PNG)",
                           data=buf.getvalue(),
                           file_name="prediction_rgb.png",
                           mime="image/png",
                           use_container_width=True)
    with d2:
        buf2 = io.BytesIO()
        np.save(buf2, mask)
        st.download_button("Download  ·  Class Mask (.npy)",
                           data=buf2.getvalue(),
                           file_name="prediction_classes.npy",
                           mime="application/octet-stream",
                           use_container_width=True)

with tab_geo:
    st.markdown("<br>", unsafe_allow_html=True)
    has_georef = geo_transform is not None and geo_crs is not None

    if not has_georef:
        st.markdown("""
        <div class='warnbar'>
          No spatial reference found. Upload a GeoTIFF to enable
          georeferenced polygon export.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='infobar'>
          Each predicted class is vectorised into polygons and filtered by
          minimum area. One <code>.geojson</code> per class, bundled as a ZIP.
        </div>
        """, unsafe_allow_html=True)

        if st.button("Generate GeoJSON", use_container_width=False):
            with st.spinner("Vectorising…"):
                geo_dict, err = vectorise_to_geojson(
                    mask, geo_transform, geo_crs, min_area_m2=min_area)

            if err:
                st.markdown(f"<div class='errbar'>✗ {err}</div>",
                            unsafe_allow_html=True)
            elif not geo_dict:
                st.markdown("<div class='warnbar'>No polygons above the area threshold.</div>",
                            unsafe_allow_html=True)
            else:
                total_polys = sum(len(g) for g in geo_dict.values())
                st.markdown(
                    f"<div class='okbar'>✓ {total_polys:,} polygons vectorised "
                    f"across {len(geo_dict)} classes</div>",
                    unsafe_allow_html=True,
                )

                # Per-class counts
                rows = ""
                for cls_name, gdf in geo_dict.items():
                    idx = CLASS_NAMES.index(cls_name)
                    r2, g2, b2 = CLASS_COLORS[idx]
                    rows += (
                        f"<div class='geo-count-row'>"
                        f"<span class='geo-swatch' style='background:rgb({r2},{g2},{b2})'></span>"
                        f"<span class='geo-name'>{cls_name}</span>"
                        f"<span class='geo-n'>{len(gdf):,} polygons</span>"
                        f"</div>"
                    )
                st.markdown(
                    f"<div style='background:white;border:1px solid var(--rule);"
                    f"border-radius:2px;padding:12px 16px;margin:12px 0'>{rows}</div>",
                    unsafe_allow_html=True,
                )

                zip_buf = build_geojson_zip(geo_dict)
                st.download_button(
                    "Download All Classes (ZIP)",
                    data=zip_buf,
                    file_name="feature_extraction.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

                gcols = st.columns(min(len(geo_dict), 3), gap="small")
                for j, (cls_name, gdf) in enumerate(geo_dict.items()):
                    with gcols[j % 3]:
                        st.download_button(
                            f"{cls_name}.geojson",
                            data=_gdf_to_bytes(gdf),
                            file_name=f"{cls_name}.geojson",
                            mime="application/geo+json",
                            use_container_width=True,
                        )
