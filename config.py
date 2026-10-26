"""Global settings."""
import os

RANDOM_SEED = 42
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("KGMSPF_OUTPUT_DIR", os.path.join(PROJECT_DIR, "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_FOLDS = 5
TIME_POINTS = [12, 36, 60, 120]
BOOTSTRAP_N = 2000
FIGURE_DPI = 300  # journal requirement: >= 300 dpi raster figures

# Peak-memory guard for the KG-MSPF network.  Patients are independent in the
# per-patient heterogeneous graph, so the risk for a batch of patients can be
# evaluated in chunks of this size with gradient checkpointing instead of a
# single full-batch forward.  This only bounds peak RAM (activations are
# recomputed during backward); 0 keeps the original single-forward behaviour.
TRAIN_CHUNK_SIZE = int(os.environ.get("KGMSPF_TRAIN_CHUNK", "128"))

COLOR_PALETTE = {
    "proposed": "#C0392B",
    "baseline1": "#2E86C1", "baseline2": "#5DADE2",
    "baseline3": "#48C9B0", "baseline4": "#F4D03F",
    "baseline5": "#E59866", "baseline6": "#AF7AC5",
}
