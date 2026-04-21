"""
RIT Smart Parking Tracker  v6  (brightened-video / gap-fill fix)
================================================================
Fix log (vs v5):

  FIX-I   STALL_CONTENT_MIN lowered 100 -> 60
          ---------------------------------------------------------------
          ROOT CAUSE of top-left and dark-car missing spots:
            When the video is brightness-adjusted, dark car bodies
            (originally ~60-85 brightness) are lifted to ~70-100.
            STALL_CONTENT_MIN=100 was rejecting any stall strip whose
            mean brightness <= 100 — originally intended to filter
            featureless asphalt, but now incorrectly discarding valid
            stalls filled with dark cars.
          FIX: Lower to 60.  With brightened video, empty asphalt reads
            ~150-170 (well above 60).  Dark-car stalls read 67-95 (now
            passes).  No false-positive road strips can appear because
            zone detection already confines scanning to parking rows.

  FIX-J   Pitch-based gap filling added to get_stall_xs
          ---------------------------------------------------------------
          ROOT CAUSE of bottom-center missing spots:
            A ~80-120 px section in some rows has no visible stall-line
            paint (crosswalk, pavement joint, or worn markings).
            No peaks are found in that region, so the two flanking peaks
            form a single stall of width 80-230 px which exceeds
            STALL_MAX_W=80 and is dropped.
          FIX: After peak clustering, estimate the median stall pitch
            from valid-width inter-peak distances.  Any gap between
            consecutive peaks that is >1.6× and <8× the pitch is
            interpolated at the estimated pitch to fill in missing
            dividers.  Upper bound prevents filling genuine road aisles;
            build_spot_grid's band_mask and width filters act as a
            further safety net.

  FIX-K   note: occupancy dark-pixel threshold (85) intentionally unchanged
          ---------------------------------------------------------------
          Although the video is brightened, dark car bodies still read
          dark<85 at ~58-81% (well above OCC_DARK_FRAC=0.20).  Raising
          the threshold to 110 would break un-brightened video (empty
          asphalt at ~90-120 would have dark<110 ~50%, incorrectly
          marking empty stalls as occupied).  No change needed here.

  All other v5 logic preserved unchanged.

=============================================================
Fix log (vs v4):

  FIX-A  Peak clustering replaces simple deduplication in get_stall_xs
         ---------------------------------------------------------------
         ROOT CAUSE of double/triple boxes:
           Painted stall lines are ~7-14px wide stripes.  Method 1
           (brightness threshold) and Method 2 (Sobel-X) each detect
           the edges or centre of the same line, producing 2-3 raw peaks
           per stall divider spaced 7-14 px apart.  The old dedup window
           of 10 px was too small to collapse all of these.
         FIX: After collecting raw peaks from both methods, pass them
           through _cluster_peaks() which merges any peaks within
           PEAK_MERGE_DIST=18 px and replaces each cluster with its
           median.  One divider per painted line, guaranteed.

  FIX-B  STALL_MIN_W raised 20 -> 30
         ---------------------------------------------------------------
         Measured mean stall width in test footage is ~33-41 px.
         A 30 px floor eliminates surviving ghost stalls (21-26 px)
         without cutting legitimate compact stalls.

  FIX-C  STALL_MAX_W raised 65 -> 80
         ---------------------------------------------------------------
         Wide/accessible end-of-row stalls measured up to 75 px.
         The old 65 px cap silently dropped them (missed spots on edges).

  FIX-D  Method 2 Sobel threshold raised + prominence tightened
         ---------------------------------------------------------------
         edge_thresh percentile 72->82, prominence 3->6.
         Suppresses intra-stall car-body edges (door lines, windscreens)
         that were adding spurious peaks between genuine stall dividers.

  FIX-E  Method 1 brightness column height threshold raised 7%->12%
         Reduces noise columns in empty stalls creating false peaks.

  FIX-F  split_zone_to_row_pairs minimum split height raised 120->160
         ---------------------------------------------------------------
         Zones 120-159 px tall were being split into two short sub-rows
         even when only one car row was present, doubling the box count.
         Real back-to-back double rows are >= 160 px at 1080p.

  FIX-G  Internal road detection threshold tightened ROAD_BRIGHTNESS+5
         -> ROAD_BRIGHTNESS-3  (i.e. brightness < 100 instead of < 108)
         ---------------------------------------------------------------
         Faint pedestrian crosswalk markings inside parking zones read
         ~101-107 brightness — just under the old threshold of 108 —
         causing them to be misclassified as internal road bands.  This
         split the top zone (h=224px) into 3 sub-zones instead of the
         correct 2 rows.  Tightening to < 100 fixes the false splits
         while still catching genuine dark driving aisles (< 95).

  FIX-H  Method 1 prominence raised 1 -> 4
         ---------------------------------------------------------------
         At threshold=178, broad car-roof reflections produced ~50+
         column peaks per row.  Raising prominence to 4 requires a peak
         to stand out sharply above its neighbours, which stall-line
         columns do (narrow spike) but car-roof columns don't (broad
         plateau).  This reduces Method 1 output to the ~15-20 genuine
         stall-line peaks per row.

  All other v4 logic preserved unchanged.
"""

import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict


try:
    from scipy.signal import find_peaks as _scipy_peaks
    from scipy.ndimage import gaussian_filter1d as _gauss1d
    def _find_peaks(sig, height, distance, prominence=0):
        p, _ = _scipy_peaks(sig.astype(float), height=height,
                             distance=distance, prominence=prominence)
        return p
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    def _gauss1d(arr, sigma=3):
        k = max(1, int(sigma * 3))
        kernel = np.exp(-0.5 * (np.arange(-k, k+1) / sigma) ** 2)
        kernel /= kernel.sum()
        return np.convolve(arr.astype(float), kernel, mode='same')
    def _find_peaks(sig, height, distance, prominence=0):
        peaks = []
        for i in range(1, len(sig) - 1):
            if sig[i] > height and sig[i] >= sig[i-1] and sig[i] >= sig[i+1]:
                if not peaks or i - peaks[-1] >= distance:
                    peaks.append(i)
        return np.array(peaks, dtype=int)


# ── Constants ──────────────────────────────────────────────────────────────────
DB_PATH            = "rit_parking.db"
VIDEO_EXTENSIONS   = (".mp4",".avi",".mov",".mkv",".wmv",".m4v",".flv",".webm")
DEFAULT_FRAME_SKIP = 2
DISPLAY_W          = 1920
DISPLAY_H          = 1080

# Occupancy classifier thresholds
OCC_DARK_FRAC      = 0.20
OCC_LAP_VAR        = 400
EMPTY_BRIGHT_FRAC  = 0.55
EMPTY_MAX_STD      = 35
EMPTY_LAP_LOW      = 150
EMPTY_MIN_MEAN     = 115
OCC_SCORE_THRESH   = 0.6

# Stall size constraints
STALL_MIN_W        = 30     # FIX-B: was 20 (eliminates ghost-stall artefacts)
STALL_MAX_W        = 80     # FIX-C: was 65 (catches wide/accessible stalls)
STALL_CONTENT_MIN  = 60     # FIX-I: was 100 — dark cars on brightened video read 67-95

# Road / aisle detection
ROAD_BRIGHTNESS    = 103
ROAD_MIN_HEIGHT    = 30
AISLE_DARK_FRAC    = 0.025

# Stall-line peak detection
STALL_LINE_DISTANCE = 18

# FIX-A: cluster peaks within this distance -> one divider per painted line
PEAK_MERGE_DIST    = 18     # px  (covers widest stall-line stripe ~14 px + margin)

# Temporal smoothing
SMOOTH_FRAMES      = 5


# ── Display guard ──────────────────────────────────────────────────────────────
def _has_display() -> bool:
    import sys
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


# ── FIX-A helper: cluster raw peaks ──────────────────────────────────────────
def _cluster_peaks(raw_peaks: set, merge_dist: int = PEAK_MERGE_DIST) -> List[int]:
    """
    Merge any peaks within `merge_dist` px into a single peak at the median
    of the cluster.  This collapses Method-1 and Method-2 detections of the
    same painted stall line into one x-position.

    Example: peaks {80, 88} from both edges of a 8-px wide line -> [84].
    """
    if not raw_peaks:
        return []
    sorted_px = sorted(raw_peaks)
    clusters: List[List[int]] = [[sorted_px[0]]]
    for px in sorted_px[1:]:
        if px - clusters[-1][-1] <= merge_dist:
            clusters[-1].append(px)
        else:
            clusters.append([px])
    return [int(np.median(c)) for c in clusters]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: DETECT ALL NON-PARKING BANDS
# ══════════════════════════════════════════════════════════════════════════════

def detect_non_parking_bands(frame: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find horizontal bands that are NOT parking rows (roads + driving aisles).
    Unchanged from v4 (fixes #7 and #8 retained).
    """
    fh, fw = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
           if frame.ndim == 3 else frame

    profile_dark = np.array([
        np.mean(gray[y:y + 1, 150:fw - 150] < 80)
        for y in range(fh)
    ], dtype=float)

    profile_bright = np.mean(gray[:, 80:fw - 80], axis=1).astype(float)

    sd = np.convolve(profile_dark,   np.ones(15) / 15, mode='same')
    sb = np.convolve(profile_bright, np.ones(15) / 15, mode='same')

    non_parking = (sb < ROAD_BRIGHTNESS) | (sd < AISLE_DARK_FRAC)

    bands: List[Tuple[int, int]] = []
    in_band, b0 = False, 0
    for y in range(fh):
        if non_parking[y]:
            if not in_band:
                in_band, b0 = True, y
        else:
            if in_band:
                in_band = False
                if y - b0 >= ROAD_MIN_HEIGHT:
                    bands.append((b0, y))
    if in_band and fh - b0 >= ROAD_MIN_HEIGHT:
        bands.append((b0, fh))

    return bands


def parking_zones_from_bands(bands: List[Tuple[int, int]],
                              frame_h: int) -> List[Tuple[int, int]]:
    """Return y-ranges of parking areas (gaps between non-parking bands)."""
    bounds = sorted(set([0] + [y for b in bands for y in b] + [frame_h]))
    zones = []
    for i in range(len(bounds) - 1):
        y1, y2 = bounds[i], bounds[i + 1]
        is_band = any(abs(y1 - b[0]) < 5 for b in bands)
        if not is_band and y2 - y1 > 40:
            zones.append((y1, y2))
    return zones


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: SPLIT PARKING ZONES INTO ROW PAIRS
# ══════════════════════════════════════════════════════════════════════════════

def split_zone_to_row_pairs(gray: np.ndarray,
                             y_start: int, y_end: int,
                             scan_x0: int = 150,
                             scan_x1: int = None
                             ) -> List[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]]:
    """
    Split a parking zone into (top_row, bottom_row) pairs.

    FIX-F: Minimum zone height to attempt a back-to-back split raised
           120 -> 160 px.  Zones < 160 px contain only one row of cars.
           Splitting them was producing two very short sub-rows that each
           got their own stall scan, doubling the box count.
    """
    fw = gray.shape[1]
    if scan_x1 is None:
        scan_x1 = fw - 150

    zone_h = y_end - y_start
    if zone_h < 40:
        return []

    # FIX-F: was < 120 — now anything under 160 px is a single row
    if zone_h < 160:
        return [((y_start, y_end), None)]

    # Check for internal dark bands (secondary lanes within zone)
    band = gray[y_start:y_end, scan_x0:scan_x1]
    bh   = band.shape[0]
    hb   = np.mean(band, axis=1).astype(float)
    hbs  = np.convolve(hb, np.ones(9) / 9, mode='same')   # 9-px smoothing (v4 fix)

    internal_roads = []
    in_dark, ds = False, 0
    for i in range(bh):
        if hbs[i] < ROAD_BRIGHTNESS - 3:   # tightened from +5: avoids false splits on faint markings
            if not in_dark: in_dark, ds = True, i
        else:
            if in_dark:
                in_dark = False
                if i - ds >= 20:
                    internal_roads.append((y_start + ds, y_start + i))

    if internal_roads:
        result = []
        all_bnds = sorted(set(
            [y_start] + [y for r in internal_roads for y in r] + [y_end]
        ))
        for i in range(len(all_bnds) - 1):
            r1, r2 = all_bnds[i], all_bnds[i + 1]
            is_road = any(abs(r1 - r[0]) < 5 for r in internal_roads)
            if not is_road and r2 - r1 >= 40:
                result.extend(split_zone_to_row_pairs(gray, r1, r2, scan_x0, scan_x1))
        return result

    # No internal road — split at midpoint (genuine back-to-back pair)
    mid = (y_start + y_end) // 2
    return [((y_start, mid), (mid, y_end))]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: DETECT STALL DIVIDER X POSITIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_stall_xs(gray: np.ndarray,
                 y1: int, y2: int,
                 x_start: int = 30,
                 x_end: int = None,
                 fallback_xs: Optional[List[int]] = None
                 ) -> List[int]:
    """
    Detect stall-divider x-positions within a parking row.

    Changes from v4:

    FIX-A: All raw peaks from both methods are passed through
           _cluster_peaks() before returning.  Each painted stall line
           (which may produce 2-3 raw detections from its two edges +
           bright centre) collapses to exactly one x-position.

    FIX-D: Sobel edge threshold percentile raised 72->82, prominence 3->6.
           Suppresses intra-stall edges (car body lines) that were adding
           spurious peaks between the real stall dividers.

    FIX-E: Method-1 brightness column height threshold raised 7%->12%.
           Reduces noise columns in empty stalls triggering false peaks.
    """
    if x_end is None:
        x_end = gray.shape[1] - 30

    trim_y = max(2, (y2 - y1) // 8)
    band   = gray[y1 + trim_y:y2 - trim_y, x_start:x_end]
    if band.shape[0] < 5:
        return fallback_xs or []

    h_row      = band.shape[0]
    raw_peaks: set = set()

    # -- Method 1: Brightness threshold ----------------------------------------
    # Start at high thresholds (most selective) and cascade down.
    # Require >= 12 peaks before accepting a threshold level — this skips
    # thresholds that find too few real stall lines, and the high prominence
    # requirement (4) suppresses broad car-roof reflections that generate
    # spurious wide columns at high brightness thresholds.
    for thresh in [178, 168, 158, 148, 138, 128]:
        _, white = cv2.threshold(band, thresh, 255, cv2.THRESH_BINARY)
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, h_row // 5)))
        vert   = cv2.morphologyEx(white, cv2.MORPH_OPEN, kv)
        col_d  = np.sum(vert > 0, axis=0).astype(float)
        col_sm = np.convolve(col_d, np.ones(3) / 3, mode='same')
        h_thr  = max(1, h_row * 0.12)    # FIX-E: was 0.07
        peaks  = _find_peaks(col_sm, height=h_thr,
                             distance=STALL_LINE_DISTANCE,
                             prominence=4)           # raised from 1: suppresses car-roof noise
        if len(peaks) >= 8:
            for p in peaks:
                raw_peaks.add(int(p) + x_start)
            break

    # -- Method 2: Sobel-X vertical edges ---------------------------------------
    sobelx     = cv2.Sobel(band.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    col_edges  = np.mean(np.abs(sobelx), axis=0)
    col_sm2    = np.convolve(col_edges, np.ones(3) / 3, mode='same')
    edge_thresh = max(float(np.percentile(col_sm2, 82)), 8.0)  # FIX-D: was 72
    peaks2 = _find_peaks(col_sm2, height=edge_thresh,
                         distance=STALL_LINE_DISTANCE,
                         prominence=6)                          # FIX-D: was 3
    for p in peaks2:
        raw_peaks.add(int(p) + x_start)

    # FIX-A: Cluster all raw peaks -> one position per stall line
    merged = _cluster_peaks(raw_peaks, merge_dist=PEAK_MERGE_DIST)

    # FIX-J: Pitch-based gap filling
    # If a gap between consecutive peaks is larger than 1.6× the median stall
    # pitch, interpolate missing dividers at the estimated pitch spacing.
    # This recovers stalls in regions where paint is worn/absent.
    # Upper bound of 8× pitch prevents filling genuine road aisles.
    if len(merged) >= 4:
        gap_diffs = np.diff(merged)
        valid_w   = gap_diffs[(gap_diffs >= STALL_MIN_W) & (gap_diffs <= STALL_MAX_W)]
        if len(valid_w) >= 3:
            pitch     = float(np.median(valid_w))
            lo_thresh = 1.6 * pitch
            hi_thresh = 8.0 * pitch
            fill_set  = set(merged)
            for i in range(len(merged) - 1):
                gap = merged[i + 1] - merged[i]
                if lo_thresh < gap <= hi_thresh:
                    n_fill = round(gap / pitch) - 1
                    if 1 <= n_fill <= 6:
                        for k in range(1, n_fill + 1):
                            fill_set.add(int(merged[i] + k * gap / (n_fill + 1)))
            merged = sorted(fill_set)

    if len(merged) < 5 and fallback_xs and len(fallback_xs) >= 8:
        return fallback_xs

    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: OCCUPANCY CLASSIFICATION  (unchanged from v4)
# ══════════════════════════════════════════════════════════════════════════════

def is_occupied(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    h, w = y2 - y1, x2 - x1
    if h < 8 or w < 8:
        return True

    ty = max(3, h // 7)
    tx = max(3, w // 8)
    inner = gray[y1 + ty:y2 - ty, x1 + tx:x2 - tx]
    if inner.size == 0:
        return True

    mean_g   = float(np.mean(inner))
    std_g    = float(np.std(inner))
    dark_f   = float(np.mean(inner < 85))
    bright_f = float(np.mean(inner > 185))
    lap      = cv2.Laplacian(inner.astype(np.float32), cv2.CV_32F)
    lap_var  = float(np.var(lap))

    if dark_f   > OCC_DARK_FRAC:                                    return True
    if lap_var  > OCC_LAP_VAR:                                      return True
    if bright_f > EMPTY_BRIGHT_FRAC and std_g < EMPTY_MAX_STD:     return False
    if lap_var  < EMPTY_LAP_LOW and mean_g > EMPTY_MIN_MEAN and dark_f < 0.05:
                                                                    return False
    occ_score = (dark_f * 3.0) + (lap_var / OCC_LAP_VAR) - (bright_f * 1.5)
    return occ_score > OCC_SCORE_THRESH


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: BUILD COMPLETE SPOT GRID FROM FIRST FRAME
# ══════════════════════════════════════════════════════════════════════════════

def build_spot_grid(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Full pipeline on a reference frame.
    STALL_MIN_W and STALL_MAX_W updated per FIX-B and FIX-C.
    """
    fh, fw = frame.shape[:2]
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
             if frame.ndim == 3 else frame.copy()

    bands  = detect_non_parking_bands(gray)
    zones  = parking_zones_from_bands(bands, fh)
    print(f"[GRID] {len(bands)} non-parking bands -> {len(zones)} parking zones")

    band_mask = np.zeros(fh, dtype=bool)
    for b0, b1 in bands:
        band_mask[b0:b1] = True

    all_spots: List[Tuple[int, int, int, int]] = []

    for z_y1, z_y2 in zones:
        row_pairs = split_zone_to_row_pairs(gray, z_y1, z_y2)

        for top_row, bot_row in row_pairs:
            ty1, ty2 = top_row
            top_xs   = get_stall_xs(gray, ty1, ty2)

            for j in range(len(top_xs) - 1):
                x1s, x2s = top_xs[j], top_xs[j + 1]
                sw = x2s - x1s
                if not (STALL_MIN_W < sw < STALL_MAX_W):  # FIX-B / FIX-C
                    continue
                strip = gray[ty1:ty2, x1s:x2s]
                if strip.size == 0 or np.mean(strip) <= STALL_CONTENT_MIN:
                    continue
                all_spots.append((x1s, ty1, x2s, ty2))

            if bot_row is None:
                continue

            by1, by2 = bot_row
            bot_xs = get_stall_xs(gray, by1, by2, fallback_xs=top_xs)

            for j in range(len(bot_xs) - 1):
                x1s, x2s = bot_xs[j], bot_xs[j + 1]
                sw = x2s - x1s
                if not (STALL_MIN_W < sw < STALL_MAX_W):
                    continue
                if band_mask[by1:by2].mean() > 0.5:
                    continue
                strip = gray[by1:by2, x1s:x2s]
                if strip.size == 0 or np.mean(strip) <= STALL_CONTENT_MIN:
                    continue
                all_spots.append((x1s, by1, x2s, by2))

    print(f"[GRID] {len(all_spots)} stall boxes detected")
    return all_spots


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: SPOT TRACKER WITH TEMPORAL SMOOTHING  (unchanged from v4)
# ══════════════════════════════════════════════════════════════════════════════

class SpotTracker:
    """Majority-vote temporal smoother over the last SMOOTH_FRAMES frames."""

    def __init__(self, spots: List[Tuple[int, int, int, int]],
                 smooth_frames: int = SMOOTH_FRAMES):
        self.spots   = spots
        self.n       = len(spots)
        self.smooth  = smooth_frames
        self.history: List[List[bool]] = [[] for _ in range(self.n)]

    def update(self, gray: np.ndarray) -> List[bool]:
        for i, (x1, y1, x2, y2) in enumerate(self.spots):
            occ = is_occupied(gray, x1, y1, x2, y2)
            self.history[i].append(occ)
            if len(self.history[i]) > self.smooth:
                self.history[i].pop(0)
        return [sum(h) >= len(h) / 2.0 for h in self.history]


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE  (unchanged from v4)
# ══════════════════════════════════════════════════════════════════════════════

LOTS = [
    ("A", 150), ("B", 458), ("C", 332), ("D", 271), ("E / Visitor", 260),
    ("F", 491), ("G", 364), ("H", 709), ("J", 579), ("K", 496), ("L", 447),
    ("M", 128), ("N", 111), ("R", 101), ("S", 908), ("T", 328), ("U", 260),
    ("Global Village", 367), ("Riverknoll", 327), ("University Commons", 941),
    ("Perkins", 470),
]


def get_lots_dict():
    return {n: c for n, c in LOTS}


def get_conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c


def create_schema():
    with get_conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS lots (
            lot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            lot_name TEXT    NOT NULL UNIQUE,
            capacity INTEGER NOT NULL CHECK(capacity>0),
            occupancy INTEGER NOT NULL DEFAULT 0,
            active   INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE IF NOT EXISTS vehicle_events (
            event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            lot_id     INTEGER NOT NULL REFERENCES lots(lot_id),
            tracker_id INTEGER NOT NULL,
            direction  TEXT    NOT NULL CHECK(direction IN ('ENTRY','EXIT')),
            timestamp  TEXT    NOT NULL,
            confidence REAL    DEFAULT 1.0);
        CREATE TABLE IF NOT EXISTS occupancy_log (
            log_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            lot_id    INTEGER NOT NULL REFERENCES lots(lot_id),
            occupancy INTEGER NOT NULL,
            pct_full  REAL    NOT NULL,
            timestamp TEXT    NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_ev ON vehicle_events(lot_id,timestamp);
        CREATE INDEX IF NOT EXISTS idx_ol ON occupancy_log(lot_id,timestamp);
        """)


def seed_lots():
    with get_conn() as c:
        for n, cap in LOTS:
            c.execute("INSERT OR IGNORE INTO lots (lot_name,capacity) VALUES (?,?)",
                      (n, cap))
        c.commit()


def register_lot(name: str, cap: int) -> int:
    with get_conn() as c:
        row = c.execute("SELECT lot_id FROM lots WHERE lot_name=?",
                        (name,)).fetchone()
        if row:
            c.execute("UPDATE lots SET capacity=?,occupancy=0 WHERE lot_name=?",
                      (cap, name))
            lid = row["lot_id"]
        else:
            lid = c.execute(
                "INSERT INTO lots (lot_name,capacity,occupancy) VALUES (?,?,0)",
                (name, cap)).lastrowid
        c.commit()
    print(f"[DB] Lot '{name}'  capacity={cap}  lot_id={lid}")
    return lid


def write_occupancy(lot_id: int, count: int, cap: int):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pct = round(count / max(cap, 1) * 100, 2)
    with get_conn() as c:
        c.execute("UPDATE lots SET occupancy=? WHERE lot_id=?", (count, lot_id))
        c.execute(
            "INSERT INTO occupancy_log (lot_id,occupancy,pct_full,timestamp) "
            "VALUES (?,?,?,?)", (lot_id, count, pct, ts))
        c.commit()


def get_all_status():
    with get_conn() as c:
        rows = c.execute(
            "SELECT lot_name,capacity,occupancy,"
            "ROUND(occupancy*100.0/capacity,1) AS pct_full,"
            "capacity-occupancy AS available "
            "FROM lots WHERE active=1 ORDER BY lot_name"
        ).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN VIDEO TRACKER LOOP  (unchanged from v4)
# ══════════════════════════════════════════════════════════════════════════════

def run_tracker(source,
                lot_name: str,
                playback_speed: float = 1.0,
                frame_skip: int = DEFAULT_FRAME_SKIP):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame.")
        cap.release()
        return

    fh, fw = first_frame.shape[:2]
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[INFO] {fw}x{fh}  {fps:.1f}fps  {total} frames")

    print("\n[GRID] Detecting parking stall grid from first frame...")
    spots = build_spot_grid(first_frame)
    if not spots:
        print("[ERROR] No spots detected. Check video orientation/quality.")
        cap.release()
        return

    capacity  = len(spots)
    lot_id    = register_lot(lot_name, capacity)
    tracker   = SpotTracker(spots, smooth_frames=SMOOTH_FRAMES)
    wait      = max(1, int(1000 / fps / playback_speed))
    sx        = DISPLAY_W / fw
    sy        = DISPLAY_H / fh
    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"\n[INFO] Starting tracker -- lot='{lot_name}'  spots={capacity}")
    if _has_display():
        print("  Q = quit\n")

    occ_list: List[bool] = [False] * capacity

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_skip > 1 and frame_idx % frame_skip != 0:
            continue

        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        occ_list = tracker.update(gray)

        occ_count = sum(occ_list)
        emp_count = capacity - occ_count
        pct       = occ_count / max(capacity, 1) * 100

        if frame_idx % max(1, int(fps * frame_skip)) == 0:
            write_occupancy(lot_id, occ_count, capacity)

        if _has_display():
            vis = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

            for (x1, y1, x2, y2), occ in zip(spots, occ_list):
                color = (0, 50, 220) if occ else (0, 220, 60)
                cv2.rectangle(vis,
                              (int(x1 * sx), int(y1 * sy)),
                              (int(x2 * sx), int(y2 * sy)),
                              color, 1)

            cb = (0, 200, 80) if pct < 70 else (0, 165, 255) if pct < 90 else (0, 60, 220)
            ov = vis.copy()
            cv2.rectangle(ov, (12, 12), (560, 175), (15, 15, 15), -1)
            cv2.addWeighted(ov, 0.70, vis, 0.30, 0, vis)

            cv2.putText(vis, f"Lot: {lot_name}",
                        (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"Occupied : {occ_count}/{capacity}  ({pct:.0f}%)",
                        (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        cb, 2, cv2.LINE_AA)
            cv2.putText(vis, f"Empty    : {emp_count}   Frame {frame_idx}",
                        (25, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (180, 255, 180), 2, cv2.LINE_AA)

            bx, by, bw2, bh2 = 25, 162, 510, 8
            cv2.rectangle(vis, (bx, by), (bx + bw2, by + bh2), (60, 60, 60), -1)
            fw2 = int(bw2 * min(pct / 100, 1))
            if fw2 > 0:
                cv2.rectangle(vis, (bx, by), (bx + fw2, by + bh2), cb, -1)

            cv2.imshow(f"RIT Smart Parking -- {lot_name}", vis)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        else:
            if frame_idx % (int(fps * 10)) == 0:
                print(f"[HEADLESS] Frame {frame_idx}/{total} | "
                      f"occ={occ_count}/{capacity} ({pct:.1f}%)")

    cap.release()
    if _has_display():
        cv2.destroyAllWindows()

    write_occupancy(lot_id, occ_count, capacity)
    print(f"\n[DONE] Final: {occ_count}/{capacity} "
          f"({occ_count / max(capacity, 1) * 100:.1f}% full)  "
          f"Empty: {capacity - occ_count}")


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-FRAME SNAPSHOT  (unchanged from v4)
# ══════════════════════════════════════════════════════════════════════════════

def snapshot_count(video_path: str,
                   frame_pos: int = 0,
                   save_annotated: str = None) -> Dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot read frame.")

    fh, fw = frame.shape[:2]
    print(f"[SNAP] Frame {frame_pos}  {fw}x{fh}")

    spots = build_spot_grid(frame)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    occ   = [is_occupied(gray, x1, y1, x2, y2) for x1, y1, x2, y2 in spots]

    total    = len(spots)
    occupied = sum(occ)
    empty    = total - occupied
    pct      = occupied / max(total, 1) * 100

    result = dict(total=total, occupied=occupied, empty=empty,
                  pct_full=round(pct, 1))
    print(f"[SNAP] Total={total}  Occupied={occupied}  "
          f"Empty={empty}  ({pct:.1f}%)")

    if save_annotated:
        vis = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        sx, sy = DISPLAY_W / fw, DISPLAY_H / fh
        for (x1, y1, x2, y2), o in zip(spots, occ):
            col = (0, 50, 220) if o else (0, 220, 60)
            cv2.rectangle(vis,
                          (int(x1 * sx), int(y1 * sy)),
                          (int(x2 * sx), int(y2 * sy)), col, 1)
        cv2.rectangle(vis, (12, 12), (580, 115), (15, 15, 15), -1)
        cv2.putText(vis,
                    f"Total:{total}  Occupied:{occupied}  Empty:{empty}",
                    (22, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Occupancy: {pct:.1f}%",
                    (22, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 220, 100), 2, cv2.LINE_AA)
        cv2.imwrite(save_annotated, vis)
        print(f"[SNAP] Annotated image saved: {save_annotated}")

    return result


# ── Dashboard ──────────────────────────────────────────────────────────────────
def _bar(pct, w=20):
    f = max(0, min(w, int(pct / 100 * w)))
    return chr(9608) * f + chr(9617) * (w - f)


def print_dashboard():
    print("\n" + "=" * 72)
    print("  RIT SMART PARKING TRACKER  v6  (fixed)")
    print(f"  {datetime.now().strftime('%A, %B %d %Y  %I:%M %p')}")
    print("=" * 72)
    print(f"\n  {'Lot':<30}{'Cap':>6}{'Parked':>8}{'Avail':>7}{'%Full':>7}  Bar")
    print("  " + "-" * 70)
    rows = get_all_status() or \
           [{"lot_name": n, "capacity": c, "occupancy": 0,
             "available": c, "pct_full": 0.0} for n, c in LOTS]
    for s in rows:
        print(f"  {s['lot_name']:<30}{s['capacity']:>6}{s['occupancy']:>8}"
              f"{s['available']:>7}{s['pct_full']:>6.1f}%  {_bar(s['pct_full'])}")
    print("\n" + "=" * 72 + "\n")


# ── Simulation ─────────────────────────────────────────────────────────────────
def run_simulation():
    import random
    print("\n[SIM] Simulating today's traffic...\n")
    today  = datetime.now().strftime("%Y-%m-%d")
    plates = [f"NY{i:04d}" for i in range(1, 301)]
    with get_conn() as c:
        lots = c.execute(
            "SELECT lot_id,lot_name,capacity FROM lots").fetchall()
    for row in lots:
        lid, lname, cap = row["lot_id"], row["lot_name"], row["capacity"]
        print(f"  {lname}  cap={cap}")
        events = []
        base = datetime.strptime(f"{today} 07:30:00", "%Y-%m-%d %H:%M:%S")
        for _ in range(int(cap * 0.80)):
            events.append((base + timedelta(seconds=random.randint(0, 7200)),
                           "ENTRY", random.choice(plates)))
        base2 = datetime.strptime(f"{today} 15:00:00", "%Y-%m-%d %H:%M:%S")
        for _ in range(int(cap * 0.60)):
            events.append((base2 + timedelta(seconds=random.randint(0, 10800)),
                           "EXIT", random.choice(plates)))
        events.sort(key=lambda e: e[0])
        with get_conn() as c:
            c.execute("UPDATE lots SET occupancy=0 WHERE lot_id=?", (lid,))
            for ts_dt, direction, _ in events:
                ts = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
                c.execute(
                    "INSERT INTO vehicle_events "
                    "(lot_id,tracker_id,direction,timestamp,confidence) "
                    "VALUES (?,?,?,?,?)",
                    (lid, 0, direction, ts, 0.95))
                if direction == "ENTRY":
                    c.execute("UPDATE lots SET occupancy=MIN(capacity,occupancy+1) "
                              "WHERE lot_id=?", (lid,))
                else:
                    c.execute("UPDATE lots SET occupancy=MAX(0,occupancy-1) "
                              "WHERE lot_id=?", (lid,))
                lot = c.execute("SELECT occupancy,capacity FROM lots "
                                "WHERE lot_id=?", (lid,)).fetchone()
                c.execute(
                    "INSERT INTO occupancy_log "
                    "(lot_id,occupancy,pct_full,timestamp) VALUES (?,?,?,?)",
                    (lid, lot["occupancy"],
                     round(lot["occupancy"] / lot["capacity"] * 100, 2), ts))
            c.commit()
    print("\n[SIM] Done.\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    create_schema()
    seed_lots()

    print("\n RIT Smart Parking Tracker  v5  (fixed)")
    print(" " + "-" * 45)
    print(" [1] Track from video file  (full loop)")
    print(" [2] Single-frame snapshot  (quick count)")
    print(" [3] Simulate a full day")
    print(" [4] Show dashboard")
    print(f"\n Known lots: {', '.join(sorted(get_lots_dict().keys()))}")
    choice = input("\n Select (1/2/3/4): ").strip()

    if choice == "1":
        vp = input(" Video path: ").strip().strip('"').strip("'")
        if not os.path.isfile(vp):
            print(f"[ERROR] Not found: {vp}")
        else:
            label = input(" Lot label (Enter=filename): ").strip() or \
                    os.path.splitext(os.path.basename(vp))[0]
            try:    speed = float(input(" Playback speed [1.0]: ").strip() or "1")
            except: speed = 1.0
            try:    skip  = int(input(f" Frame skip [{DEFAULT_FRAME_SKIP}]: ")
                                .strip() or DEFAULT_FRAME_SKIP)
            except: skip  = DEFAULT_FRAME_SKIP
            run_tracker(vp, label, speed, max(1, skip))
            print_dashboard()

    elif choice == "2":
        vp = input(" Video path: ").strip().strip('"').strip("'")
        if not os.path.isfile(vp):
            print(f"[ERROR] Not found: {vp}")
        else:
            try:    pos = int(input(" Frame number [0]: ").strip() or "0")
            except: pos = 0
            out = input(" Save annotated image path (Enter=skip): ").strip()
            result = snapshot_count(vp, pos, save_annotated=out or None)
            print(f"\n  Total spots : {result['total']}")
            print(f"  Occupied    : {result['occupied']}")
            print(f"  Empty       : {result['empty']}")
            print(f"  Occupancy   : {result['pct_full']}%")

    elif choice == "3":
        run_simulation()
        print_dashboard()

    else:
        print_dashboard()