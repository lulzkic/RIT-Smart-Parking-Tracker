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




# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH            = "rit_parking.db"
VIDEO_EXTENSIONS   = (".mp4",".avi",".mov",".mkv",".wmv",".m4v",".flv",".webm")
DEFAULT_FRAME_SKIP = 2
DISPLAY_W          = 1920   # visualization output width
DISPLAY_H          = 1080


# Occupancy classifier thresholds (tuned for this video)
OCC_DARK_FRAC      = 0.20   # >20% dark pixels → car roof/windows → occupied
OCC_LAP_VAR        = 800    # high laplacian variance → car texture → occupied
EMPTY_BRIGHT_FRAC  = 0.55   # >55% bright + uniform → clean asphalt → empty
EMPTY_MAX_STD      = 35     # max std for "uniform" classification
EMPTY_LAP_LOW      = 200    # low texture threshold for empty asphalt
EMPTY_MIN_MEAN     = 130    # min mean brightness for empty classification
OCC_SCORE_THRESH   = 0.6    # combined score threshold


# Stall size constraints (pixels at native 3840px width)
STALL_MIN_W        = 15
STALL_MAX_W        = 110
ROAD_BRIGHTNESS    = 108    # below this = driving lane
ROAD_MIN_HEIGHT    = 25     # min height to call something a road band
STALL_CONTENT_MIN  = 108    # stall mean brightness must exceed this (avoids road cells)


# Temporal smoothing: stall occupancy averaged over this many frames
SMOOTH_FRAMES      = 5




# ── Display guard ─────────────────────────────────────────────────────────────
def _has_display() -> bool:
   import sys
   if sys.platform.startswith("linux"):
       return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
   return True




# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: DETECT DRIVING ROADS (horizontal dark bands)
# ══════════════════════════════════════════════════════════════════════════════


def detect_road_bands(frame: np.ndarray) -> List[Tuple[int,int]]:
   """
   Find all horizontal driving-lane bands by looking at the brightness
   profile averaged over the middle 80% of the frame width.
   Returns list of (y_start, y_end) road band y-ranges.
   """
   fh, fw = frame.shape[:2]
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
          if frame.ndim == 3 else frame


   x0 = fw // 10
   x1 = fw - x0
   profile = np.mean(gray[:, x0:x1], axis=1).astype(float)
   smooth  = np.convolve(profile, np.ones(5)/5, mode='same')


   roads: List[Tuple[int,int]] = []
   in_road = False
   ry0 = 0
   for y in range(fh):
       if smooth[y] < ROAD_BRIGHTNESS:
           if not in_road:
               in_road = True
               ry0 = y
       else:
           if in_road:
               in_road = False
               if y - ry0 >= ROAD_MIN_HEIGHT:
                   roads.append((ry0, y))
   if in_road and fh - ry0 >= ROAD_MIN_HEIGHT:
       roads.append((ry0, fh))


   return roads




def parking_zones_from_roads(road_bands: List[Tuple[int,int]],
                             frame_h: int) -> List[Tuple[int,int]]:
   """
   Return y-ranges of parking areas (between roads).
   Zones taller than ~200px are split recursively at any internal road-like dip.
   """
   bounds = sorted(set([0] + [y for r in road_bands for y in r] + [frame_h]))
   zones  = []
   for i in range(len(bounds)-1):
       y1, y2 = bounds[i], bounds[i+1]
       is_road = any(y1 == r[0] for r in road_bands)
       if not is_road and y2 - y1 > 40:
           zones.append((y1, y2))
   return zones




# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: SPLIT PARKING ZONES INTO INDIVIDUAL CAR ROWS
# ══════════════════════════════════════════════════════════════════════════════


def split_zone_to_row_pairs(gray: np.ndarray,
                            y_start: int, y_end: int,
                            scan_x0: int = 150,
                            scan_x1: int = None
                            ) -> List[Tuple[Tuple[int,int], Optional[Tuple[int,int]]]]:
   """
   Split a parking zone into (top_row, bottom_row) pairs.
   Top row = the sub-band with the most visible stall-divider lines.
   Bottom row = the other half (may have weaker lines; inherits X positions).


   Returns list of ((y1_top, y2_top), (y1_bot, y2_bot)) or
                   ((y1_top, y2_top), None) for standalone rows.
   """
   fh = gray.shape[0]
   fw = gray.shape[1]
   if scan_x1 is None:
       scan_x1 = fw - 150


   zone_h = y_end - y_start
   if zone_h < 40:
       return []


   # Single row: zone_h < 120 → just return it as standalone
   if zone_h < 120:
       mid = (y_start + y_end) // 2
       return [((y_start, mid), (mid, y_end))]


   # Check for internal road bands within this zone
   band     = gray[y_start:y_end, scan_x0:scan_x1]
   bh       = band.shape[0]
   hb       = np.mean(band, axis=1).astype(float)
   hbs      = np.convolve(hb, np.ones(5)/5, mode='same')


   # Find internal dark bands
   internal_roads = []
   in_dark = False
   ds = 0
   for i in range(bh):
       if hbs[i] < ROAD_BRIGHTNESS + 5:
           if not in_dark:
               in_dark = True
               ds = i
       else:
           if in_dark:
               in_dark = False
               if i - ds >= 20:
                   internal_roads.append((y_start+ds, y_start+i))


   if internal_roads:
       # Zone has internal roads → recurse on each sub-zone
       result   = []
       all_bnds = sorted(set(
           [y_start] + [y for r in internal_roads for y in r] + [y_end]
       ))
       for i in range(len(all_bnds)-1):
           r1, r2 = all_bnds[i], all_bnds[i+1]
           is_road = any(abs(r1-r[0]) < 5 for r in internal_roads)
           if not is_road and r2 - r1 >= 40:
               result.extend(split_zone_to_row_pairs(gray, r1, r2, scan_x0, scan_x1))
       return result


   # No internal road: split at center (back-to-back car pair)
   mid = (y_start + y_end) // 2
   return [((y_start, mid), (mid, y_end))]




# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: DETECT STALL DIVIDER X POSITIONS
# ══════════════════════════════════════════════════════════════════════════════


def get_stall_xs(gray: np.ndarray,
                y1: int, y2: int,
                x_start: int = 100, x_end: int = None,
                fallback_xs: Optional[List[int]] = None
                ) -> List[int]:
   """
   Detect stall divider (white/yellow vertical line) x-positions within a row.
   Tries thresholds from strict→loose until ≥20 peaks are found.
   Falls back to fallback_xs (from an adjacent row) if lines are too faint.
   """
   if x_end is None:
       x_end = gray.shape[1] - 100


   trim_y = max(3, (y2-y1)//6)
   band   = gray[y1+trim_y:y2-trim_y, x_start:x_end]
   if band.shape[0] < 5:
       return fallback_xs or []


   h_row = band.shape[0]
   best_peaks: List[int] = []


   for thresh in [178, 168, 158, 148, 138]:
       _, white = cv2.threshold(band, thresh, 255, cv2.THRESH_BINARY)
       kv   = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (1, max(3, h_row//4)))
       vert = cv2.morphologyEx(white, cv2.MORPH_OPEN, kv)
       col_d  = np.sum(vert > 0, axis=0).astype(float)
       col_sm = np.convolve(col_d, np.ones(3)/3, mode='same')
       h_thr  = max(1, h_row * 0.10)
       peaks  = _find_peaks(col_sm, height=h_thr, distance=14, prominence=2)
       if len(peaks) >= 20:
           best_peaks = [int(p) + x_start for p in peaks]
           break


   if len(best_peaks) < 10 and fallback_xs and len(fallback_xs) >= 20:
       return fallback_xs


   return best_peaks




# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: OCCUPANCY CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════


def is_occupied(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
   """
   Multi-signal occupancy classifier.
   Returns True = OCCUPIED, False = EMPTY.


   Tuned on real pixel data from 4K overhead parking footage:
     Signal 1 (hard):  dark_frac > 0.20  → car roof/windows → OCCUPIED
     Signal 2 (hard):  lap_var   > 800   → rich car texture  → OCCUPIED
     Signal 3 (hard):  bright + uniform  → clean asphalt/lines → EMPTY
     Signal 4 (hard):  low texture + mid-gray              → EMPTY
     Signal 5 (soft):  combined weighted score
   """
   h, w = y2-y1, x2-x1
   if h < 8 or w < 8:
       return True


   ty = max(3, h//7)
   tx = max(3, w//8)
   inner = gray[y1+ty:y2-ty, x1+tx:x2-tx]
   if inner.size == 0:
       return True


   mean_g   = float(np.mean(inner))
   std_g    = float(np.std(inner))
   dark_f   = float(np.mean(inner < 85))
   bright_f = float(np.mean(inner > 185))
   lap      = cv2.Laplacian(inner.astype(np.float32), cv2.CV_32F)
   lap_var  = float(np.var(lap))


   # Hard rules
   if dark_f   > OCC_DARK_FRAC:                                    return True
   if lap_var  > OCC_LAP_VAR:                                      return True
   if bright_f > EMPTY_BRIGHT_FRAC and std_g < EMPTY_MAX_STD:     return False
   if lap_var  < EMPTY_LAP_LOW and mean_g > EMPTY_MIN_MEAN and dark_f < 0.05:
                                                                   return False


   # Soft combined score
   occ_score = (dark_f * 3.0) + (lap_var / OCC_LAP_VAR) - (bright_f * 1.5)
   return occ_score > OCC_SCORE_THRESH




# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: BUILD COMPLETE SPOT GRID FROM FIRST FRAME
# ══════════════════════════════════════════════════════════════════════════════


def build_spot_grid(frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
   """
   Full pipeline on a reference frame:
     1. Detect driving roads
     2. Split frame into parking zones
     3. Split each zone into row pairs
     4. Detect stall X positions per row pair
     5. Filter out stalls that overlap roads or look like road content


   Returns list of (x1, y1, x2, y2) stall rectangles.
   """
   fh, fw = frame.shape[:2]
   gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
            if frame.ndim == 3 else frame.copy()


   # 1. Roads
   roads  = detect_road_bands(gray)
   zones  = parking_zones_from_roads(roads, fh)
   print(f"[GRID] {len(roads)} road bands → {len(zones)} parking zones")


   # Build a road mask for fast stall filtering
   road_mask = np.zeros(fh, dtype=bool)
   for ry1, ry2 in roads:
       road_mask[ry1:ry2] = True


   all_spots: List[Tuple[int,int,int,int]] = []


   for z_y1, z_y2 in zones:
       row_pairs = split_zone_to_row_pairs(gray, z_y1, z_y2)


       for top_row, bot_row in row_pairs:
           ty1, ty2 = top_row
           top_xs   = get_stall_xs(gray, ty1, ty2)


           # Top row stalls
           for j in range(len(top_xs)-1):
               x1s, x2s = top_xs[j], top_xs[j+1]
               sw = x2s - x1s
               if not (STALL_MIN_W < sw < STALL_MAX_W):
                   continue
               strip = gray[ty1:ty2, x1s:x2s]
               if strip.size == 0 or np.mean(strip) <= STALL_CONTENT_MIN:
                   continue
               all_spots.append((x1s, ty1, x2s, ty2))


           if bot_row is None:
               continue


           by1, by2 = bot_row
           bot_xs = get_stall_xs(gray, by1, by2, fallback_xs=top_xs)


           for j in range(len(bot_xs)-1):
               x1s, x2s = bot_xs[j], bot_xs[j+1]
               sw = x2s - x1s
               if not (STALL_MIN_W < sw < STALL_MAX_W):
                   continue
               # Check stall y-range doesn't fall in a road band
               if road_mask[by1:by2].mean() > 0.5:
                   continue
               strip = gray[by1:by2, x1s:x2s]
               if strip.size == 0 or np.mean(strip) <= STALL_CONTENT_MIN:
                   continue
               all_spots.append((x1s, by1, x2s, by2))


   print(f"[GRID] {len(all_spots)} stall boxes detected")
   return all_spots




# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: CLASSIFY ALL SPOTS ON A FRAME (with temporal smoothing)
# ══════════════════════════════════════════════════════════════════════════════


class SpotTracker:
   """
   Tracks occupancy state per spot with temporal smoothing.
   Each spot keeps a short history; majority vote decides current state.
   """
   def __init__(self, spots: List[Tuple[int,int,int,int]],
                smooth_frames: int = SMOOTH_FRAMES):
       self.spots   = spots
       self.n       = len(spots)
       self.smooth  = smooth_frames
       # history[i] = list of recent bool values (True=occupied)
       self.history: List[List[bool]] = [[] for _ in range(self.n)]


   def update(self, gray: np.ndarray) -> List[bool]:
       """Classify all spots and return smoothed occupancy list."""
       for i, (x1,y1,x2,y2) in enumerate(self.spots):
           occ = is_occupied(gray, x1, y1, x2, y2)
           self.history[i].append(occ)
           if len(self.history[i]) > self.smooth:
               self.history[i].pop(0)


       # Majority vote
       result = []
       for i in range(self.n):
           h = self.history[i]
           result.append(sum(h) >= len(h) / 2.0)
       return result




# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════


LOTS = [
   ("A",150),("B",458),("C",332),("D",271),("E / Visitor",260),
   ("F",491),("G",364),("H",709),("J",579),("K",496),("L",447),
   ("M",128),("N",111),("R",101),("S",908),("T",328),("U",260),
   ("Global Village",367),("Riverknoll",327),("University Commons",941),
   ("Perkins",470),
]


def get_lots_dict(): return {n: c for n, c in LOTS}


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
       c.execute("UPDATE lots SET occupancy=? WHERE lot_id=?",
                 (count, lot_id))
       c.execute(
           "INSERT INTO occupancy_log (lot_id,occupancy,pct_full,timestamp) "
           "VALUES (?,?,?,?)",
           (lot_id, count, pct, ts))
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
#  MAIN VIDEO TRACKER LOOP
# ══════════════════════════════════════════════════════════════════════════════


def run_tracker(source,
               lot_name: str,
               playback_speed: float = 1.0,
               frame_skip: int = DEFAULT_FRAME_SKIP):
   """
   Process a video file or live camera feed.
   Builds the stall grid on the first frame, then tracks occupancy per frame.
   """
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
   print(f"[INFO] {fw}×{fh}  {fps:.1f}fps  {total} frames")


   # ── Build stall grid ─────────────────────────────────────────────────────
   print("\n[GRID] Detecting parking stall grid from first frame…")
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


   # Reset to start
   cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


   print(f"\n[INFO] Starting tracker — lot='{lot_name}'  spots={capacity}")
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


       # Write to DB every ~1 second of video
       if frame_idx % max(1, int(fps * frame_skip)) == 0:
           write_occupancy(lot_id, occ_count, capacity)


       # ── Display ──────────────────────────────────────────────────────────
       if _has_display():
           vis = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))


           # Draw stall boxes
           for (x1,y1,x2,y2), occ in zip(spots, occ_list):
               color = (0, 50, 220) if occ else (0, 220, 60)
               cv2.rectangle(vis,
                             (int(x1*sx), int(y1*sy)),
                             (int(x2*sx), int(y2*sy)),
                             color, 1)


           # HUD background
           cb = (0,200,80) if pct < 70 else (0,165,255) if pct < 90 else (0,60,220)
           ov = vis.copy()
           cv2.rectangle(ov, (12,12), (560,175), (15,15,15), -1)
           cv2.addWeighted(ov, 0.70, vis, 0.30, 0, vis)


           cv2.putText(vis, f"Lot: {lot_name}",
                       (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                       (255,255,255), 2, cv2.LINE_AA)
           cv2.putText(vis, f"Occupied : {occ_count}/{capacity}  ({pct:.0f}%)",
                       (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                       cb, 2, cv2.LINE_AA)
           cv2.putText(vis, f"Empty    : {emp_count}   Frame {frame_idx}",
                       (25, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                       (180,255,180), 2, cv2.LINE_AA)


           # Occupancy bar
           bx,by,bw2,bh2 = 25, 162, 510, 8
           cv2.rectangle(vis, (bx,by), (bx+bw2, by+bh2), (60,60,60), -1)
           fw2 = int(bw2 * min(pct/100, 1))
           if fw2 > 0:
               cv2.rectangle(vis, (bx,by), (bx+fw2, by+bh2), cb, -1)


           cv2.imshow(f"RIT Smart Parking — {lot_name}", vis)
           if cv2.waitKey(wait) & 0xFF == ord('q'):
               break
       else:
           if frame_idx % (int(fps * 10)) == 0:
               print(f"[HEADLESS] Frame {frame_idx}/{total} | "
                     f"occ={occ_count}/{capacity} ({pct:.1f}%)")


   cap.release()
   if _has_display():
       cv2.destroyAllWindows()


   # Final DB write
   write_occupancy(lot_id, occ_count, capacity)
   print(f"\n[DONE] Final: {occ_count}/{capacity} "
         f"({occ_count/max(capacity,1)*100:.1f}% full)  "
         f"Empty: {capacity-occ_count}")




# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-FRAME SNAPSHOT (useful for testing without running full video)
# ══════════════════════════════════════════════════════════════════════════════


def snapshot_count(video_path: str,
                  frame_pos: int = 0,
                  save_annotated: str = None) -> Dict:
   """
   Count occupied/empty on a single frame.
   Returns dict with total, occupied, empty, pct_full.
   If save_annotated is a path, writes a JPEG with the grid overlay.
   """
   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
       raise RuntimeError(f"Cannot open: {video_path}")
   cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
   ret, frame = cap.read()
   cap.release()
   if not ret:
       raise RuntimeError("Cannot read frame.")


   fh, fw = frame.shape[:2]
   print(f"[SNAP] Frame {frame_pos}  {fw}×{fh}")


   spots = build_spot_grid(frame)
   gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   occ   = [is_occupied(gray, x1,y1,x2,y2) for x1,y1,x2,y2 in spots]


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
       sx, sy = DISPLAY_W/fw, DISPLAY_H/fh
       for (x1,y1,x2,y2), o in zip(spots, occ):
           col = (0,50,220) if o else (0,220,60)
           cv2.rectangle(vis,
                         (int(x1*sx),int(y1*sy)),
                         (int(x2*sx),int(y2*sy)), col, 1)
       # HUD
       cv2.rectangle(vis,(12,12),(580,115),(15,15,15),-1)
       cv2.putText(vis,
                   f"Total:{total}  Occupied:{occupied}  Empty:{empty}",
                   (22,50), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                   (255,255,255), 2, cv2.LINE_AA)
       cv2.putText(vis, f"Occupancy: {pct:.1f}%",
                   (22,95), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                   (0,220,100), 2, cv2.LINE_AA)
       cv2.imwrite(save_annotated, vis)
       print(f"[SNAP] Annotated image saved: {save_annotated}")


   return result




# ── Dashboard ──────────────────────────────────────────────────────────────────
def _bar(pct, w=20):
   f = max(0, min(w, int(pct/100*w)))
   return "█"*f + "░"*(w-f)


def print_dashboard():
   print("\n" + "="*72)
   print("  RIT SMART PARKING TRACKER  v4")
   print(f"  {datetime.now().strftime('%A, %B %d %Y  %I:%M %p')}")
   print("="*72)
   print(f"\n  {'Lot':<30}{'Cap':>6}{'Parked':>8}{'Avail':>7}{'%Full':>7}  Bar")
   print("  "+"-"*70)
   rows = get_all_status() or \
          [{"lot_name":n,"capacity":c,"occupancy":0,
            "available":c,"pct_full":0.0} for n,c in LOTS]
   for s in rows:
       print(f"  {s['lot_name']:<30}{s['capacity']:>6}{s['occupancy']:>8}"
             f"{s['available']:>7}{s['pct_full']:>6.1f}%  {_bar(s['pct_full'])}")
   print("\n" + "="*72 + "\n")




# ── Simulation ─────────────────────────────────────────────────────────────────
def run_simulation():
   import random
   print("\n[SIM] Simulating today's traffic…\n")
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
       for _ in range(int(cap*0.80)):
           events.append((base+timedelta(seconds=random.randint(0,7200)),
                          "ENTRY", random.choice(plates)))
       base2 = datetime.strptime(f"{today} 15:00:00", "%Y-%m-%d %H:%M:%S")
       for _ in range(int(cap*0.60)):
           events.append((base2+timedelta(seconds=random.randint(0,10800)),
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
                    round(lot["occupancy"]/lot["capacity"]*100, 2), ts))
           c.commit()
   print("\n[SIM] Done.\n")




# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
   create_schema()
   seed_lots()


   print("\n RIT Smart Parking Tracker  v4")
   print(" ─────────────────────────────────────────────")
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
           try:   pos = int(input(" Frame number [0]: ").strip() or "0")
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

