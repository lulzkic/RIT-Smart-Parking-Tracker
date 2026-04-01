# RIT Smart Parking Tracker — camera/video vehicle counter + SQLite backend
# Dependencies: pip install opencv-python numpy


import cv2
import numpy as np
import sqlite3
import hashlib
import os
import re
from datetime import datetime, timedelta
from typing import Optional


# ── Config ───────────────────────────────────────────────────────────────────
DB_PATH          = "rit_parking.db"  # SQLite file (auto-created)
DEFAULT_LOT      = "S"               # pre-filled lot name shown in the menu
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".flv", ".webm")


# ── Scene profiles ────────────────────────────────────────────────────────────
# Each profile holds all CV parameters for a specific camera angle + lighting combo.
# The tracker selects one automatically (or asks the user) at startup.
#
# line_axis : "horizontal" = line spans width  (top-down, cars move vertically)
#             "vertical"   = line spans height (side/angled, cars move horizontally)
# entry_dir : which crossing direction counts as ENTRY
#             "top_to_bottom" | "bottom_to_top" | "right_to_left" | "left_to_right"


SCENE_PROFILES = {
   "overhead_day": {
       "label":           "Top-down · daytime",
       "min_area":        800,    # overhead cars appear as small compact blobs
       "tracker_dist":    80,     # drone altitude = more px per frame of travel
       "cooldown":        2,      # cars cross the line quickly from above
       "mog2_history":    300,    # static drone camera needs longer history
       "mog2_var":        40,     # bright flat asphalt → lower threshold is fine
       "mog2_shadows":    True,   # hard daytime shadows must be suppressed
       "morph_kernel":    3,      # small kernel keeps tight overhead shapes intact
       "morph_shape":     cv2.MORPH_ELLIPSE,
       "line_axis":       "horizontal",
       "entry_dir":       "top_to_bottom",
   },
   "overhead_night": {
       "label":           "Top-down · night / low light",
       "min_area":        600,    # headlights make cars appear as bright blobs
       "tracker_dist":    80,
       "cooldown":        2,
       "mog2_history":    200,    # scene changes faster under artificial lighting
       "mog2_var":        25,     # dark background → very sensitive threshold
       "mog2_shadows":    False,  # no hard shadows at night
       "morph_kernel":    3,
       "morph_shape":     cv2.MORPH_ELLIPSE,
       "line_axis":       "horizontal",
       "entry_dir":       "top_to_bottom",
   },
   "side_day": {
       "label":           "Side view · daytime",
       "min_area":        2500,   # full car silhouette from the side is large
       "tracker_dist":    60,
       "cooldown":        3,
       "mog2_history":    150,
       "mog2_var":        50,
       "mog2_shadows":    True,
       "morph_kernel":    5,
       "morph_shape":     cv2.MORPH_RECT,
       "line_axis":       "vertical",
       "entry_dir":       "right_to_left",
   },
   "side_night": {
       "label":           "Side view · night / low light",
       "min_area":        1500,
       "tracker_dist":    60,
       "cooldown":        3,
       "mog2_history":    100,
       "mog2_var":        30,
       "mog2_shadows":    False,
       "morph_kernel":    5,
       "morph_shape":     cv2.MORPH_RECT,
       "line_axis":       "vertical",
       "entry_dir":       "right_to_left",
   },
   "angled_day": {
       "label":           "Angled / security cam · daytime",
       "min_area":        1800,
       "tracker_dist":    70,
       "cooldown":        3,
       "mog2_history":    200,
       "mog2_var":        45,
       "mog2_shadows":    True,
       "morph_kernel":    4,
       "morph_shape":     cv2.MORPH_RECT,
       "line_axis":       "vertical",
       "entry_dir":       "right_to_left",
   },
   "angled_night": {
       "label":           "Angled / security cam · night",
       "min_area":        1200,
       "tracker_dist":    70,
       "cooldown":        3,
       "mog2_history":    120,
       "mog2_var":        28,
       "mog2_shadows":    False,
       "morph_kernel":    4,
       "morph_shape":     cv2.MORPH_RECT,
       "line_axis":       "vertical",
       "entry_dir":       "right_to_left",
   },
}




# ── Auto scene detection ──────────────────────────────────────────────────────


def detect_scene(cap: cv2.VideoCapture) -> tuple[str, float]:
   """
   Samples ~30 frames from the video and returns (profile_key, confidence 0-1).


   Heuristics used:
     Angle  — measures the aspect ratio of the largest moving blobs and the
              average optical-flow angle across sample frames.
              Overhead cars appear as near-square compact blobs moving at
              roughly 0° or 90°; side-view cars are wide rectangles moving
              nearly horizontally.
     Light  — mean brightness of the sampled frames.
              < 80  → night profile
              >= 80 → day profile
   """
   original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
   total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
   fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0


   # Sample up to 30 evenly-spaced frames
   sample_positions = [int(total * i / 30) for i in range(1, 31)]


   brightness_vals = []
   aspect_ratios   = []   # w/h of detected blobs
   flow_angles     = []   # dominant optical flow direction in degrees


   back_sub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)
   prev_gray = None


   for pos in sample_positions:
       cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
       ret, frame = cap.read()
       if not ret:
           continue


       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       brightness_vals.append(float(np.mean(gray)))


       # Background subtraction to find moving blobs
       fg = back_sub.apply(frame)
       _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
       contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       for c in contours:
           area = cv2.contourArea(c)
           if area > 300:
               _, _, w, h = cv2.boundingRect(c)
               if h > 0:
                   aspect_ratios.append(w / h)


       # Sparse optical flow to measure movement direction
       if prev_gray is not None:
           pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
           if pts is not None:
               next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
               good_old = pts[status == 1]
               good_new = next_pts[status == 1]
               for old, new in zip(good_old, good_new):
                   dx = new[0] - old[0]
                   dy = new[1] - old[1]
                   if abs(dx) + abs(dy) > 1.5:
                       angle = abs(np.degrees(np.arctan2(dy, dx)))
                       flow_angles.append(angle)
       prev_gray = gray


   # Restore playback position
   cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)


   # ── Interpret results ──────────────────────────────────────────────────
   mean_brightness = float(np.mean(brightness_vals)) if brightness_vals else 128.0
   is_night        = mean_brightness < 80


   # Aspect ratio: overhead blobs are near-square (ratio ~0.7–1.4)
   #               side-view blobs are wide (ratio > 1.8)
   mean_aspect = float(np.mean(aspect_ratios)) if aspect_ratios else 1.0


   # Flow angle: overhead movement is often ~0° or ~90° (horizontal OR vertical)
   #             side-view movement clusters near 0° or 180° (mostly horizontal)
   #             We detect overhead by checking how many angles fall in the
   #             vertical band (60°–120°), which doesn't happen in side-view.
   vertical_flow_pct = 0.0
   if flow_angles:
       vertical_count  = sum(1 for a in flow_angles if 60 <= a <= 120)
       vertical_flow_pct = vertical_count / len(flow_angles)


   # Score overhead likelihood: compact blobs + any vertical flow component
   overhead_score = 0.0
   if mean_aspect < 1.6:
       overhead_score += 0.5
   if vertical_flow_pct > 0.25:
       overhead_score += 0.5


   confidence = overhead_score if overhead_score >= 0.5 else (1.0 - overhead_score)


   if overhead_score >= 0.5:
       profile_key = "overhead_night" if is_night else "overhead_day"
   else:
       profile_key = "side_night" if is_night else "side_day"


   return profile_key, round(confidence, 2)




def ask_user_scene() -> str:
   # Fallback prompt when auto-detection confidence is too low
   print("\n  Camera angle could not be determined automatically.")
   print("  Please select the scene type:\n")
   options = list(SCENE_PROFILES.keys())
   for i, key in enumerate(options):
       print(f"    [{i+1}] {SCENE_PROFILES[key]['label']}")
   while True:
       choice = input("\n  Enter number: ").strip()
       if choice.isdigit() and 1 <= int(choice) <= len(options):
           return options[int(choice) - 1]
       print("  Invalid choice — try again.")




# ── RIT Lot Inventory 2024-2025 (source: Lot_Inventory_2024-2025.xlsx) ───────
# Keys: lot_name, capacity, sp_acc, sp_admin, sp_reserved, sp_general,
#       sp_parkmobile, sp_mc, sp_ev, sp_visitor, sp_vendor, sp_20min,
#       sp_resident, sp_other, comments
LOTS = [
   {"lot_name": "A",                              "capacity": 150, "sp_acc":  0, "sp_admin":  0, "sp_reserved":   0, "sp_general": 135, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident": 15, "sp_other":  0, "comments": "15-Housing"},
   {"lot_name": "B",                              "capacity": 458, "sp_acc":  2, "sp_admin":  0, "sp_reserved":   0, "sp_general": 456, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "C",                              "capacity": 332, "sp_acc":  0, "sp_admin":  0, "sp_reserved":   0, "sp_general": 332, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": "Motorcycle Pad"},
   {"lot_name": "D",                              "capacity": 271, "sp_acc": 10, "sp_admin":  0, "sp_reserved": 122, "sp_general": 107, "sp_parkmobile":  20, "sp_mc": 0, "sp_ev": 10, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  2, "comments": "2-Motorcycle Pad"},
   {"lot_name": "E / Visitor",                    "capacity": 260, "sp_acc": 27, "sp_admin":  8, "sp_reserved": 145, "sp_general":  32, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor": 46, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "F",                              "capacity": 491, "sp_acc": 32, "sp_admin":  0, "sp_reserved": 242, "sp_general": 217, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "G",                              "capacity": 364, "sp_acc":  0, "sp_admin":  0, "sp_reserved":   0, "sp_general": 300, "sp_parkmobile":  64, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": "PM Daily Decision"},
   {"lot_name": "H",                              "capacity": 709, "sp_acc":  0, "sp_admin":  0, "sp_reserved":   0, "sp_general": 588, "sp_parkmobile": 121, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": "PM Daily Decision"},
   {"lot_name": "J",                              "capacity": 579, "sp_acc": 14, "sp_admin":  6, "sp_reserved": 136, "sp_general": 414, "sp_parkmobile":   9, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "K",                              "capacity": 496, "sp_acc":  0, "sp_admin":  0, "sp_reserved":   0, "sp_general": 496, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "L",                              "capacity": 447, "sp_acc": 16, "sp_admin":  2, "sp_reserved":   0, "sp_general": 415, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev": 14, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": "PM Daily Decision"},
   {"lot_name": "M",                              "capacity": 128, "sp_acc": 12, "sp_admin":  2, "sp_reserved":  82, "sp_general":   0, "sp_parkmobile":  19, "sp_mc": 0, "sp_ev":  2, "sp_visitor":  0, "sp_vendor": 3, "sp_20min":  6, "sp_resident":  2, "sp_other":  0, "comments": "2-Audiologist 2-Rossica"},
   {"lot_name": "N",                              "capacity": 111, "sp_acc":  5, "sp_admin":  0, "sp_reserved":   0, "sp_general": 106, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "R",                              "capacity": 101, "sp_acc": 14, "sp_admin":  0, "sp_reserved":  83, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  4, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "S",                              "capacity": 908, "sp_acc": 28, "sp_admin": 10, "sp_reserved": 284, "sp_general": 516, "sp_parkmobile":  46, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  4, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": "3-GSA, 6-ShopOne, 3-Shear Global, 4-30min, 4-1hr"},
   {"lot_name": "T",                              "capacity": 328, "sp_acc":  6, "sp_admin":  0, "sp_reserved":   0, "sp_general": 273, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev": 11, "sp_visitor": 18, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other": 20, "comments": "18-RRH spaces, 2-Motorcycle Pads"},
   {"lot_name": "U",                              "capacity": 260, "sp_acc": 21, "sp_admin":  0, "sp_reserved": 239, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "Grace Watson Circle",            "capacity":  75, "sp_acc":  6, "sp_admin":  0, "sp_reserved":  34, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  9, "sp_resident":  3, "sp_other": 23, "comments": "9-PS 13-PATS 1-FMS"},
   {"lot_name": "Sol Heumann Circle",             "capacity":  14, "sp_acc":  2, "sp_admin":  0, "sp_reserved":   8, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  4, "comments": "2-FMS 2-ResLife"},
   {"lot_name": "Kate Gleason Circle",            "capacity":  30, "sp_acc":  2, "sp_admin":  0, "sp_reserved":  23, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  3, "sp_other":  0, "comments": "3-RA Spaces"},
   {"lot_name": "Fredericka Douglass Perry Hall", "capacity":  24, "sp_acc":  4, "sp_admin":  0, "sp_reserved":   0, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 3, "sp_20min": 11, "sp_resident":  0, "sp_other":  6, "comments": "2-FMS, 4-ResLife"},
   {"lot_name": "CIMS West",                      "capacity":  11, "sp_acc":  0, "sp_admin":  0, "sp_reserved":  10, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  1, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "7B Lot",                         "capacity":  31, "sp_acc":  0, "sp_admin":  0, "sp_reserved":  27, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  4, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "Eastman Lot",                    "capacity":  26, "sp_acc":  5, "sp_admin": 19, "sp_reserved":   0, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  2, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "Crossroads",                     "capacity":   3, "sp_acc":  0, "sp_admin":  0, "sp_reserved":   0, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  3, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": "4-Global Village"},
   {"lot_name": "Student Health Center",          "capacity":   8, "sp_acc":  4, "sp_admin":  0, "sp_reserved":   2, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 1, "sp_20min":  0, "sp_resident":  0, "sp_other":  3, "comments": "3-Patient Parking, 1 Vendor Only"},
   {"lot_name": "SAU (04 Service)",               "capacity":  12, "sp_acc":  8, "sp_admin":  0, "sp_reserved":   3, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  1, "comments": "1-Brick City"},
   {"lot_name": "LBJ Circle",                     "capacity":   7, "sp_acc":  3, "sp_admin":  2, "sp_reserved":   0, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  1, "sp_resident":  0, "sp_other":  1, "comments": "1-Audio"},
   {"lot_name": "FMS (Bldg 99)",                  "capacity":  43, "sp_acc":  5, "sp_admin":  0, "sp_reserved":  35, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  3, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "Ross (10 Service)",              "capacity":   3, "sp_acc":  0, "sp_admin":  3, "sp_reserved":   0, "sp_general":   0, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "Global Village",                 "capacity": 367, "sp_acc":  8, "sp_admin":  0, "sp_reserved":   0, "sp_general": 358, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  1, "comments": "1-ResLife"},
   {"lot_name": "Riverknoll",                     "capacity": 327, "sp_acc":  3, "sp_admin":  0, "sp_reserved":   0, "sp_general": 320, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor":  0, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  6, "comments": "2-FoodShare, 4-Motorcycle Pads"},
   {"lot_name": "University Commons",             "capacity": 941, "sp_acc": 49, "sp_admin":  0, "sp_reserved":   0, "sp_general": 821, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor": 71, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": ""},
   {"lot_name": "Perkins",                        "capacity": 470, "sp_acc":  8, "sp_admin":  0, "sp_reserved":   0, "sp_general": 380, "sp_parkmobile":   0, "sp_mc": 0, "sp_ev":  0, "sp_visitor": 82, "sp_vendor": 0, "sp_20min":  0, "sp_resident":  0, "sp_other":  0, "comments": ""},
]


def get_inventory() -> list[dict]:
   # Returns the hardcoded lot list
   return LOTS


def get_lots_dict() -> dict[str, int]:
   # Returns {lot_name: capacity} for quick lookups
   return {lot["lot_name"]: lot["capacity"] for lot in LOTS}




# ── Euclidean distance tracker ────────────────────────────────────────────────


class EuclideanDistTracker:
   # Gives each detected bounding box a stable integer ID across frames


   def __init__(self):
       self.center_points: dict[int, tuple[int, int]] = {}
       self.id_count = 0


   def update(self, rects: list[tuple[int, int, int, int]]) -> list[list]:
       # Matches each rect to a known ID by proximity; creates a new ID if no match
       objects_bbs_ids = []
       for x, y, w, h in rects:
           cx, cy     = x + w // 2, y + h // 2
           matched_id = None
           for obj_id, (px, py) in self.center_points.items():
               if np.hypot(cx - px, cy - py) < TRACKER_DIST_THRESH:
                   matched_id = obj_id
                   break
           if matched_id is None:
               matched_id = self.id_count
               self.id_count += 1
           self.center_points[matched_id] = (cx, cy)
           objects_bbs_ids.append([x, y, w, h, matched_id])


       # Prune IDs that didn't appear this frame
       live = {obj[4] for obj in objects_bbs_ids}
       self.center_points = {k: v for k, v in self.center_points.items() if k in live}
       return objects_bbs_ids




# ── Database helpers ──────────────────────────────────────────────────────────


def get_conn() -> sqlite3.Connection:
   # Opens a WAL-mode SQLite connection with dict-style row access
   conn = sqlite3.connect(DB_PATH)
   conn.row_factory = sqlite3.Row
   conn.execute("PRAGMA journal_mode=WAL")
   conn.execute("PRAGMA foreign_keys=ON")
   return conn




def create_schema():
   # Creates all tables on first run; also migrates old DBs missing the sp_ columns
   ddl = """
   CREATE TABLE IF NOT EXISTS lots (
       lot_id        INTEGER PRIMARY KEY AUTOINCREMENT,
       lot_name      TEXT    NOT NULL UNIQUE,
       capacity      INTEGER NOT NULL CHECK(capacity > 0),
       occupancy     INTEGER NOT NULL DEFAULT 0 CHECK(occupancy >= 0),
       active        INTEGER NOT NULL DEFAULT 1,
       sp_acc        INTEGER DEFAULT 0,
       sp_admin      INTEGER DEFAULT 0,
       sp_reserved   INTEGER DEFAULT 0,
       sp_general    INTEGER DEFAULT 0,
       sp_parkmobile INTEGER DEFAULT 0,
       sp_mc         INTEGER DEFAULT 0,
       sp_ev         INTEGER DEFAULT 0,
       sp_visitor    INTEGER DEFAULT 0,
       sp_vendor     INTEGER DEFAULT 0,
       sp_20min      INTEGER DEFAULT 0,
       sp_resident   INTEGER DEFAULT 0,
       sp_other      INTEGER DEFAULT 0,
       comments      TEXT    DEFAULT ''
   );
   CREATE TABLE IF NOT EXISTS vehicle_events (
       event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
       lot_id       INTEGER NOT NULL REFERENCES lots(lot_id),
       tracker_id   INTEGER NOT NULL,
       plate_hash   TEXT,
       plate_region TEXT,
       direction    TEXT NOT NULL CHECK(direction IN ('ENTRY','EXIT')),
       timestamp    TEXT NOT NULL,
       confidence   REAL DEFAULT 1.0
   );
   CREATE TABLE IF NOT EXISTS occupancy_log (
       log_id    INTEGER PRIMARY KEY AUTOINCREMENT,
       lot_id    INTEGER NOT NULL REFERENCES lots(lot_id),
       occupancy INTEGER NOT NULL,
       pct_full  REAL    NOT NULL,
       timestamp TEXT    NOT NULL
   );
   CREATE TABLE IF NOT EXISTS active_sessions (
       session_id INTEGER PRIMARY KEY AUTOINCREMENT,
       lot_id     INTEGER NOT NULL REFERENCES lots(lot_id),
       plate_hash TEXT    NOT NULL,
       entry_time TEXT    NOT NULL,
       UNIQUE(lot_id, plate_hash)
   );
   CREATE TABLE IF NOT EXISTS completed_sessions (
       session_id   INTEGER PRIMARY KEY AUTOINCREMENT,
       lot_id       INTEGER NOT NULL REFERENCES lots(lot_id),
       plate_hash   TEXT    NOT NULL,
       entry_time   TEXT    NOT NULL,
       exit_time    TEXT    NOT NULL,
       duration_min REAL    NOT NULL
   );
   CREATE INDEX IF NOT EXISTS idx_events_lot_ts  ON vehicle_events(lot_id, timestamp);
   CREATE INDEX IF NOT EXISTS idx_events_plate   ON vehicle_events(plate_hash);
   CREATE INDEX IF NOT EXISTS idx_occ_lot_ts     ON occupancy_log(lot_id, timestamp);
   CREATE INDEX IF NOT EXISTS idx_sessions_plate ON active_sessions(plate_hash);
   """
   with get_conn() as conn:
       conn.executescript(ddl)


   # Migration: add any sp_ columns that are missing from an older DB version
   # This runs safely every startup — ALTER TABLE is a no-op if the column exists
   SP_COLUMNS = [
       "sp_acc", "sp_admin", "sp_reserved", "sp_general", "sp_parkmobile",
       "sp_mc", "sp_ev", "sp_visitor", "sp_vendor", "sp_20min",
       "sp_resident", "sp_other", "comments",
   ]
   with get_conn() as conn:
       existing = {row[1] for row in conn.execute("PRAGMA table_info(lots)").fetchall()}
       for col in SP_COLUMNS:
           if col not in existing:
               default = "''" if col == "comments" else "0"
               conn.execute(f"ALTER TABLE lots ADD COLUMN {col} INTEGER DEFAULT {default}")
               print(f"[MIGRATE] Added missing column: lots.{col}")
       conn.commit()




def seed_lots():
   # Inserts all lots from the xlsx; skips rows that already exist (INSERT OR IGNORE)
   with get_conn() as conn:
       for lot in get_inventory():
           conn.execute(
               """INSERT OR IGNORE INTO lots
                  (lot_name, capacity, sp_acc, sp_admin, sp_reserved, sp_general,
                   sp_parkmobile, sp_mc, sp_ev, sp_visitor, sp_vendor, sp_20min,
                   sp_resident, sp_other, comments)
                  VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
               (lot["lot_name"], lot["capacity"],
                lot["sp_acc"],   lot["sp_admin"],   lot["sp_reserved"],
                lot["sp_general"], lot["sp_parkmobile"], lot["sp_mc"],
                lot["sp_ev"],    lot["sp_visitor"], lot["sp_vendor"],
                lot["sp_20min"], lot["sp_resident"], lot["sp_other"],
                lot["comments"])
           )
       conn.commit()




# ── Privacy helpers ───────────────────────────────────────────────────────────


def hash_plate(raw: str) -> str:
   # One-way SHA-256 hash so the original plate string is never persisted
   return hashlib.sha256(raw.strip().upper().encode()).hexdigest()


def plate_region(raw: str) -> str:
   # Pulls the 2-letter state prefix only (e.g. "NY") — non-identifying
   m = re.match(r'^([A-Z]{2})', raw.strip().upper())
   return m.group(1) if m else "XX"




# ── Event writers ─────────────────────────────────────────────────────────────


def get_lot(lot_name: str) -> Optional[sqlite3.Row]:
   # Fetches one lot row by name
   with get_conn() as conn:
       return conn.execute("SELECT * FROM lots WHERE lot_name = ?", (lot_name,)).fetchone()




def log_entry(lot_id: int, tracker_id: int, raw_plate: Optional[str] = None, confidence: float = 1.0) -> int:
   # Writes an ENTRY event, increments occupancy, logs a snapshot, opens a session
   ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   ph = hash_plate(raw_plate) if raw_plate else None
   pr = plate_region(raw_plate) if raw_plate else None
   with get_conn() as conn:
       cur = conn.execute(
           "INSERT INTO vehicle_events (lot_id,tracker_id,plate_hash,plate_region,direction,timestamp,confidence) VALUES (?,?,?,?,?,?,?)",
           (lot_id, tracker_id, ph, pr, "ENTRY", ts, confidence))
       event_id = cur.lastrowid
       conn.execute("UPDATE lots SET occupancy=MIN(capacity,occupancy+1) WHERE lot_id=?", (lot_id,))
       lot = conn.execute("SELECT occupancy,capacity FROM lots WHERE lot_id=?", (lot_id,)).fetchone()
       conn.execute("INSERT INTO occupancy_log (lot_id,occupancy,pct_full,timestamp) VALUES (?,?,?,?)",
                    (lot_id, lot["occupancy"], round(lot["occupancy"]/lot["capacity"]*100,2), ts))
       if ph:
           conn.execute("INSERT OR REPLACE INTO active_sessions (lot_id,plate_hash,entry_time) VALUES (?,?,?)", (lot_id, ph, ts))
       conn.commit()
   return event_id




def log_exit(lot_id: int, tracker_id: int, raw_plate: Optional[str] = None, confidence: float = 1.0) -> int:
   # Writes an EXIT event, decrements occupancy, logs a snapshot, closes the session
   ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   ph = hash_plate(raw_plate) if raw_plate else None
   pr = plate_region(raw_plate) if raw_plate else None
   with get_conn() as conn:
       cur = conn.execute(
           "INSERT INTO vehicle_events (lot_id,tracker_id,plate_hash,plate_region,direction,timestamp,confidence) VALUES (?,?,?,?,?,?,?)",
           (lot_id, tracker_id, ph, pr, "EXIT", ts, confidence))
       event_id = cur.lastrowid
       conn.execute("UPDATE lots SET occupancy=MAX(0,occupancy-1) WHERE lot_id=?", (lot_id,))
       lot = conn.execute("SELECT occupancy,capacity FROM lots WHERE lot_id=?", (lot_id,)).fetchone()
       conn.execute("INSERT INTO occupancy_log (lot_id,occupancy,pct_full,timestamp) VALUES (?,?,?,?)",
                    (lot_id, lot["occupancy"], round(lot["occupancy"]/lot["capacity"]*100,2), ts))
       if ph:
           session = conn.execute("SELECT session_id,entry_time FROM active_sessions WHERE lot_id=? AND plate_hash=?", (lot_id, ph)).fetchone()
           if session:
               dur = (datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") -
                      datetime.strptime(session["entry_time"], "%Y-%m-%d %H:%M:%S")).total_seconds() / 60
               conn.execute("INSERT INTO completed_sessions (lot_id,plate_hash,entry_time,exit_time,duration_min) VALUES (?,?,?,?,?)",
                            (lot_id, ph, session["entry_time"], ts, round(dur, 2)))
               conn.execute("DELETE FROM active_sessions WHERE session_id=?", (session["session_id"],))
       conn.commit()
   return event_id




# ── Analytics queries ─────────────────────────────────────────────────────────


def get_all_lot_status() -> list[dict]:
   # Current occupancy and availability for every active lot
   with get_conn() as conn:
       rows = conn.execute(
           "SELECT lot_name,capacity,occupancy,ROUND(occupancy*100.0/capacity,1) AS pct_full,capacity-occupancy AS available FROM lots WHERE active=1 ORDER BY lot_name"
       ).fetchall()
   return [dict(r) for r in rows]


def get_hourly_traffic(lot_name: str, date: str) -> list[dict]:
   # Entry/exit counts grouped by hour for a given lot and date (YYYY-MM-DD)
   with get_conn() as conn:
       rows = conn.execute(
           "SELECT STRFTIME('%H',ve.timestamp) AS hour,direction,COUNT(*) AS count FROM vehicle_events ve JOIN lots l ON ve.lot_id=l.lot_id WHERE l.lot_name=? AND DATE(ve.timestamp)=? GROUP BY hour,direction ORDER BY hour",
           (lot_name, date)).fetchall()
   return [dict(r) for r in rows]


def get_avg_duration_by_lot() -> list[dict]:
   # Average / min / max dwell time per lot from completed sessions
   with get_conn() as conn:
       rows = conn.execute(
           "SELECT l.lot_name,COUNT(*) AS total_sessions,ROUND(AVG(cs.duration_min),1) AS avg_duration_min,ROUND(MIN(cs.duration_min),1) AS min_duration_min,ROUND(MAX(cs.duration_min),1) AS max_duration_min FROM completed_sessions cs JOIN lots l ON cs.lot_id=l.lot_id GROUP BY l.lot_name ORDER BY avg_duration_min DESC"
       ).fetchall()
   return [dict(r) for r in rows]


def get_peak_occupancy_times(lot_name: str) -> list[dict]:
   # Top-10 moments a lot was most full (feeds the predictive "full by X AM" feature)
   with get_conn() as conn:
       rows = conn.execute(
           "SELECT ol.timestamp,ol.occupancy,ol.pct_full FROM occupancy_log ol JOIN lots l ON ol.lot_id=l.lot_id WHERE l.lot_name=? ORDER BY ol.pct_full DESC LIMIT 10",
           (lot_name,)).fetchall()
   return [dict(r) for r in rows]


def get_daily_summary(date: str) -> list[dict]:
   # Per-lot entry and exit totals for a given date
   with get_conn() as conn:
       rows = conn.execute(
           "SELECT l.lot_name,SUM(CASE WHEN ve.direction='ENTRY' THEN 1 ELSE 0 END) AS entries,SUM(CASE WHEN ve.direction='EXIT' THEN 1 ELSE 0 END) AS exits FROM vehicle_events ve JOIN lots l ON ve.lot_id=l.lot_id WHERE DATE(ve.timestamp)=? GROUP BY l.lot_name ORDER BY entries DESC",
           (date,)).fetchall()
   return [dict(r) for r in rows]


def get_vehicles_over_duration(lot_name: str, hours: float) -> list[dict]:
   # Active sessions longer than `hours`; returns hashed plates only (privacy)
   threshold = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
   with get_conn() as conn:
       rows = conn.execute(
           "SELECT a.plate_hash,a.entry_time,ROUND((JULIANDAY('now')-JULIANDAY(a.entry_time))*24*60,1) AS parked_min FROM active_sessions a JOIN lots l ON a.lot_id=l.lot_id WHERE l.lot_name=? AND a.entry_time<? ORDER BY parked_min DESC",
           (lot_name, threshold)).fetchall()
   return [dict(r) for r in rows]


def get_space_breakdown_summary() -> dict:
   # Campus-wide totals for each space type across all active lots
   with get_conn() as conn:
       row = conn.execute(
           "SELECT SUM(capacity) AS total_spaces,SUM(sp_acc) AS accessible,SUM(sp_admin) AS admin,SUM(sp_reserved) AS reserved,SUM(sp_general) AS general,SUM(sp_parkmobile) AS parkmobile,SUM(sp_ev) AS ev_charging,SUM(sp_visitor) AS visitor,SUM(sp_vendor) AS vendor,SUM(sp_20min) AS short_stay,SUM(sp_resident) AS resident,SUM(sp_other) AS other FROM lots WHERE active=1"
       ).fetchone()
   return dict(row) if row else {}




# ── Camera / video tracking loop ──────────────────────────────────────────────


def count_spaces_in_frame(frame: np.ndarray, profile_key: str) -> int:
   """
   Estimates the number of parking spaces visible in a single frame.


   Strategy:
     1. Convert to grayscale and apply Canny edge detection
     2. Find contours and filter to parking-space-sized rectangles
        (aspect ratio 1.5–4.0, area within expected range for the profile)
     3. Show the annotated frame so the user can verify
     4. Ask the user to confirm the count or enter a manual override


   Returns the confirmed space count.
   """
   p          = SCENE_PROFILES[profile_key]
   gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   blurred    = cv2.GaussianBlur(gray, (5, 5), 0)


   # Canny thresholds tuned per profile: overhead lines are faint, side lines bold
   if "overhead" in profile_key:
       canny_lo, canny_hi = 30, 100
   else:
       canny_lo, canny_hi = 50, 150


   edges    = cv2.Canny(blurred, canny_lo, canny_hi)
   dilated  = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
   contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


   h_img, w_img = frame.shape[:2]
   frame_area   = h_img * w_img


   # Expected size of one parking space in the frame:
   #   overhead: ~0.5–3% of frame area   side/angled: ~2–8%
   if "overhead" in profile_key:
       area_lo, area_hi = frame_area * 0.003, frame_area * 0.04
   else:
       area_lo, area_hi = frame_area * 0.01,  frame_area * 0.10


   annotated   = frame.copy()
   space_count = 0


   for c in contours:
       area = cv2.contourArea(c)
       if not (area_lo < area < area_hi):
           continue
       x, y, w, h = cv2.boundingRect(c)
       aspect = max(w, h) / max(min(w, h), 1)
       if 1.4 < aspect < 5.0:          # parking spaces are elongated rectangles
           space_count += 1
           cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 100), 2)
           cv2.putText(annotated, str(space_count), (x+4, y+16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


   # Show the annotated frame for user confirmation
   win = "Space detection — press any key to continue"
   cv2.namedWindow(win, cv2.WINDOW_NORMAL)
   cv2.putText(annotated, f"Detected: {space_count} spaces", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
   cv2.putText(annotated, "Press any key, then confirm in terminal", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
   cv2.imshow(win, annotated)
   cv2.waitKey(0)
   cv2.destroyWindow(win)


   print(f"\n[SPACES] Auto-detected {space_count} parking spaces in the frame.")
   print(f"         If this looks wrong, enter the correct number.")
   override = input(f"         Confirm count [{space_count}]: ").strip()
   try:
       confirmed = int(override) if override else space_count
   except ValueError:
       confirmed = space_count


   confirmed = max(1, confirmed)   # must be at least 1
   print(f"[SPACES] Using capacity: {confirmed}")
   return confirmed




def register_video_lot(lot_name: str, capacity: int) -> int:
   """
   Ensures a lot row exists in the DB for this video session.
   If the lot already exists its capacity is updated to the detected value.
   Returns the lot_id.
   """
   with get_conn() as conn:
       existing = conn.execute(
           "SELECT lot_id FROM lots WHERE lot_name=?", (lot_name,)).fetchone()
       if existing:
           conn.execute(
               "UPDATE lots SET capacity=?, occupancy=0 WHERE lot_name=?",
               (capacity, lot_name))
           lot_id = existing["lot_id"]
       else:
           cur = conn.execute(
               "INSERT INTO lots (lot_name, capacity, occupancy) VALUES (?,?,0)",
               (lot_name, capacity))
           lot_id = cur.lastrowid
       conn.commit()
   print(f"[DB] Lot '{lot_name}' registered — capacity={capacity}  lot_id={lot_id}")
   return lot_id




def run_camera_tracker(source=0, lot_name: str = "", playback_speed: float = 1.0):
   # Tracks vehicles from any video file or live camera.
   # Auto-detects scene profile + counts visible spaces; no hardcoded lot needed.
   #
   # Keyboard controls (video):
   #   SPACE  pause/resume    →/←  step one frame    +/-  speed
   #   d      toggle debug mask    r  restart         q  quit


   is_video = isinstance(source, str)
   if is_video and not os.path.isfile(source):
       print(f"[ERROR] File not found: {source}"); return


   cap = cv2.VideoCapture(source)
   if not cap.isOpened():
       print("[ERROR] Could not open video source."); return


   frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0
   native_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
   duration_sec = total_frames / native_fps if total_frames else 0
   speed        = max(0.25, playback_speed)


   print(f"[INFO] {frame_w}x{frame_h}  {native_fps:.1f}fps  {duration_sec:.1f}s")


   # ── 1. Auto-detect scene profile ───────────────────────────────────────
   CONFIDENCE_THRESHOLD = 0.70
   if is_video:
       print("[INFO] Analysing scene type...")
       profile_key, confidence = detect_scene(cap)
       if confidence >= CONFIDENCE_THRESHOLD:
           print(f"[INFO] Detected: '{SCENE_PROFILES[profile_key]['label']}'  "
                 f"(confidence {confidence:.0%})")
       else:
           print(f"[INFO] Low confidence ({confidence:.0%}) — asking user.")
           profile_key = ask_user_scene()
   else:
       profile_key = ask_user_scene()


   p = SCENE_PROFILES[profile_key]
   print(f"[INFO] Profile: {p['label']}")
   if is_video:
       print("  SPACE=pause  →/←=step  +/-=speed  d=mask  r=restart  q=quit")


   # ── 2. Count parking spaces from a sample frame ────────────────────────
   if is_video and total_frames > 0:
       cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.05))
   ret, sample_frame = cap.read()
   if not ret:
       print("[ERROR] Could not read a frame."); cap.release(); return


   print("[INFO] Counting parking spaces in frame...")
   detected_capacity = count_spaces_in_frame(sample_frame, profile_key)


   # ── 3. Register the lot in the DB ──────────────────────────────────────
   if not lot_name:
       lot_name = os.path.splitext(os.path.basename(source))[0] if is_video else "live_camera"
   lot_id = register_video_lot(lot_name, detected_capacity)
   cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # reset to start after sampling


   # ── 4. Build CV objects from profile ───────────────────────────────────
   def make_back_sub():
       return cv2.createBackgroundSubtractorMOG2(
           history=p["mog2_history"],
           varThreshold=p["mog2_var"],
           detectShadows=p["mog2_shadows"]
       )


   back_sub = make_back_sub()
   tracker  = EuclideanDistTracker()
   kernel   = cv2.getStructuringElement(p["morph_shape"],
                                        (p["morph_kernel"], p["morph_kernel"]))
   cooldown = timedelta(seconds=p["cooldown"])


   if p["line_axis"] == "horizontal":
       line_pos = frame_h // 2
       line_pt1, line_pt2 = (0, line_pos), (frame_w, line_pos)
   else:
       line_pos = frame_w // 2
       line_pt1, line_pt2 = (line_pos, 0), (line_pos, frame_h)


   ENTRY_CROSSING = {
       "top_to_bottom":  ("top",    "bottom"),
       "bottom_to_top":  ("bottom","top"),
       "right_to_left":  ("right", "left"),
       "left_to_right":  ("left",  "right"),
   }
   entry_cross = ENTRY_CROSSING[p["entry_dir"]]


   def get_side(x, y):
       # Returns which side of the counting line a detection centre is on
       if p["line_axis"] == "horizontal":
           return "top" if y < line_pos else "bottom"
       else:
           return "left" if x < line_pos else "right"


   side_state:      dict[int, str]      = {}
   already_counted: dict[int, bool]     = {}
   last_logged:     dict[int, datetime] = {}


   entry_count = exit_count = 0
   paused     = False
   show_mask  = False
   frame_num  = 0
   last_frame = None
   title      = f"Parking Tracker — {lot_name}  [{p['label']}]  cap={detected_capacity}"
   mask_title = "Debug mask  (d = hide)"


   cv2.namedWindow(title, cv2.WINDOW_NORMAL)


   if is_video and total_frames > 0:
       scrubbing = [False]
       def on_trackbar(pos):
           scrubbing[0] = True
           cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
           ret, f = cap.read()
           if ret: cv2.imshow(title, f)
           scrubbing[0] = False
       cv2.createTrackbar("Frame", title, 0, max(1, total_frames - 1), on_trackbar)
   else:
       scrubbing = [False]


   def _reset():
       nonlocal back_sub, tracker, entry_count, exit_count
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
       back_sub = make_back_sub()
       tracker  = EuclideanDistTracker()
       side_state.clear(); already_counted.clear(); last_logged.clear()
       entry_count = exit_count = 0
       with get_conn() as conn:
           conn.execute("UPDATE lots SET occupancy=0 WHERE lot_id=?", (lot_id,))
           conn.commit()


   while True:
       if not paused:
           ret, frame = cap.read()
           if not ret:
               if is_video:
                   print("\n[INFO] End of video — r=restart  q=quit")
                   paused = True
                   if last_frame is not None: cv2.imshow(title, last_frame)
                   key = cv2.waitKey(0) & 0xFF
                   if key == ord('r'):
                       _reset(); paused = False; continue
                   else: break
               else:
                   print("[INFO] Stream ended."); break


           frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
           if is_video and total_frames > 0 and not scrubbing[0]:
               cv2.setTrackbarPos("Frame", title, min(frame_num, total_frames - 1))


           # 1. Background subtraction
           fg_raw = back_sub.apply(frame)


           # 2. Shadow suppression: threshold > 200 keeps only true foreground (255);
           #    shadow pixels (127) are discarded
           _, fg = cv2.threshold(fg_raw, 200, 255, cv2.THRESH_BINARY)


           # 3. Morphological cleanup: open removes noise, close fills holes in car body
           cleaned = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel, iterations=2)
           cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)


           # 4. Find and filter contours by minimum area from profile
           contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
           detections  = [cv2.boundingRect(c) for c in contours
                          if cv2.contourArea(c) > p["min_area"]]
           boxes_ids   = tracker.update(detections)


           # Draw counting line
           cv2.line(frame, line_pt1, line_pt2, (0, 220, 255), 2)


           for x, y, w, h, vid in boxes_ids:
               cx_box = x + w // 2
               cy_box = y + h // 2
               now_side = get_side(cx_box, cy_box)


               if vid not in side_state:
                   side_state[vid] = now_side
                   already_counted[vid] = False
                   last_logged[vid]     = datetime.min


               prev_side       = side_state[vid]
               side_state[vid] = now_side


               if prev_side != now_side and not already_counted[vid]:
                   now = datetime.now()
                   if now - last_logged[vid] > cooldown:
                       if (prev_side, now_side) == entry_cross:
                           entry_count += 1; log_entry(lot_id, vid)
                           print(f"  ENTRY  frame={frame_num:>6}  id={vid}  total_in={entry_count}")
                       else:
                           exit_count += 1; log_exit(lot_id, vid)
                           print(f"  EXIT   frame={frame_num:>6}  id={vid}  total_out={exit_count}")
                       already_counted[vid] = True
                       last_logged[vid]     = now


               color = (0, 255, 80) if now_side == entry_cross[1] else (0, 165, 255)
               cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
               cv2.putText(frame, f"ID {vid}", (x, y - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


           # HUD — live occupancy pulled from DB, capacity from space detection
           lot_s   = get_lot(lot_name)
           occ     = lot_s["occupancy"] if lot_s else (entry_count - exit_count)
           avail   = detected_capacity - occ
           pct     = round(occ / detected_capacity * 100) if detected_capacity else 0
           avail_col = (0,255,80) if pct < 60 else ((0,165,255) if pct < 85 else (0,0,255))
           hud = [
               (lot_name,                                   (255, 220,   0), 0.60, 2),
               (f"Capacity: {detected_capacity} spaces",    (180, 180, 180), 0.48, 1),
               (f"In: {entry_count}   Out: {exit_count}",   (200, 200, 200), 0.52, 1),
               (f"Occupied: {occ}   Available: {avail}",    avail_col,       0.52, 1),
               (f"{pct}% full",                             avail_col,       0.52, 1),
               (p["label"],                                 (130, 130, 130), 0.42, 1),
           ]
           if is_video:
               hud.append(("PAUSED" if paused else f"x{speed:.2f}  [d]=mask", (0,200,255), 0.42, 1))
           cv2.rectangle(frame, (0, 0), (300, len(hud) * 23 + 10), (0, 0, 0), -1)
           for i, (txt, col, sc, th2) in enumerate(hud):
               cv2.putText(frame, txt, (6, 20 + i * 22),
                           cv2.FONT_HERSHEY_SIMPLEX, sc, col, th2)


           # Progress bar (video only)
           if is_video and total_frames > 0:
               prog  = frame_num / total_frames
               bar_w = frame_w - 20
               cv2.rectangle(frame, (10, frame_h-14), (10+bar_w, frame_h-6), (50,50,50), -1)
               cv2.rectangle(frame, (10, frame_h-14),
                             (10+int(bar_w*prog), frame_h-6), (0,200,255), -1)
               elapsed = frame_num / native_fps
               cv2.putText(frame,
                   f"{int(elapsed//60):02d}:{int(elapsed%60):02d} / "
                   f"{int(duration_sec//60):02d}:{int(duration_sec%60):02d}",
                   (10, frame_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)


           last_frame = frame.copy()
           cv2.imshow(title, frame)


           if show_mask:
               mask_vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
               cv2.line(mask_vis, line_pt1, line_pt2, (0, 220, 255), 2)
               cv2.imshow(mask_title, mask_vis)


       wait_ms = max(1, int(1000 / native_fps / speed)) if (is_video and not paused) else 30
       key = cv2.waitKey(1 if not is_video else wait_ms) & 0xFF


       if   key == ord('q'):                             break
       elif key == ord(' ') and is_video:                paused = not paused
       elif key == ord('d'):
           show_mask = not show_mask
           if not show_mask: cv2.destroyWindow(mask_title)
       elif key == ord('r') and is_video:                _reset(); paused = False
       elif key in (ord('+'), ord('=')) and is_video:    speed = round(min(speed+0.25, 8.0), 2)
       elif key == ord('-') and is_video:                speed = round(max(speed-0.25, 0.25), 2)
       elif key == 83 and paused and is_video:           # → step forward
           ret, frame = cap.read()
           if ret:
               frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
               cv2.imshow(title, frame); last_frame = frame.copy()
               if total_frames > 0:
                   cv2.setTrackbarPos("Frame", title, min(frame_num, total_frames-1))
       elif key == 81 and paused and is_video:           # ← step backward
           cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_num-2))
           ret, frame = cap.read()
           if ret:
               frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
               cv2.imshow(title, frame); last_frame = frame.copy()
               if total_frames > 0:
                   cv2.setTrackbarPos("Frame", title, min(frame_num, total_frames-1))


   cap.release()
   cv2.destroyAllWindows()
   print(f"\n[INFO] Done — Entered: {entry_count}  Exited: {exit_count}  "
         f"Final occupancy: {entry_count - exit_count}/{detected_capacity}")




# ── Simulation ────────────────────────────────────────────────────────────────


def run_simulation():
   # Generates a realistic day of synthetic traffic across all lots for DB testing
   import random
   print("\n[SIM] Simulating today's traffic...\n")
   today       = datetime.now().strftime("%Y-%m-%d")
   fake_plates = [f"NY{i:04d}" for i in range(1, 301)]


   with get_conn() as conn:
       lots_rows = conn.execute("SELECT lot_id, lot_name, capacity FROM lots").fetchall()


   for lot_row in lots_rows:
       lot_id, lot_name, capacity = lot_row["lot_id"], lot_row["lot_name"], lot_row["capacity"]
       print(f"  {lot_name}  cap={capacity}")
       with get_conn() as conn:
           conn.execute("UPDATE lots SET occupancy=0 WHERE lot_id=?", (lot_id,))
           conn.execute("DELETE FROM active_sessions WHERE lot_id=?", (lot_id,))
           conn.commit()


       active: dict[str, datetime] = {}
       events = []


       base = datetime.strptime(f"{today} 07:30:00", "%Y-%m-%d %H:%M:%S")
       for _ in range(int(capacity * 0.80)):  # morning rush
           events.append((base + timedelta(seconds=random.randint(0, 7200)), "ENTRY", random.choice(fake_plates)))


       base2 = datetime.strptime(f"{today} 10:00:00", "%Y-%m-%d %H:%M:%S")
       for _ in range(int(capacity * 0.15)):  # midday turnover
           out = timedelta(seconds=random.randint(0, 10800))
           pl  = random.choice(fake_plates)
           events.append((base2 + out, "EXIT", pl))
           events.append((base2 + out + timedelta(minutes=random.randint(5, 30)), "ENTRY", pl))


       base3 = datetime.strptime(f"{today} 15:00:00", "%Y-%m-%d %H:%M:%S")
       for _ in range(int(capacity * 0.70)):  # afternoon departures
           events.append((base3 + timedelta(seconds=random.randint(0, 10800)), "EXIT", random.choice(fake_plates)))


       events.sort(key=lambda e: e[0])


       for ts_dt, direction, raw_plate in events:
           ts = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
           ph = hash_plate(raw_plate)
           with get_conn() as conn:
               conn.execute(
                   "INSERT INTO vehicle_events (lot_id,tracker_id,plate_hash,plate_region,direction,timestamp,confidence) VALUES (?,?,?,?,?,?,?)",
                   (lot_id, 0, ph, "NY", direction, ts, 0.95))
               if direction == "ENTRY":
                   conn.execute("UPDATE lots SET occupancy=MIN(capacity,occupancy+1) WHERE lot_id=?", (lot_id,))
                   lot = conn.execute("SELECT occupancy,capacity FROM lots WHERE lot_id=?", (lot_id,)).fetchone()
                   conn.execute("INSERT INTO occupancy_log (lot_id,occupancy,pct_full,timestamp) VALUES (?,?,?,?)",
                                (lot_id, lot["occupancy"], round(lot["occupancy"]/lot["capacity"]*100,2), ts))
                   conn.execute("INSERT OR REPLACE INTO active_sessions (lot_id,plate_hash,entry_time) VALUES (?,?,?)", (lot_id, ph, ts))
                   active[ph] = ts_dt
               else:
                   conn.execute("UPDATE lots SET occupancy=MAX(0,occupancy-1) WHERE lot_id=?", (lot_id,))
                   lot = conn.execute("SELECT occupancy,capacity FROM lots WHERE lot_id=?", (lot_id,)).fetchone()
                   conn.execute("INSERT INTO occupancy_log (lot_id,occupancy,pct_full,timestamp) VALUES (?,?,?,?)",
                                (lot_id, lot["occupancy"], round(lot["occupancy"]/lot["capacity"]*100,2), ts))
                   if ph in active:
                       dur = (ts_dt - active.pop(ph)).total_seconds() / 60
                       conn.execute("INSERT INTO completed_sessions (lot_id,plate_hash,entry_time,exit_time,duration_min) VALUES (?,?,?,?,?)",
                                    (lot_id, ph, (ts_dt-timedelta(minutes=dur)).strftime("%Y-%m-%d %H:%M:%S"), ts, round(dur,2)))
                       conn.execute("DELETE FROM active_sessions WHERE lot_id=? AND plate_hash=?", (lot_id, ph))
               conn.commit()


   print("\n[SIM] Done.\n")




# ── CLI dashboard ─────────────────────────────────────────────────────────────


def print_dashboard():
   # Prints a formatted summary: lot status, space breakdown, traffic, and alerts
   today     = datetime.now().strftime("%Y-%m-%d")
   inv       = get_inventory()
   total_cap = sum(l["capacity"] for l in inv)


   print("\n" + "="*70)
   print("  RIT SMART PARKING TRACKER")
   print(f"  {datetime.now().strftime('%A, %B %d %Y  %I:%M %p')}")
   print(f"  {len(inv)} lots  |  {total_cap:,} spaces  |  source: {INVENTORY_XLSX}")
   print("="*70)


   print("\n  LOT AVAILABILITY\n")
   print(f"  {'Lot':<32}{'Cap':>6}{'Parked':>8}{'Avail':>7}{'%Full':>7}  Bar")
   print("  " + "-"*70)
   for s in get_all_lot_status():
       bar = "█" * int(s["pct_full"]/5) + "░" * (20 - int(s["pct_full"]/5))
       print(f"  {s['lot_name']:<32}{s['capacity']:>6}{s['occupancy']:>8}{s['available']:>7}{s['pct_full']:>6.1f}%  {bar}")


   print(f"\n  CAMPUS SPACE BREAKDOWN\n")
   bd = get_space_breakdown_summary()
   if bd:
       total = bd.get("total_spaces", 1) or 1
       rows  = sorted([
           ("General permit",    bd.get("general",    0)),
           ("Reserved",          bd.get("reserved",   0)),
           ("ParkMobile",        bd.get("parkmobile", 0)),
           ("Visitor",           bd.get("visitor",    0)),
           ("Accessible (ACC)",  bd.get("accessible", 0)),
           ("EV Charging",       bd.get("ev_charging",0)),
           ("Admin",             bd.get("admin",      0)),
           ("Resident",          bd.get("resident",   0)),
           ("Short-stay ≤20min", bd.get("short_stay", 0)),
           ("Vendor",            bd.get("vendor",     0)),
           ("Other/Special",     bd.get("other",      0)),
       ], key=lambda x: -x[1])
       print(f"  {'Type':<22}{'Spaces':>8}  {'Share':>6}")
       print("  " + "-"*42)
       for label, count in rows:
           print(f"  {label:<22}{count:>8}  {count/total*100:>5.1f}%")
       print(f"  {'TOTAL':<22}{total:>8}  100.0%")


   print(f"\n  TODAY'S TRAFFIC  ({today})\n")
   summary = get_daily_summary(today)
   if summary:
       print(f"  {'Lot':<32}{'Entries':>9}{'Exits':>8}")
       print("  " + "-"*50)
       for r in summary:
           print(f"  {r['lot_name']:<32}{r['entries']:>9}{r['exits']:>8}")
   else:
       print("  No events today.")


   print("\n  AVERAGE DURATION\n")
   for d in get_avg_duration_by_lot():
       print(f"  {d['lot_name']:<32}{d['total_sessions']:>6} sessions  avg={d['avg_duration_min']}min")


   print("\n  OVERSTAY ALERTS  (Lot S > 4 hours)\n")
   alerts = get_vehicles_over_duration("S", hours=4)
   if alerts:
       for a in alerts[:5]:
           print(f"  {a['plate_hash'][:16]}...  entry={a['entry_time']}  parked={a['parked_min']}min")
   else:
       print("  None.")


   print("\n" + "="*70 + "\n")




# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
   create_schema()  # build tables if this is the first run
   seed_lots()      # load lot inventory from xlsx into the DB


   print("\n RIT Smart Parking Tracker")
   print(" ──────────────────────────")
   print(" [1] Track from video file")
   print(" [2] Live camera tracker")
   print(" [3] Simulate a full day (fill DB with test data)")
   print(" [4] Show dashboard only")
   print(f"\n Lots: {', '.join(sorted(get_lots_dict().keys()))}")
   choice = input("\n Select (1/2/3/4): ").strip()


   if choice == "1":
       print(f"\n Formats: {', '.join(VIDEO_EXTENSIONS)}")
       video_path = input(" Video path: ").strip().strip('"').strip("'")
       label      = input(" Label for this session (optional, press Enter to use filename): ").strip()
       speed_str  = input(" Speed — 1.0=normal, 2.0=double [1.0]: ").strip()
       try:    speed = float(speed_str) if speed_str else 1.0
       except: speed = 1.0
       run_camera_tracker(source=video_path, lot_name=label, playback_speed=speed)
       print_dashboard()


   elif choice == "2":
       label = input(" Label for this live session (optional): ").strip()
       run_camera_tracker(source=0, lot_name=label)
       print_dashboard()


   elif choice == "3":
       run_simulation()
       print_dashboard()


   else:
       print_dashboard()

