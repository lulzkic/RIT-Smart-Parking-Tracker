import cv2
import numpy as np
import json
import os


# ─────────────────────────────────────────────
# MODE SELECTOR
# ─────────────────────────────────────────────
def choose_mode():
    print("\n=== PARKING DETECTION MODES ===")
    print("1 - AUTO DETECT (your original method)")
    print("2 - POLYGON MODE (recommended, uses spots.json)")
    print("3 - HYBRID (auto detect + cleanup)\n")

    choice = input("Select mode (1/2/3): ")

    if choice == "2":
        return "polygon"
    elif choice == "3":
        return "hybrid"
    else:
        return "auto"


# ─────────────────────────────────────────────
# LOAD POLYGONS (OPTIONAL MODE)
# ─────────────────────────────────────────────
def load_spots(path="spots.json"):
    if not os.path.exists(path):
        print("[WARN] spots.json not found. Switching to AUTO mode.")
        return []

    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# YOUR ORIGINAL STYLE: AUTO DETECTION
# (fixed slightly for stability)
# ─────────────────────────────────────────────
def detect_spaces_auto(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    spaces = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if 2000 < area < 20000:
            x, y, w, h = cv2.boundingRect(cnt)

            # filter weird shapes
            aspect = w / float(h)

            if 0.5 < aspect < 3.0:
                spaces.append((x, y, w, h))

    return spaces


# ─────────────────────────────────────────────
# OCCUPANCY CHECK (FIXED LOGIC)
# ─────────────────────────────────────────────
def is_occupied_mask(frame, mask, bg_sub, threshold=900):
    fg = bg_sub.apply(frame)
    fg = cv2.bitwise_and(fg, fg, mask=mask)

    count = cv2.countNonZero(fg)

    return count > threshold


# ─────────────────────────────────────────────
# CONVERT RECT TO POLYGON MASK
# ─────────────────────────────────────────────
def rect_to_mask(frame_shape, rect):
    x, y, w, h = rect

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)

    pts = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], np.int32)

    cv2.fillPoly(mask, [pts], 255)

    return mask, pts


# ─────────────────────────────────────────────
# POLYGON MODE CHECK
# ─────────────────────────────────────────────
def polygon_occupied(frame, polygon, bg_sub, threshold=1200):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(polygon, np.int32)

    cv2.fillPoly(mask, [pts], 255)

    fg = bg_sub.apply(frame)
    fg = cv2.bitwise_and(fg, fg, mask=mask)

    return cv2.countNonZero(fg) > threshold


# ─────────────────────────────────────────────
# MAIN PROGRAM
# ─────────────────────────────────────────────
def run(video_path=0):

    mode = choose_mode()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Cannot open video")
        return

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False
    )

    spots = load_spots() if mode == "polygon" else []

    print(f"[INFO] Running in {mode.upper()} mode")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        occupied = 0
        total = 0

        # ─────────────────────────────
        # MODE 1: AUTO (your original idea)
        # ─────────────────────────────
        if mode == "auto":

            spaces = detect_spaces_auto(frame)

            for (x, y, w, h) in spaces:
                mask, pts = rect_to_mask(frame.shape, (x, y, w, h))

                occ = is_occupied_mask(frame, mask, bg_sub)

                color = (0, 0, 255) if occ else (0, 255, 0)

                if occ:
                    occupied += 1

                total += 1

                cv2.polylines(frame, [pts], True, color, 2)

        # ─────────────────────────────
        # MODE 2: POLYGON (best accuracy)
        # ─────────────────────────────
        elif mode == "polygon":

            for spot in spots:

                occ = polygon_occupied(frame, spot, bg_sub)

                color = (0, 0, 255) if occ else (0, 255, 0)

                if occ:
                    occupied += 1

                total += 1

                cv2.polylines(frame, [np.array(spot)], True, color, 2)

        # ─────────────────────────────
        # MODE 3: HYBRID
        # ─────────────────────────────
        else:

            spaces = detect_spaces_auto(frame)

            for (x, y, w, h) in spaces:

                mask, pts = rect_to_mask(frame.shape, (x, y, w, h))

                occ = is_occupied_mask(frame, mask, bg_sub)

                # extra smoothing (reduces flicker)
                if occ and w * h < 30000:
                    occ = True

                color = (0, 0, 255) if occ else (0, 255, 0)

                if occ:
                    occupied += 1

                total += 1

                cv2.polylines(frame, [pts], True, color, 2)

        # ─────────────────────────────
        # UI TEXT
        # ─────────────────────────────
        free = max(total - occupied, 0)

        cv2.putText(frame,
                    f"Mode: {mode.upper()}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        cv2.putText(frame,
                    f"Occupied: {occupied}  Free: {free}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2)

        cv2.imshow("Parking System", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run("D:\download folder\parkinglot1.mp4")   # change this