import cv2
import json

spots = []
current = []

def mouse(event, x, y, flags, param):
    global current, spots

    if event == cv2.EVENT_LBUTTONDOWN:
        current.append((x, y))
        print(f"Point added: {(x, y)}")

        # once 4 points → save a spot
        if len(current) == 4:
            spots.append(current.copy())
            print(f"Spot saved: {current}")
            current = []

    elif event == cv2.EVENT_RBUTTONDOWN:
        if spots:
            removed = spots.pop()
            print(f"Removed spot: {removed}")


def run(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open video")
        return

    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        return

    cv2.namedWindow("Label Parking Spots")
    cv2.setMouseCallback("Label Parking Spots", mouse)

    print("\nINSTRUCTIONS:")
    print("- Click 4 points per parking spot")
    print("- Right click = undo last spot")
    print("- Press S = save")
    print("- Press Q = quit\n")

    while True:
        img = frame.copy()

        # draw completed spots
        for spot in spots:
            for i in range(4):
                cv2.line(img, spot[i], spot[(i+1) % 4], (0,255,0), 2)

        # draw current in-progress shape
        for p in current:
            cv2.circle(img, p, 5, (0,0,255), -1)

        cv2.imshow("Label Parking Spots", img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            with open("spots.json", "w") as f:
                json.dump(spots, f)
            print(f"\nSaved {len(spots)} spots")
            break

        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    path = input("Video path: ").strip().strip('"')
    run(path)