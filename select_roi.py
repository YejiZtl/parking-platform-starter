import argparse
import json
from pathlib import Path

import cv2


WINDOW = "Select parking ROI"


class RoiSelector:
    def __init__(self, image_path: Path, output_path: Path):
        self.image_path = image_path
        self.output_path = output_path
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise SystemExit(f"Could not read image: {image_path}")
        self.points = []
        self.preview = self.image.copy()

    def draw(self):
        self.preview = self.image.copy()
        if len(self.points) >= 2:
            for index in range(len(self.points) - 1):
                cv2.line(self.preview, self.points[index], self.points[index + 1], (0, 0, 255), 3)
        if len(self.points) >= 3:
            cv2.line(self.preview, self.points[-1], self.points[0], (0, 0, 255), 3)
        for point in self.points:
            cv2.circle(self.preview, point, 5, (0, 255, 255), -1)

        text = "left-click:add point  right-click:undo  s:save  q:quit"
        cv2.putText(self.preview, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(self.preview, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.draw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self.draw()

    def save(self):
        if len(self.points) < 3:
            print("ROI needs at least 3 points.")
            return
        data = {
            "image": str(self.image_path),
            "points": [[int(x), int(y)] for x, y in self.points],
        }
        self.output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved ROI: {self.output_path}")

    def run(self):
        self.draw()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        while True:
            cv2.imshow(WINDOW, self.preview)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                self.save()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Draw a polygon ROI for the parking lot.")
    parser.add_argument("--image", default="first_frame.jpg", help="Image used to draw the ROI.")
    parser.add_argument("--output", default="parking_roi.json", help="ROI JSON output path.")
    args = parser.parse_args()

    RoiSelector(Path(args.image), Path(args.output)).run()


if __name__ == "__main__":
    main()
