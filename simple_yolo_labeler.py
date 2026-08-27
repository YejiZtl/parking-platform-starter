import argparse
import os
from pathlib import Path

import cv2

from parking_geometry import load_roi

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
WINDOW = "YOLO vehicle labeler"


class Labeler:
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        start: int = 1,
        limit: int | None = None,
        roi_path: str | None = "parking_roi.json",
        max_width: int = 1500,
        max_height: int = 850,
        confirmed_output: Path | None = None,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_dir.mkdir(parents=True, exist_ok=True)
        all_images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        start_index = max(0, start - 1)
        end_index = None if limit is None else start_index + limit
        self.images = all_images[start_index:end_index]
        if not self.images:
            raise SystemExit(f"No images found in {image_dir}")

        self.index = 0
        self.boxes = []
        self.drawing = False
        self.start = None
        self.current = None
        self.preview = None
        self.display_scale = 1.0
        self.max_width = max(320, max_width)
        self.max_height = max(240, max_height)
        self.mouse_pos = (0, 0)
        self.hover_index = None
        self.roi = load_roi(roi_path)
        self.confirmed_output = confirmed_output
        self.confirmed_images: set[str] = set()

    def label_path(self, image_path: Path) -> Path:
        return self.label_dir / f"{image_path.stem}.txt"

    def load_boxes(self, image_path: Path, width: int, height: int):
        path = self.label_path(image_path)
        boxes = []
        if not path.exists():
            return boxes

        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw / 2) * width)
            y1 = int((cy - bh / 2) * height)
            x2 = int((cx + bw / 2) * width)
            y2 = int((cy + bh / 2) * height)
            boxes.append((x1, y1, x2, y2))
        return boxes

    def save_boxes(self, image_path: Path, width: int, height: int):
        lines = []
        for x1, y1, x2, y2 in self.boxes:
            x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
            y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            if bw <= 0 or bh <= 0:
                continue
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        self.label_path(image_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def confirm_current(self, image_path: Path, width: int, height: int):
        self.save_boxes(image_path, width, height)
        self.confirmed_images.add(image_path.name)
        print(f"Confirmed labels: {self.label_path(image_path)}")

    def write_confirmed_output(self):
        if self.confirmed_output is None:
            return
        self.confirmed_output.parent.mkdir(parents=True, exist_ok=True)
        self.confirmed_output.write_text(
            "\n".join(sorted(self.confirmed_images)) + ("\n" if self.confirmed_images else ""),
            encoding="utf-8",
        )

    def normalized_box(self, box):
        x1, y1, x2, y2 = box
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        return left, top, right, bottom

    def box_at_point(self, x: int, y: int):
        for index in range(len(self.boxes) - 1, -1, -1):
            left, top, right, bottom = self.normalized_box(self.boxes[index])
            if left <= x <= right and top <= y <= bottom:
                return index
        return None

    def delete_box_at(self, x: int, y: int) -> bool:
        index = self.box_at_point(x, y)
        if index is None:
            return False
        self.boxes.pop(index)
        self.hover_index = None
        self.draw()
        return True

    def draw(self):
        canvas = self.current.copy()
        if self.roi is not None:
            cv2.polylines(canvas, [self.roi], isClosed=True, color=(0, 0, 255), thickness=4)
        self.hover_index = None if self.drawing else self.box_at_point(*self.mouse_pos)
        for index, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self.normalized_box(box)
            color = (0, 255, 255) if index == self.hover_index else (0, 255, 0)
            thickness = 3 if index == self.hover_index else 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        if self.drawing and self.start:
            x1, y1 = self.start
            x2, y2 = self.mouse_pos
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

        height, width = canvas.shape[:2]
        self.display_scale = min(1.0, self.max_width / width, self.max_height / height)
        if self.display_scale < 1.0:
            self.preview = cv2.resize(
                canvas,
                (round(width * self.display_scale), round(height * self.display_scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            self.preview = canvas

        text = f"{self.index + 1}/{len(self.images)} boxes:{len(self.boxes)}  right-click/d=delete s=save n=next p=prev u=undo q=quit"
        cv2.putText(self.preview, text, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(self.preview, text, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    def on_mouse(self, event, x, y, flags, param):
        if self.current is None:
            return
        height, width = self.current.shape[:2]
        x = max(0, min(width - 1, round(x / self.display_scale)))
        y = max(0, min(height - 1, round(y / self.display_scale)))
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.delete_box_at(x, y):
                print("Deleted box. Press s or n to save.")
            else:
                self.draw()
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.draw()
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1, y1 = self.start
            if abs(x - x1) > 5 and abs(y - y1) > 5:
                self.boxes.append((x1, y1, x, y))
            self.start = None
            self.draw()

    def open_current(self):
        image_path = self.images[self.index]
        self.current = cv2.imread(str(image_path))
        if self.current is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = self.current.shape[:2]
        self.boxes = self.load_boxes(image_path, width, height)
        self.mouse_pos = (0, 0)
        self.draw()

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.open_current()

        while True:
            cv2.imshow(WINDOW, self.preview)
            key = cv2.waitKey(30) & 0xFF
            image_path = self.images[self.index]
            height, width = self.current.shape[:2]

            if key == ord("q"):
                break
            if key == ord("s"):
                self.confirm_current(image_path, width, height)
            elif key == ord("u"):
                if self.boxes:
                    self.boxes.pop()
                    self.draw()
            elif key == ord("d"):
                if self.delete_box_at(*self.mouse_pos):
                    print("Deleted box. Press s or n to save.")
            elif key == ord("n"):
                self.confirm_current(image_path, width, height)
                self.index = min(self.index + 1, len(self.images) - 1)
                self.open_current()
            elif key == ord("p"):
                self.confirm_current(image_path, width, height)
                self.index = max(self.index - 1, 0)
                self.open_current()

        cv2.destroyAllWindows()
        self.write_confirmed_output()
        return set(self.confirmed_images)


def main():
    parser = argparse.ArgumentParser(description="Simple one-class YOLO labeler for vehicles.")
    parser.add_argument("--images", default="datasets/parking_vehicles/raw", help="Directory with images to label.")
    parser.add_argument("--labels", default="datasets/parking_vehicles/raw_labels", help="Directory to save YOLO labels.")
    parser.add_argument("--start", type=int, default=1, help="1-based first image index to open.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to open.")
    parser.add_argument("--roi", default="parking_roi.json", help="Optional ROI JSON shown as a red polygon.")
    parser.add_argument("--max-width", type=int, default=int(os.getenv("EDITOR_MAX_WIDTH", "1500")))
    parser.add_argument("--max-height", type=int, default=int(os.getenv("EDITOR_MAX_HEIGHT", "850")))
    parser.add_argument("--confirmed-output", default=None, help="Write image names explicitly confirmed in this session.")
    args = parser.parse_args()

    Labeler(
        Path(args.images),
        Path(args.labels),
        start=args.start,
        limit=args.limit,
        roi_path=args.roi,
        max_width=args.max_width,
        max_height=args.max_height,
        confirmed_output=Path(args.confirmed_output) if args.confirmed_output else None,
    ).run()


if __name__ == "__main__":
    main()
