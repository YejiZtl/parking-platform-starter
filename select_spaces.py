import argparse
import copy
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv


WINDOW_NAME = "Parking Space Editor"


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def load_regions(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Parking JSON must contain a list: {path}")

    regions = []
    for index, item in enumerate(data, start=1):
        points = item.get("points") if isinstance(item, dict) else None
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"Invalid points in parking space {index}")
        regions.append(item)
    return regions


class ParkingSpaceEditor:
    def __init__(
        self,
        image_path: Path,
        json_path: Path,
        max_width: int,
        max_height: int,
    ) -> None:
        self.image_path = image_path
        self.json_path = json_path
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise FileNotFoundError(f"Could not open image: {image_path}")

        self.regions = load_regions(json_path)
        self.current_points: list[list[int]] = []
        self.history: list[tuple[list[dict], list[list[int]]]] = []
        self.mouse_point: tuple[int, int] | None = None
        self.hovered_index: int | None = None
        self.dirty = False

        height, width = self.image.shape[:2]
        self.scale = min(max_width / width, max_height / height, 1.0)
        self.display_size = (max(1, round(width * self.scale)), max(1, round(height * self.scale)))

    def remember(self) -> None:
        self.history.append((copy.deepcopy(self.regions), copy.deepcopy(self.current_points)))
        if len(self.history) > 1000:
            self.history.pop(0)

    def to_image_point(self, x: int, y: int) -> tuple[int, int]:
        height, width = self.image.shape[:2]
        image_x = min(width - 1, max(0, round(x / self.scale)))
        image_y = min(height - 1, max(0, round(y / self.scale)))
        return image_x, image_y

    def find_region(self, point: tuple[int, int]) -> int | None:
        nearest_index = None
        nearest_distance = float("-inf")
        tolerance = 12.0 / self.scale

        for index, region in enumerate(self.regions):
            polygon = np.asarray(region["points"], dtype=np.float32)
            distance = cv2.pointPolygonTest(polygon, point, True)
            if distance >= 0:
                return index
            if distance >= -tolerance and distance > nearest_distance:
                nearest_index = index
                nearest_distance = distance
        return nearest_index

    def add_point(self, point: tuple[int, int]) -> None:
        self.remember()
        self.current_points.append([point[0], point[1]])
        if len(self.current_points) == 4:
            self.regions.append({"points": self.current_points})
            self.current_points = []
            self.dirty = True
            print(f"Added space {len(self.regions)}")

    def delete_hovered(self) -> None:
        if self.hovered_index is None:
            return
        self.remember()
        deleted = self.hovered_index + 1
        self.regions.pop(self.hovered_index)
        self.hovered_index = None
        self.dirty = True
        print(f"Deleted space {deleted}; {len(self.regions)} spaces remain")

    def undo(self) -> None:
        if not self.history:
            print("Nothing to undo")
            return
        self.regions, self.current_points = self.history.pop()
        self.dirty = True
        print(f"Undo complete; {len(self.regions)} spaces")

    def backup_path(self) -> Path:
        backup_dir = self.json_path.parent / "runs" / "parking_train" / "backups" / "parking_spaces"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return backup_dir / f"{self.json_path.stem}_{stamp}{self.json_path.suffix}"

    def save(self) -> None:
        if not self.dirty:
            return

        if self.json_path.exists():
            backup = self.backup_path()
            shutil.copy2(self.json_path, backup)
            print(f"Backup: {backup}")

        temp_path = self.json_path.with_name(f".{self.json_path.name}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(self.regions, file, ensure_ascii=False, indent=4)
            file.write("\n")
        os.replace(temp_path, self.json_path)
        self.dirty = False
        print(f"Saved {len(self.regions)} spaces to {self.json_path}")

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        del flags, param
        self.mouse_point = self.to_image_point(x, y)
        self.hovered_index = self.find_region(self.mouse_point)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(self.mouse_point)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.delete_hovered()

    def render(self) -> np.ndarray:
        canvas = self.image.copy()
        for index, region in enumerate(self.regions):
            polygon = np.asarray(region["points"], dtype=np.int32)
            color = (0, 0, 255) if index == self.hovered_index else (0, 220, 0)
            thickness = 5 if index == self.hovered_index else 3
            cv2.polylines(canvas, [polygon], True, color, thickness, cv2.LINE_AA)

        if self.current_points:
            points = np.asarray(self.current_points, dtype=np.int32)
            if len(points) > 1:
                cv2.polylines(canvas, [points], False, (0, 220, 255), 3, cv2.LINE_AA)
            for point in points:
                cv2.circle(canvas, tuple(point), 7, (0, 220, 255), -1, cv2.LINE_AA)

        if self.scale < 1.0:
            canvas = cv2.resize(canvas, self.display_size, interpolation=cv2.INTER_AREA)
        return canvas

    def update_title(self) -> None:
        state = "UNSAVED" if self.dirty else "SAVED"
        title = f"{WINDOW_NAME} | {len(self.regions)} spaces | {state}"
        try:
            cv2.setWindowTitle(WINDOW_NAME, title)
        except cv2.error:
            pass

    def run(self) -> None:
        print(f"Image: {self.image_path.resolve()}")
        print(f"Loaded {len(self.regions)} existing spaces")
        print("Left click: add 4 corners | Right click or D: delete under cursor")
        print("Z: undo | S: save | Q/Esc: save and exit")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        try:
            while True:
                cv2.imshow(WINDOW_NAME, self.render())
                self.update_title()
                key = cv2.waitKey(30)

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("d"), ord("D"), 3014656):
                    self.delete_hovered()
                elif key in (ord("z"), ord("Z"), 26):
                    self.undo()
                elif key in (ord("s"), ord("S"), 19):
                    self.save()
                elif key & 0xFF in (ord("q"), 27):
                    break
        except KeyboardInterrupt:
            print("Stopping editor")
        finally:
            self.save()
            if self.current_points:
                print(f"Ignored {len(self.current_points)} unfinished corner(s)")
            cv2.destroyAllWindows()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Continue editing parking spaces on a captured frame.")
    parser.add_argument("--image", default="first_frame.jpg", help="Still frame from the target camera.")
    parser.add_argument("--json", default=os.getenv("PARKING_JSON", "bounding_boxes.json"), help="Parking JSON file.")
    parser.add_argument("--max-width", type=int, default=env_int("EDITOR_MAX_WIDTH", 1500))
    parser.add_argument("--max-height", type=int, default=env_int("EDITOR_MAX_HEIGHT", 850))
    args = parser.parse_args()

    editor = ParkingSpaceEditor(
        image_path=Path(args.image),
        json_path=Path(args.json),
        max_width=args.max_width,
        max_height=args.max_height,
    )
    editor.run()


if __name__ == "__main__":
    main()
