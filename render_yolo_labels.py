import argparse
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
BOX_COLOR = (0, 210, 80)
TEXT_COLOR = (255, 255, 255)


def read_yolo_boxes(label_path: Path, width: int, height: int):
    boxes = []
    if not label_path.exists():
        return boxes

    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split()
        if len(parts) != 5:
            print(f"Skip invalid label {label_path}:{line_number}")
            continue

        class_id, cx, cy, box_width, box_height = map(float, parts)
        x1 = round((cx - box_width / 2) * width)
        y1 = round((cy - box_height / 2) * height)
        x2 = round((cx + box_width / 2) * width)
        y2 = round((cy + box_height / 2) * height)
        boxes.append(
            (
                int(class_id),
                max(0, min(width - 1, x1)),
                max(0, min(height - 1, y1)),
                max(0, min(width - 1, x2)),
                max(0, min(height - 1, y2)),
            )
        )
    return boxes


def draw_label(image, text: str, x: int, y: int, scale: float, thickness: int):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    top = max(0, y - text_height - baseline - 8)
    cv2.rectangle(
        image,
        (x, top),
        (x + text_width + 10, top + text_height + baseline + 8),
        BOX_COLOR,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 5, top + text_height + 3),
        font,
        scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render saved YOLO labels onto image copies."
    )
    parser.add_argument("--images", default="datasets/parking_vehicles/raw")
    parser.add_argument("--labels", default="datasets/parking_vehicles/raw_labels")
    parser.add_argument(
        "--output", default="datasets/parking_vehicles/labeled_images"
    )
    parser.add_argument("--class-name", default="vehicle")
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Show the class name above each box. Hidden by default for dense scenes.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    total_boxes = 0
    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        height, width = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes = read_yolo_boxes(label_path, width, height)
        total_boxes += len(boxes)
        line_width = max(2, round(min(width, height) / 700))
        text_scale = max(0.55, min(width, height) / 1800)

        for class_id, x1, y1, x2, y2 in boxes:
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                BOX_COLOR,
                line_width,
                cv2.LINE_AA,
            )
            if args.show_text:
                draw_label(
                    image,
                    f"{args.class_name} {class_id}",
                    x1,
                    y1,
                    text_scale,
                    max(1, line_width - 1),
                )

        output_path = output_dir / image_path.name
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            cv2.imwrite(
                str(output_path),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, args.jpeg_quality))],
            )
        else:
            cv2.imwrite(str(output_path), image)

        if index % 20 == 0 or index == len(images):
            print(f"Rendered {index}/{len(images)} images, boxes: {total_boxes}")

    print(f"Done. Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
