import argparse
from pathlib import Path

from parking_geometry import file_sha256, load_parking_regions, shape_from_image, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Write parking calibration metadata for startup validation.")
    parser.add_argument("--image", default="first_frame.jpg", help="Reference frame used to draw ROI and parking spaces.")
    parser.add_argument("--regions", default="bounding_boxes.json", help="Parking spaces JSON.")
    parser.add_argument("--roi", default="parking_roi.json", help="Parking ROI JSON.")
    parser.add_argument("--output", default="parking_calibration.json", help="Calibration metadata output.")
    parser.add_argument("--calibration-id", default=None, help="Human-readable camera/calibration identifier.")
    args = parser.parse_args()

    image_path = Path(args.image)
    regions_path = Path(args.regions)
    roi_path = Path(args.roi)
    shape = shape_from_image(image_path)
    if shape is None:
        raise SystemExit(f"Could not read reference image: {image_path}")
    height, width = shape
    regions = load_parking_regions(regions_path)
    calibration_id = args.calibration_id or f"{image_path.stem}_{width}x{height}"
    write_json(
        args.output,
        {
            "calibration_id": calibration_id,
            "reference": {
                "image": str(image_path),
                "size": {"width": width, "height": height},
                "sha256": file_sha256(image_path),
            },
            "regions_file": str(regions_path),
            "regions_sha256": file_sha256(regions_path),
            "roi_file": str(roi_path),
            "roi_sha256": file_sha256(roi_path),
            "expected_spaces": len(regions),
        },
    )
    print(f"Wrote {args.output} for {len(regions)} parking spaces at {width}x{height}.")


if __name__ == "__main__":
    main()
