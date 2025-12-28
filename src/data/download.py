"""
Download original MSCOCO images for the Defacto inpainting dataset.
Reads MSCOCO IDs from JSON files and downloads originals from COCO API.

Usage:
    python download_mscoco_originals.py
"""

import os
import json
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Paths
PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / "Dataset"
GRAPH_DIR = DATASET_DIR / "inpainting_annotations" / "graph"
OUTPUT_DIR = DATASET_DIR / "original_images"

# COCO image URL template
# train2017: http://images.cocodataset.org/train2017/000000001900.jpg
# val2017: http://images.cocodataset.org/val2017/000000001900.jpg
COCO_URL_TEMPLATE = "http://images.cocodataset.org/{split}/{image_id}.jpg"


def extract_coco_info_from_json(json_path):
    """Extract MSCOCO information from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for entry in data:
            if entry.get("Name") == "Ground Truth":
                path = entry.get("Property", {}).get("Path", "")
                # Path format: "MSCOCO/images/train2017/000000001900.jpg"
                if "MSCOCO" in path:
                    parts = path.split("/")
                    if len(parts) >= 4:
                        split = parts[2]  # train2017 or val2017
                        image_name = parts[3]  # 000000001900.jpg
                        image_id = image_name.replace(".jpg", "").replace(".tif", "")
                        return {
                            "split": split,
                            "image_id": image_id,
                            "json_file": json_path.name
                        }
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
    return None


def download_image(info, output_dir):
    """Download image from COCO dataset."""
    if info is None:
        return None
    
    url = COCO_URL_TEMPLATE.format(split=info["split"], image_id=info["image_id"])
    output_path = output_dir / f"{info['image_id']}.jpg"
    
    if output_path.exists():
        return f"Already exists: {info['image_id']}"
    
    try:
        urllib.request.urlretrieve(url, output_path)
        return f"Downloaded: {info['image_id']}"
    except Exception as e:
        return f"Failed {info['image_id']}: {e}"


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # List JSON files
    json_files = list(GRAPH_DIR.glob("*.json"))
    total_files = len(json_files)
    print(f"Found {total_files} JSON files.")
    
    # Extract MSCOCO info
    print("Extracting MSCOCO IDs...")
    coco_infos = []
    for json_file in json_files:
        info = extract_coco_info_from_json(json_file)
        if info:
            coco_infos.append(info)
    
    print(f"Found {len(coco_infos)} MSCOCO images.")
    
    # Find unique images (same image may be used multiple times)
    unique_infos = {}
    for info in coco_infos:
        key = f"{info['split']}_{info['image_id']}"
        if key not in unique_infos:
            unique_infos[key] = info
    
    print(f"Unique images: {len(unique_infos)}")
    
    # Download images
    print("\nDownloading images...")
    downloaded = 0
    failed = 0
    skipped = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(download_image, info, OUTPUT_DIR): info 
            for info in unique_infos.values()
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                if "Downloaded" in result:
                    downloaded += 1
                elif "Already exists" in result:
                    skipped += 1
                else:
                    failed += 1
            
            if i % 100 == 0:
                print(f"Progress: {i}/{len(unique_infos)} | Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed}")
    
    print(f"\n=== COMPLETE ===")
    print(f"Downloaded: {downloaded}")
    print(f"Already existed: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
