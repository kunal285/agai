"""Exploratory Data Analysis (EDA) Script for DENTEX Dataset.

Analyzes:
  1. Class distribution & imbalance ratios across the 4 target classes.
  2. Panoramic image resolutions, aspect ratios, and color/channel properties.
  3. Bounding box scales, relative areas (% of image), and aspect ratios.
  4. Lesion co-occurrence per X-ray image.

Generates 4 publication-quality figures and an eda_summary.json.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from dentistry.config import load_config
from dentistry.dataset import parse_coco_annotations, validate_coco, DEFAULT_CLASSES, _normalise_category_name

# Use clean styling for plots
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

CLASS_COLORS = {
    "caries": "#e74c3c",           # Red
    "deep_caries": "#c0392b",      # Dark Red
    "periapical_lesion": "#e67e22", # Orange
    "impacted_tooth": "#2980b9"    # Blue
}

def run_eda(annotation_path: str | Path, image_root: str | Path, output_dir: str | Path):
    ann_path = Path(annotation_path).resolve()
    img_root = Path(image_root).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from: {ann_path}")
    parsed = parse_coco_annotations(ann_path, img_root)
    validation = validate_coco(parsed)

    # 1. Image metrics
    images_data = []
    for img in parsed["images"]:
        w, h = img.get("width"), img.get("height")
        images_data.append({
            "image_id": img["id"],
            "file_name": img["file_name"],
            "width": w,
            "height": h,
            "aspect_ratio": (w / h) if (w and h) else np.nan
        })
    df_images = pd.DataFrame(images_data)

    # 2. Annotation metrics
    img_lookup = {img["id"]: img for img in parsed["images"]}
    ann_data = []
    for ann in parsed["annotations"]:
        if not ann.get("bbox_valid", True) or not ann.get("bbox"):
            continue
        cname = _normalise_category_name(ann.get("category_name", ""))
        bx, by, bw, bh = map(float, ann["bbox"])
        img = img_lookup.get(ann["image_id"], {})
        iw, ih = img.get("width"), img.get("height")
        
        rel_area = (bw * bh) / (iw * ih) if (iw and ih) else np.nan
        aspect = bw / bh if bh > 0 else np.nan
        
        ann_data.append({
            "ann_id": ann["id"],
            "image_id": ann["image_id"],
            "category": cname,
            "x": bx, "y": by, "width": bw, "height": bh,
            "aspect_ratio": aspect,
            "relative_area_pct": rel_area * 100.0 if not np.isnan(rel_area) else np.nan
        })
    df_anns = pd.DataFrame(ann_data)

    print(f"Total valid images: {len(df_images)}")
    print(f"Total valid annotations: {len(df_anns)}")

    # Summary dictionary
    class_counts = df_anns["category"].value_counts().to_dict()
    total_anns = len(df_anns)
    imbalance_ratios = {c: round(class_counts.get(c, 0) / max(1, min(class_counts.values())), 2) for c in class_counts}

    summary = {
        "dataset_level": {
            "total_images": len(df_images),
            "total_annotations": total_anns,
            "categories_found": list(class_counts.keys())
        },
        "class_distribution": {
            c: {
                "count": int(cnt),
                "percentage": round((cnt / total_anns) * 100, 2) if total_anns else 0
            } for c, cnt in class_counts.items()
        },
        "imbalance_ratios": imbalance_ratios,
        "image_resolution_stats": {
            "width_mean": round(float(df_images["width"].mean()), 1),
            "width_min": int(df_images["width"].min()),
            "width_max": int(df_images["width"].max()),
            "height_mean": round(float(df_images["height"].mean()), 1),
            "height_min": int(df_images["height"].min()),
            "height_max": int(df_images["height"].max()),
            "aspect_ratio_mean": round(float(df_images["aspect_ratio"].mean()), 2)
        },
        "bbox_stats": {
            "mean_area_pct_of_image": round(float(df_anns["relative_area_pct"].mean()), 3),
            "median_area_pct_of_image": round(float(df_anns["relative_area_pct"].median()), 3),
            "mean_aspect_ratio": round(float(df_anns["aspect_ratio"].mean()), 2)
        }
    }

    # Save summary JSON
    summary_path = out / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved EDA summary metrics to {summary_path}")

    # ==================== FIGURE 1: Class Distribution ====================
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = list(class_counts.keys())
    counts = [class_counts[c] for c in cats]
    colors = [CLASS_COLORS.get(c, "#7f8c8d") for c in cats]

    bars = ax.bar(cats, counts, color=colors, edgecolor="black", linewidth=0.5, width=0.55)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_anns) * 100 if total_anns else 0
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{int(h)} ({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("DENTEX Abnormal Tooth Class Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Bounding Box Count", fontsize=11)
    ax.set_xlabel("Diagnostic Class", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(counts) * 1.18 if counts else 10)
    plt.tight_layout()
    fig1_path = out / "class_distribution.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved Figure 1: {fig1_path}")

    # ==================== FIGURE 2: Image Resolution Scatter ====================
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_images["width"], df_images["height"], color="#2c3e50", alpha=0.7, s=40, edgecolors="none")
    ax.axvline(df_images["width"].mean(), color="#e74c3c", linestyle="--", label=f"Mean W ({int(df_images['width'].mean())}px)")
    ax.axhline(df_images["height"].mean(), color="#3498db", linestyle="--", label=f"Mean H ({int(df_images['height'].mean())}px)")
    ax.set_title("Panoramic X-ray Resolution Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Width (pixels)", fontsize=11)
    ax.set_ylabel("Height (pixels)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig2_path = out / "image_resolution_scatter.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved Figure 2: {fig2_path}")

    # ==================== FIGURE 3: Bounding Box Size & Aspect Ratio ====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    # Relative area
    for c in cats:
        sub = df_anns[df_anns["category"] == c]
        if len(sub) > 0:
            ax1.hist(sub["relative_area_pct"], bins=15, alpha=0.6, label=c, color=CLASS_COLORS.get(c, None))
    ax1.set_title("BBox Scale (% of Total Image Area)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Relative Area (%)", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Aspect ratio
    for c in cats:
        sub = df_anns[df_anns["category"] == c]
        if len(sub) > 0:
            ax2.hist(sub["aspect_ratio"], bins=15, alpha=0.6, label=c, color=CLASS_COLORS.get(c, None))
    ax2.set_title("BBox Aspect Ratio (Width / Height)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Aspect Ratio", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig3_path = out / "bbox_aspect_area.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved Figure 3: {fig3_path}")

    # ==================== FIGURE 4: Sample Visualizations ====================
    sample_images = parsed["images"][:4]  # Take up to 4 images
    num_samples = len(sample_images)
    if num_samples > 0:
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3.5 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for i, img_info in enumerate(sample_images):
            ax = axes[i]
            img_path = img_info.get("path")
            if not img_path or not Path(img_path).exists():
                cand = img_root / img_info["file_name"]
                if cand.exists():
                    img_path = str(cand)

            if img_path and Path(img_path).exists():
                pil_img = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(pil_img)
                # Draw boxes for this image
                for ann in parsed["annotations"]:
                    if ann["image_id"] == img_info["id"] and ann.get("bbox_valid") and ann.get("bbox"):
                        cname = _normalise_category_name(ann.get("category_name", ""))
                        bx, by, bw, bh = map(float, ann["bbox"])
                        color = CLASS_COLORS.get(cname, "yellow")
                        draw.rectangle([bx, by, bx + bw, by + bh], outline=color, width=4)
                        draw.text((bx + 2, max(0, by - 14)), cname, fill=color)

                ax.imshow(pil_img)
                ax.set_title(f"Image ID: {img_info['id']} ({img_info['file_name']})", fontsize=10, fontweight="bold")
                ax.axis("off")
            else:
                ax.text(0.5, 0.5, f"Image not found: {img_info['file_name']}", ha="center", va="center")
                ax.axis("off")

        plt.tight_layout()
        fig4_path = out / "sample_annotated_xrays.png"
        plt.savefig(fig4_path, dpi=200)
        plt.close()
        print(f"Saved Figure 4: {fig4_path}")

    print("\n=== EDA Completed Successfully ===")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Run Exploratory Data Analysis on DENTEX annotations.")
    parser.add_argument("--annotations", required=True, help="Path to COCO JSON annotations")
    parser.add_argument("--image-root", required=True, help="Path to images directory")
    parser.add_argument("--output-dir", default="artifacts/eda", help="Output directory for EDA plots & summary")
    args = parser.parse_args()

    run_eda(args.annotations, args.image_root, args.output_dir)

if __name__ == "__main__":
    main()
