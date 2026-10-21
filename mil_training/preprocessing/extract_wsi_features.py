"""
Tissue segmentation + 20x/256px tiles + UNI v1 features (Trident).

Example:

    python extract_wsi_features.py \\
      --slides_csv /path/to/slides.csv \\
      --output_dir /path/to/output \\
      --path_column path \\
      --encoder_name uni_v1 \\
      --seg_model grandqc_artifact

Requires HUGGINGFACE_HUB_TOKEN. The slides CSV must contain a column of WSI paths.
"""
import argparse
import csv
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv


@contextmanager
def complete_stderr_suppression():
    """Redirect stderr to a temp file (suppresses noisy TIFF warnings)."""
    import tempfile

    original_stderr = os.dup(2)
    try:
        with tempfile.TemporaryFile() as temp_stderr:
            os.dup2(temp_stderr.fileno(), 2)
            yield
    finally:
        os.dup2(original_stderr, 2)
        os.close(original_stderr)


def setup_logging(output_dir: str) -> None:
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )


def process_slide(
    sample_svs_file: str,
    output_path: str,
    encoder,
    *,
    device: str,
    seg_model: str,
    confidence_thresh: float,
    target_mag: int,
    patch_size: int,
    min_tissue_proportion: float,
    reader_type: str,
    encoder_name: str,
    batch_limit: int,
) -> dict:
    from trident import load_wsi
    from trident.segmentation_models import segmentation_model_factory
    results = {
        "file_path": sample_svs_file,
        "step1_status": "failed",
        "step2_status": "failed",
        "step3_status": "failed",
        "step1_time": 0.0,
        "step2_time": 0.0,
        "step3_time": 0.0,
    }

    logging.info(f"Processing {sample_svs_file}...")

    try:
        slide_name = os.path.splitext(os.path.basename(sample_svs_file))[0]
        tissue_seg_path = os.path.join(
            output_path, "contours_geojson", f"{slide_name}.geojson"
        )
        contours_image_path = os.path.join(
            output_path, "contours", f"{slide_name}.jpg"
        )

        segmentation_exists = os.path.exists(tissue_seg_path) and os.path.exists(
            contours_image_path
        )

        if segmentation_exists:
            logging.info(
                f"Existing segmentation found for {slide_name}: "
                "Loading existing segmentation results."
            )
            with complete_stderr_suppression():
                slide = load_wsi(
                    sample_svs_file,
                    reader_type=reader_type,
                    tissue_seg_path=tissue_seg_path,
                )
        else:
            logging.info(f"No existing segmentation found for {slide_name}.")
            with complete_stderr_suppression():
                slide = load_wsi(sample_svs_file, reader_type=reader_type)

        # Step 1: Tissue Segmentation
        start_time = time.time()
        if segmentation_exists:
            results["step1_status"] = "skipped"
            results["step1_time"] = 0.0
        else:
            logging.info("Running tissue segmentation...")
            try:
                segmentation_model = segmentation_model_factory(
                    model_name=seg_model,
                    confidence_thresh=confidence_thresh,
                )
                with complete_stderr_suppression():
                    slide.segment_tissue(
                        segmentation_model=segmentation_model,
                        target_mag=segmentation_model.target_mag,
                        job_dir=output_path,
                        device=device,
                    )
                results["step1_status"] = "success"
                logging.info(
                    f"Tissue segmentation completed. Results saved to {output_path}"
                )
            except Exception as e:
                logging.error(f"Error in step 1: {str(e)}")
            results["step1_time"] = round(time.time() - start_time, 2)

        # Step 2: Extract and visualize tissue coordinates
        start_time = time.time()
        save_coords = os.path.join(
            output_path, f"{target_mag}x_{patch_size}px_0px_overlap"
        )
        coords_file = os.path.join(
            save_coords, "patches", f"{slide.name}_patches.h5"
        )
        coords_exists = os.path.exists(coords_file)

        if coords_exists:
            logging.info(
                "Tissue coordinates already exist. Skipping coordinate extraction."
            )
            results["step2_status"] = "skipped"
            results["step2_time"] = 0.0
        else:
            logging.info("Extracting tissue coordinates...")
            try:
                coords_path = slide.extract_tissue_coords(
                    target_mag=target_mag,
                    patch_size=patch_size,
                    min_tissue_proportion=min_tissue_proportion,
                    save_coords=save_coords,
                )
                logging.info(
                    f"Tissue coordinates extracted and saved to {coords_path}."
                )

                viz_coords_path = slide.visualize_coords(
                    coords_path=coords_path,
                    save_patch_viz=os.path.join(save_coords, "visualization"),
                )
                logging.info(
                    f"Tissue coordinates visualization saved to {viz_coords_path}."
                )
                results["step2_status"] = "success"
            except Exception as e:
                logging.error(f"Error in step 2: {str(e)}")
            results["step2_time"] = round(time.time() - start_time, 2)

        # Step 3: Feature Extraction
        start_time = time.time()
        features_dir = os.path.join(save_coords, f"features_{encoder_name}")
        features_file = os.path.join(features_dir, f"{slide.name}.h5")
        features_exists = os.path.exists(features_file)

        if features_exists:
            logging.info("Features already exist. Skipping feature extraction.")
            results["step3_status"] = "skipped"
            results["step3_time"] = 0.0
        else:
            logging.info("Extracting features from patches...")
            try:
                with complete_stderr_suppression():
                    slide.extract_patch_features(
                        patch_encoder=encoder,
                        coords_path=os.path.join(
                            save_coords, "patches", f"{slide.name}_patches.h5"
                        ),
                        save_features=features_dir,
                        device=device,
                        batch_limit=batch_limit,
                    )
                logging.info(
                    f"Feature extraction completed. Results saved to {features_dir}"
                )
                results["step3_status"] = "success"
            except Exception as e:
                logging.error(f"Error in step 3: {str(e)}")
            results["step3_time"] = round(time.time() - start_time, 2)

    except Exception as e:
        logging.error(f"Error loading slide: {str(e)}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run tissue segmentation, patch coordinate extraction, and patch "
            "feature encoding over a list of WSIs from a CSV."
        )
    )
    parser.add_argument(
        "--slides_csv",
        type=str,
        required=True,
        help="Path to CSV listing WSI paths (e.g. /path/to/slides.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for segmentation outputs, features, logs, and results CSV.",
    )
    parser.add_argument(
        "--path_column",
        type=str,
        default="path",
        help="CSV column containing WSI file paths (default: path).",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Inclusive start row index into the slides CSV (default: 0).",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Exclusive end row index into the slides CSV (default: all rows).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for segmentation and encoding (default: cuda:0).",
    )
    parser.add_argument(
        "--seg_model",
        type=str,
        default="grandqc_artifact",
        help="Trident segmentation model name (default: grandqc_artifact).",
    )
    parser.add_argument(
        "--confidence_thresh",
        type=float,
        default=0.5,
        help="Segmentation confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default="uni_v1",
        help="Trident patch encoder name (default: uni_v1).",
    )
    parser.add_argument(
        "--batch_limit",
        type=int,
        default=512,
        help="Batch size for feature extraction (default: 512).",
    )
    parser.add_argument(
        "--target_mag",
        type=int,
        default=20,
        help="Target magnification for patch coords (default: 20).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size in pixels (default: 256).",
    )
    parser.add_argument(
        "--min_tissue_proportion",
        type=float,
        default=0.2,
        help="Minimum tissue proportion per patch (default: 0.2).",
    )
    parser.add_argument(
        "--reader_type",
        type=str,
        default="openslide",
        help="WSI reader backend (default: openslide).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["TIFF_NOCLEANUP"] = "1"
    os.environ["LIBTIFF_SUPPRESS_WARNINGS"] = "1"
    load_dotenv()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_HUB_TOKEN not found in environment variables")
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    from huggingface_hub import login
    from trident.patch_encoder_models import encoder_factory

    login(token=token)

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    df = pd.read_csv(args.slides_csv)
    if args.path_column not in df.columns:
        raise ValueError(
            f"Column '{args.path_column}' not found in {args.slides_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    slice_end = args.end_idx if args.end_idx is not None else len(df)
    df = df.iloc[args.start_idx:slice_end]
    sample_svs_files = (
        df[args.path_column].dropna().replace("", pd.NA).dropna().to_list()
    )

    if not sample_svs_files:
        raise ValueError(
            f"No slide paths found in column '{args.path_column}' "
            f"(rows [{args.start_idx}:{slice_end}])."
        )

    logging.info(f"Initializing encoder ({args.encoder_name})...")
    encoder = encoder_factory(model_name=args.encoder_name)
    encoder.eval()
    encoder.to(args.device)
    logging.info("Encoder initialized successfully")

    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(results_dir, f"processing_results_{timestamp}.csv")

    logging.info(
        f"Processing {len(sample_svs_files)} slides "
        f"(CSV rows [{args.start_idx}:{slice_end}]) -> {args.output_dir}"
    )

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_path",
                "step1_status",
                "step2_status",
                "step3_status",
                "step1_time",
                "step2_time",
                "step3_time",
            ],
        )
        writer.writeheader()

        for i, sample_svs_file in enumerate(sample_svs_files):
            logging.info(
                f"Processing slide {i + 1} of {len(sample_svs_files)}..."
            )
            results = process_slide(
                sample_svs_file,
                args.output_dir,
                encoder,
                device=args.device,
                seg_model=args.seg_model,
                confidence_thresh=args.confidence_thresh,
                target_mag=args.target_mag,
                patch_size=args.patch_size,
                min_tissue_proportion=args.min_tissue_proportion,
                reader_type=args.reader_type,
                encoder_name=args.encoder_name,
                batch_limit=args.batch_limit,
            )
            writer.writerow(results)

    logging.info(f"Processing completed. Results saved to {csv_file}")


if __name__ == "__main__":
    main()
