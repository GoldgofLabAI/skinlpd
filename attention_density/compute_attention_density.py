import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import h5py
import argparse
import time
import csv
import os
import pandas as pd
import logging
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon, Point, mapping
from shapely.ops import unary_union
from shapely.prepared import prep

# Try to import rasterio, fall back to alternative if not available
try:
    from rasterio import features
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description='Compute nuclear density in high-attention tiles')
    parser.add_argument('--attention-threshold', type=float, default=80.0,
                        help='Percentile threshold for attention values (default: 80.0)')
    parser.add_argument('--display-cells', action='store_true',
                        help='Display nuclei centroids in high-attention zones')
    parser.add_argument('--display-coverage', action='store_true',
                        help='Display tissue coverage ratios on patches')
    parser.add_argument('--window-x', type=float, default=None,
                        help='X coordinate of window of interest (pixels)')
    parser.add_argument('--window-y', type=float, default=None,
                        help='Y coordinate of window of interest (pixels)')
    parser.add_argument('--window-width', type=float, default=None,
                        help='Width of window of interest (pixels)')
    parser.add_argument('--window-height', type=float, default=None,
                        help='Height of window of interest (pixels)')
    parser.add_argument('--cell-type', type=str, default=None,
                        help='Comma-separated cell types to filter (case-insensitive, e.g., "Neoplastic,Inflammatory")')
    parser.add_argument('--base-filename', type=str, default=None,
                        help='Slide stem for single-slide mode')
    parser.add_argument('--cell-sampling-percentage', type=float, default=100.0,
                        help='Percentage of cells to sample for display (default: 1.0)')
    parser.add_argument('--filter-cells-by-segmentation', action='store_true',
                        help='Filter cells to only those within the segmentation polygon (more accurate but slower)')
    parser.add_argument('--output-analysis', type=str, default=None,
                        help='Output directory for analysis results CSV (required for batch processing)')
    parser.add_argument('--visualization', action='store_true',
                        help='Save visualization PNGs (saved in output_analysis/visualization/ if output-analysis is set)')
    parser.add_argument('--process-batch', action='store_true',
                        help='Process multiple cases in batch mode')
    parser.add_argument('--input-csv', type=str, default=None,
                        help='Input CSV file with base_file_name column for batch processing (if not provided, processes all available cases)')
    parser.add_argument('--geojson-dir', type=str, required=True,
                        help='Directory of tissue-contour GeoJSON files named {slide}.geojson')
    parser.add_argument('--attention-dir', type=str, required=True,
                        help='Directory of attention H5 files named {slide}.h5')
    parser.add_argument('--nuclei-dir', type=str, required=True,
                        help='Directory of CellViT outputs, each slide in a subdirectory with cell_detection.json.snappy')
    
    args = parser.parse_args()
    configure_data_roots(args.geojson_dir, args.attention_dir, args.nuclei_dir)
    
    # Validate batch processing requirements
    if args.process_batch and args.output_analysis is None:
        parser.error("--output-analysis is required when using --process-batch")
    
    # Validate window parameters (all or none must be provided)
    window_params = [args.window_x, args.window_y, args.window_width, args.window_height]
    if any(p is not None for p in window_params) and not all(p is not None for p in window_params):
        parser.error("--window-x, --window-y, --window-width, and --window-height must all be provided together, or all omitted")
    
    return args


DATA_ROOTS = {"geojson_dir": None, "attention_dir": None, "nuclei_dir": None}


def configure_data_roots(geojson_dir, attention_dir, nuclei_dir):
    DATA_ROOTS["geojson_dir"] = geojson_dir
    DATA_ROOTS["attention_dir"] = attention_dir
    DATA_ROOTS["nuclei_dir"] = nuclei_dir


def get_file_paths(base_filename):
    """Construct file paths from CLI data roots."""
    geojson_dir = DATA_ROOTS["geojson_dir"]
    attention_dir = DATA_ROOTS["attention_dir"]
    nuclei_dir = DATA_ROOTS["nuclei_dir"]
    if not geojson_dir or not attention_dir or not nuclei_dir:
        raise ValueError("Call configure_data_roots() or pass --geojson-dir/--attention-dir/--nuclei-dir")
    nuclei_slide = Path(nuclei_dir) / base_filename
    snappy_path = nuclei_slide / "cell_detection.json.snappy"
    json_path = nuclei_slide / "cell_detection.json"
    cell_detection = str(snappy_path if snappy_path.exists() else json_path)
    return {
        "geo_json": str(Path(geojson_dir) / f"{base_filename}.geojson"),
        "attention_h5": str(Path(attention_dir) / f"{base_filename}.h5"),
        "cell_detection": cell_detection,
    }


def load_geojson(geo_json_path):
    """Load and validate GeoJSON file."""
    with open(geo_json_path, 'r') as f:
        geo_json = json.load(f)
    
    features = geo_json.get('features', [])
    if not features:
        raise ValueError("No features found in GeoJSON")
    
    return geo_json, features


def load_attention_data(attention_h5_path, attention_threshold, window_params=None, logger=None):
    """Load attention data from H5 file and filter by threshold."""
    if logger is None:
        logger = logging.getLogger('compute_attention_density')
    
    try:
        with h5py.File(attention_h5_path, 'r') as f:
            attention = f['attention'][:]
            coords = f['coords'][:]
            if 'patch_size_level0' in f['coords'].attrs:
                patch_size_level0 = f['coords'].attrs['patch_size_level0']
            elif 'patch_size_level0' in f:
                patch_size_level0 = int(np.array(f['patch_size_level0']).reshape(-1)[0])
            else:
                patch_size_level0 = 256
        
        logger.info(f"Loaded {len(attention)} attention scores")
        logger.info(f"Attention range: [{attention.min():.6e}, {attention.max():.6e}]")
        logger.info(f"Patch size (level0): {patch_size_level0}")
        logger.info(f"Coordinate range: X=[{coords[:, 0].min()}, {coords[:, 0].max()}], "
              f"Y=[{coords[:, 1].min()}, {coords[:, 1].max()}]")
        
        # Calculate threshold (percentile)
        threshold_value = np.percentile(attention, attention_threshold)
        
        # Filter high-attention patches
        high_attention_mask = attention > threshold_value
        high_attention_coords = coords[high_attention_mask]
        high_attention_values = attention[high_attention_mask]
        
        logger.info(f"Attention threshold ({attention_threshold}th percentile): {threshold_value:.6e}")
        logger.info(f"Number of high-attention patches: {len(high_attention_coords)} ({len(high_attention_coords)/len(attention)*100:.1f}%)")
        
        # Initialize all_coords and all_attention (used for density calculations)
        all_coords = coords
        all_attention = attention
        
        # Filter by window if specified (for both visualization and calculations)
        if window_params is not None:
            # Filter all patches for density calculations
            all_coords, all_attention = filter_patches_by_window(
                coords, attention, patch_size_level0, window_params
            )
            # Filter high-attention patches for visualization
            high_attention_coords, high_attention_values = filter_patches_by_window(
                high_attention_coords, high_attention_values, patch_size_level0, window_params
            )
            print(f"Window filtering applied to all patches for density calculations")
        
        return {
            'all_attention': all_attention,
            'all_coords': all_coords,
            'high_attention_coords': high_attention_coords,
            'high_attention_values': high_attention_values,
            'patch_size_level0': patch_size_level0
        }
        
    except FileNotFoundError:
        print(f"Warning: Attention file not found at {attention_h5_path}")
        return None
    except Exception as e:
        print(f"Error reading attention file: {e}")
        return None


def filter_patches_by_window(coords, values, patch_size, window_params):
    """Filter patches to only those overlapping with the specified window."""
    window_x, window_y, window_width, window_height = window_params
    window_x_end = window_x + window_width
    window_y_end = window_y + window_height
    
    window_mask = np.zeros(len(coords), dtype=bool)
    for i, coord in enumerate(coords):
        x, y = coord[0], coord[1]
        patch_x_end = x + patch_size
        patch_y_end = y + patch_size
        # Check if patch overlaps with window
        if not (patch_x_end < window_x or x > window_x_end or 
                patch_y_end < window_y or y > window_y_end):
            window_mask[i] = True
    
    filtered_coords = coords[window_mask]
    if values is not None:
        filtered_values = values[window_mask]
    else:
        filtered_values = None
    
    return filtered_coords, filtered_values


def load_cell_detection_data(cell_detection_path, cell_type_filter=None):
    """Load cell detection data from snappy-compressed JSON or plain JSON."""
    try:
        path = Path(cell_detection_path)
        if str(path).endswith(".snappy"):
            import snappy
            with open(path, "rb") as f:
                decompressed = snappy.decompress(f.read())
            cell_data = json.loads(decompressed.decode("utf-8"))
        else:
            cell_data = json.loads(path.read_text(encoding="utf-8"))
        
        cells = cell_data.get('cells', [])
        type_map = cell_data.get('type_map', {})
        wsi_metadata = cell_data.get('wsi_metadata', {})
        base_mpp = wsi_metadata.get('base_mpp', None)
        
        # Filter by cell type if specified
        if cell_type_filter is not None:
            cells, type_map = filter_cells_by_type(cells, type_map, cell_type_filter)
        
        return {
            'cells': cells,
            'type_map': type_map,
            'base_mpp': base_mpp,
            'wsi_metadata': wsi_metadata
        }
        
    except FileNotFoundError:
        print(f"Warning: Cell detection file not found at {cell_detection_path}")
        return None
    except Exception as e:
        print(f"Error reading cell detection file: {e}")
        import traceback
        traceback.print_exc()
        return None


def filter_cells_by_type(cells, type_map, cell_type_filter):
    """Filter cells by specified cell types (case-insensitive)."""
    requested_types = [t.strip().lower() for t in cell_type_filter.split(',')]
    
    # Create reverse mapping: type name (lowercase) -> type ID
    type_name_to_id = {}
    for type_id, type_name in type_map.items():
        type_name_to_id[type_name.lower()] = int(type_id)
    
    # Find matching type IDs
    matching_type_ids = set()
    for req_type in requested_types:
        if req_type in type_name_to_id:
            matching_type_ids.add(type_name_to_id[req_type])
        else:
            print(f"Warning: Cell type '{req_type}' not found in type_map. Available types: {list(type_map.values())}")
    
    if matching_type_ids:
        filtered_cells = [cell for cell in cells if cell.get('type') in matching_type_ids]
        type_names = [type_map[str(tid)] for tid in sorted(matching_type_ids)]
        print(f"Filtered to {len(filtered_cells)} cells of type(s): {', '.join(type_names)}")
        return filtered_cells, type_map
    else:
        print("Warning: No valid cell types found. Using all cells.")
        return cells, type_map


def extract_segmentation_polygons(features, logger=None):
    """Extract and build Shapely polygons from GeoJSON features."""
    if logger is None:
        logger = logging.getLogger('compute_attention_density')
    
    segmentation_polygons = []
    
    for feature in features:
        geometry = feature.get('geometry', {})
        geom_type = geometry.get('type', '')
        coords = geometry.get('coordinates', [])
        
        if geom_type == 'Polygon':
            if coords and len(coords) > 0:
                ring_coords = coords[0]
                if ring_coords[0] != ring_coords[-1]:
                    ring_coords = ring_coords + [ring_coords[0]]
                try:
                    poly = Polygon(ring_coords)
                    if poly.is_valid:
                        segmentation_polygons.append(poly)
                    else:
                        poly = poly.buffer(0)
                        if poly.is_valid:
                            segmentation_polygons.append(poly)
                except Exception as e:
                    print(f"Warning: Could not create polygon from feature: {e}")
        elif geom_type == 'MultiPolygon':
            for polygon in coords:
                if polygon and len(polygon) > 0:
                    ring_coords = polygon[0]
                    if ring_coords[0] != ring_coords[-1]:
                        ring_coords = ring_coords + [ring_coords[0]]
                    try:
                        poly = Polygon(ring_coords)
                        if poly.is_valid:
                            segmentation_polygons.append(poly)
                        else:
                            poly = poly.buffer(0)
                            if poly.is_valid:
                                segmentation_polygons.append(poly)
                    except Exception as e:
                        print(f"Warning: Could not create polygon from MultiPolygon: {e}")
    
    if segmentation_polygons:
        if len(segmentation_polygons) == 1:
            segmentation_union = segmentation_polygons[0]
        else:
            segmentation_union = unary_union(segmentation_polygons)
        logger.info(f"Building segmentation polygon(s) for coverage calculation...")
        logger.info(f"Created segmentation polygon(s) with {len(segmentation_polygons)} feature(s)")
        return segmentation_union
    else:
        logger.warning("No valid segmentation polygons created")
        return None


def create_patch_polygon(x, y, patch_size):
    """Create a Shapely polygon for a patch rectangle."""
    return Polygon([
        (x, y),
        (x + patch_size, y),
        (x + patch_size, y + patch_size),
        (x, y + patch_size),
        (x, y)
    ])


def compute_tissue_coverage(high_attention_coords, patch_size, segmentation_union, ax):
    """Compute and display tissue coverage ratios for attention patches."""
    patch_area = patch_size * patch_size
    
    for coord in high_attention_coords:
        x, y = coord[0], coord[1]
        patch_poly = create_patch_polygon(x, y, patch_size)
        
        try:
            intersection = patch_poly.intersection(segmentation_union)
            if intersection.is_empty:
                coverage_ratio = 0.0
            else:
                intersection_area = intersection.area
                coverage_ratio = intersection_area / patch_area
        except Exception:
            coverage_ratio = 0.0
        
        # Display coverage ratio in the center of the patch
        center_x = x + patch_size / 2
        center_y = y + patch_size / 2
        ax.text(center_x, center_y, f'{coverage_ratio:.1f}', 
                fontsize=6, ha='center', va='center', 
                color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5, edgecolor='none'))


def filter_cells_by_segmentation(all_cells, segmentation_union):
    """Filter cells to only those within the segmentation polygon (global-level filtering)."""
    if segmentation_union is None:
        print("Warning: Cannot filter cells - segmentation polygon not available")
        return all_cells, 0.0
    
    print(f"\nFiltering {len(all_cells)} cells by segmentation polygon...")
    start_time = time.time()
    
    # Convert cell centroids to Shapely Points and check containment
    filtered_cells = []
    for cell in all_cells:
        centroid = cell['centroid']
        point = Point(centroid[0], centroid[1])
        if segmentation_union.contains(point):
            filtered_cells.append(cell)
    
    elapsed_time = time.time() - start_time
    print(f"Filtered to {len(filtered_cells)} cells within segmentation ({len(filtered_cells)/len(all_cells)*100:.1f}%)")
    print(f"Filtering time: {elapsed_time:.2f} seconds")
    
    return filtered_cells, elapsed_time


def compute_cell_density_metrics(all_cells, all_attention, all_coords, high_attention_coords,
                                  patch_size, segmentation_union, base_mpp, attention_threshold,
                                  window_params=None, filter_by_segmentation=False, logger=None):
    """Compute cell density metrics using two methods."""
    if logger is None:
        logger = logging.getLogger('compute_attention_density')
    
    if not all_cells or base_mpp is None:
        logger.warning("Skipping density calculation - missing required data")
        return None
    
    computation_start_time = time.time()
    
    logger.info(f"{'='*60}")
    logger.info("COMPUTING CELL DENSITY METRICS")
    logger.info(f"{'='*60}")
    
    # Filter cells by segmentation if requested
    if filter_by_segmentation:
        all_cells, filter_time = filter_cells_by_segmentation(all_cells, segmentation_union)
    else:
        filter_time = 0.0
    
    # Convert all cells to numpy array
    all_centroids = np.array([cell['centroid'] for cell in all_cells])
    
    # Filter cells by window if specified
    if window_params is not None:
        window_x, window_y, window_width, window_height = window_params
        window_x_end = window_x + window_width
        window_y_end = window_y + window_height
        window_cell_mask = (
            (all_centroids[:, 0] >= window_x) & (all_centroids[:, 0] < window_x_end) &
            (all_centroids[:, 1] >= window_y) & (all_centroids[:, 1] < window_y_end)
        )
        all_centroids = all_centroids[window_cell_mask]
    
    # Conversion factor: pixels^2 to mm^2
    pixels_to_mm2 = (base_mpp / 1000.0) ** 2
    logger.info(f"Conversion factor: {pixels_to_mm2:.6e} mm^2 per pixel^2")
    
    # Create rasterized mask once (shared between both methods)
    raster_mask, transform, seg_bounds, raster_pixel_size = create_segmentation_raster(
        segmentation_union, patch_size, logger
    )
    
    # Debug: Test coordinate transformation with a known patch
    if len(all_coords) > 0:
        test_x, test_y = all_coords[0]
        if HAS_RASTERIO:
            test_col, test_row = ~transform * (test_x, test_y)
        else:
            test_col, test_row = transform.world_to_raster(test_x, test_y)
        logger.debug(f"Test patch coordinate: world=({test_x:.1f}, {test_y:.1f}) -> "
                    f"raster=({test_col:.1f}, {test_row:.1f}), "
                    f"raster_shape={raster_mask.shape}, "
                    f"within_bounds={0 <= test_row < raster_mask.shape[0] and 0 <= test_col < raster_mask.shape[1]}")
    
    # Method 1: Thresholded (High-Attention) Cell Density
    density_high_attn = compute_thresholded_density(
        high_attention_coords, all_centroids, patch_size, 
        segmentation_union, pixels_to_mm2, attention_threshold, logger,
        raster_mask, transform, seg_bounds
    )
    
    # Method 2: Attention-Weighted Cell Density
    density_weighted = compute_weighted_density(
        all_coords, all_attention, all_centroids, patch_size,
        segmentation_union, pixels_to_mm2, logger,
        raster_mask, transform, seg_bounds
    )
    
    # Print summary
    total_computation_time = time.time() - computation_start_time
    logger.info(f"{'='*60}")
    logger.info("DENSITY METRICS SUMMARY")
    logger.info(f"{'='*60}")
    if density_high_attn is not None:
        logger.info(f"Thresholded (High-Attention) Density: {density_high_attn:.2f} cells/mm^2")
    if density_weighted is not None:
        logger.info(f"Attention-Weighted Density: {density_weighted:.2f} cells/mm^2")
    logger.info(f"{'='*60}")
    logger.info(f"Total computation time: {total_computation_time:.2f} seconds")
    logger.info(f"{'='*60}")
    
    return {
        'density_high_attn': density_high_attn,
        'density_weighted': density_weighted,
        'computation_time': total_computation_time
    }


def compute_thresholded_density(high_attention_coords, all_centroids, patch_size,
                                segmentation_union, pixels_to_mm2, attention_threshold, logger=None,
                                raster_mask=None, transform=None, seg_bounds=None):
    """Compute thresholded (high-attention) cell density (using rasterization for speed)."""
    if logger is None:
        logger = logging.getLogger('compute_attention_density')
    
    logger.info(f"--- Method 1: Thresholded (High-Attention) Cell Density ---")
    logger.info(f"Using tiles with attention > {attention_threshold}th percentile")
    
    # Create rasterized mask if not provided
    if raster_mask is None or transform is None or seg_bounds is None:
        raster_mask, transform, seg_bounds, raster_pixel_size = create_segmentation_raster(
            segmentation_union, patch_size, logger
        )
    
    # Vectorize cell counting using broadcasting
    patch_x = high_attention_coords[:, 0:1]  # Shape: (n_patches, 1)
    patch_y = high_attention_coords[:, 1:2]  # Shape: (n_patches, 1)
    
    cell_x = all_centroids[:, 0]  # Shape: (n_cells,)
    cell_y = all_centroids[:, 1]  # Shape: (n_cells,)
    
    # Broadcasting: (n_patches, 1) op (n_cells,) -> (n_patches, n_cells)
    x_in_patch = (cell_x >= patch_x) & (cell_x < patch_x + patch_size)
    y_in_patch = (cell_y >= patch_y) & (cell_y < patch_y + patch_size)
    cells_in_patches = np.sum(x_in_patch & y_in_patch, axis=1)  # Shape: (n_patches,)
    total_cells = np.sum(cells_in_patches)
    
    # Compute tissue areas using raster mask
    total_tissue_area_mm2 = 0.0
    patch_x_coords = high_attention_coords[:, 0]
    patch_y_coords = high_attention_coords[:, 1]
    
        # Sample raster mask for each patch
    for i in range(len(high_attention_coords)):
        x, y = patch_x_coords[i], patch_y_coords[i]
        patch_x_end = x + patch_size
        patch_y_end = y + patch_size
        
        # Check if patch overlaps with segmentation bounds
        if not (patch_x_end >= seg_bounds[0] and x <= seg_bounds[2] and
                patch_y_end >= seg_bounds[1] and y <= seg_bounds[3]):
            # Patch is completely outside segmentation bounds, skip it
            continue
        
        # Convert patch corners to raster coordinates
        if HAS_RASTERIO:
            # Use rasterio transform inverse - returns (col, row)
            # Note: rasterio uses top-left origin, Y increases downward
            col_start, row_start = ~transform * (x, y)
            col_end, row_end = ~transform * (patch_x_end, patch_y_end)
        else:
            # Use custom transform method - returns (col, row)
            col_start, row_start = transform.world_to_raster(x, y)
            col_end, row_end = transform.world_to_raster(patch_x_end, patch_y_end)
        
        # Get integer indices (rasterio returns col,row but numpy arrays use [row,col])
        # Use floor for start (inclusive) and ceil for end (exclusive for proper slicing)
        row_start = int(np.floor(row_start))
        row_end = int(np.ceil(row_end))
        col_start = int(np.floor(col_start))
        col_end = int(np.ceil(col_end))
        
        # Clamp to valid array bounds [0, shape)
        # For slicing, end index can equal shape (exclusive), but start must be < shape
        row_start = max(0, min(row_start, raster_mask.shape[0] - 1))
        row_end = max(row_start + 1, min(row_end, raster_mask.shape[0]))  # Ensure at least 1 row
        col_start = max(0, min(col_start, raster_mask.shape[1] - 1))
        col_end = max(col_start + 1, min(col_end, raster_mask.shape[1]))  # Ensure at least 1 col
        
        if row_end > row_start and col_end > col_start:
            # Sample the mask for this patch region
            # Note: numpy arrays use [row, col] indexing
            patch_mask_region = raster_mask[row_start:row_end, col_start:col_end]
            tissue_pixels = np.sum(patch_mask_region)
            total_pixels = patch_mask_region.size
            
            if total_pixels > 0:
                # Calculate tissue coverage ratio
                coverage_ratio = tissue_pixels / total_pixels
                # Convert to mm^2
                patch_area_pixels2 = patch_size * patch_size
                tissue_area_mm2 = patch_area_pixels2 * coverage_ratio * pixels_to_mm2
                total_tissue_area_mm2 += tissue_area_mm2
            # Debug first few patches
            if i < 3:
                logger.debug(f"Patch {i}: world=({x:.1f},{y:.1f})->({patch_x_end:.1f},{patch_y_end:.1f}), "
                           f"raster_coords=({col_start:.1f},{row_start:.1f})->({col_end:.1f},{row_end:.1f}), "
                           f"array_indices=[{row_start}:{row_end}, {col_start}:{col_end}], "
                           f"region_shape={patch_mask_region.shape}, tissue={tissue_pixels}/{total_pixels}")
    
    if total_tissue_area_mm2 > 0:
        density = total_cells / total_tissue_area_mm2
        logger.info(f"Total cells in high-attention tiles: {total_cells}")
        logger.info(f"Total tissue area in high-attention tiles: {total_tissue_area_mm2:.4f} mm^2")
        logger.info(f"Density (high-attention): {density:.2f} cells/mm^2")
        return density
    else:
        logger.warning("No tissue area found in high-attention tiles")
        return None


def create_segmentation_raster(segmentation_union, patch_size, logger=None):
    """Create a rasterized mask of the segmentation polygon with intelligent downsampling."""
    if logger is None:
        logger = logging.getLogger('compute_attention_density')
    
    seg_bounds = segmentation_union.bounds  # (minx, miny, maxx, maxy)
    seg_width = seg_bounds[2] - seg_bounds[0]
    seg_height = seg_bounds[3] - seg_bounds[1]
    
    # Choose resolution: use 1 pixels per patch_size for good accuracy
    # This gives us sub-patch accuracy while keeping memory reasonable
    raster_pixel_size = patch_size / 1.0  # Each raster pixel represents patch_size/1 original pixels
    
    # Calculate raster dimensions
    raster_width = int(np.ceil(seg_width / raster_pixel_size))
    raster_height = int(np.ceil(seg_height / raster_pixel_size))
    
    # Cap maximum dimensions to avoid excessive memory (e.g., max 5000x5000 = 25M pixels = ~25MB)
    max_dimension = 5000
    if raster_width > max_dimension or raster_height > max_dimension:
        scale_factor = max(raster_width, raster_height) / max_dimension
        raster_width = int(raster_width / scale_factor)
        raster_height = int(raster_height / scale_factor)
        raster_pixel_size = seg_width / raster_width  # Recalculate to maintain coverage
        logger.info(f"Raster dimensions capped to {raster_width}x{raster_height} for memory efficiency")
    
    logger.info(f"Creating raster mask: {raster_width}x{raster_height} pixels "
                f"(pixel size: {raster_pixel_size:.1f}, memory: ~{raster_width*raster_height/1e6:.1f}MB)")
    logger.info(f"Segmentation bounds: X=[{seg_bounds[0]:.1f}, {seg_bounds[2]:.1f}], "
                f"Y=[{seg_bounds[1]:.1f}, {seg_bounds[3]:.1f}]")
    
    if HAS_RASTERIO:
        # Use rasterio for efficient rasterization
        transform = from_bounds(
            seg_bounds[0], seg_bounds[1], seg_bounds[2], seg_bounds[3],
            raster_width, raster_height
        )
        
        # Rasterize the polygon
        shapes = [(mapping(segmentation_union), 1)]
        raster_mask = features.rasterize(
            shapes,
            out_shape=(raster_height, raster_width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        ).astype(bool)
        
        # Debug: Check if raster has any tissue pixels
        tissue_pixel_count = np.sum(raster_mask)
        logger.info(f"Raster mask created: {tissue_pixel_count} tissue pixels out of {raster_mask.size} total "
                   f"({100.0*tissue_pixel_count/raster_mask.size:.1f}% coverage)")
        
        # Verify raster bounds match expected
        test_col, test_row = ~transform * (seg_bounds[0], seg_bounds[1])
        logger.debug(f"Raster coordinate test: bottom-left world ({seg_bounds[0]:.1f}, {seg_bounds[1]:.1f}) -> "
                    f"raster ({test_col:.1f}, {test_row:.1f})")
    else:
        # Fallback: use a grid-based approach with Shapely
        logger.info("Using grid-based rasterization (rasterio not available)")
        raster_mask = np.zeros((raster_height, raster_width), dtype=bool)
        
        # Create a prepared geometry for faster checks
        prep_seg = prep(segmentation_union)
        
        # Calculate pixel size in world coordinates
        pixel_width = seg_width / raster_width
        pixel_height = seg_height / raster_height
        
        # Sample grid points (check center of each pixel)
        for row in range(raster_height):
            for col in range(raster_width):
                # Convert raster coordinates to world coordinates
                world_x = seg_bounds[0] + (col + 0.5) * pixel_width
                world_y = seg_bounds[1] + (row + 0.5) * pixel_height
                point = Point(world_x, world_y)
                if prep_seg.contains(point):
                    raster_mask[row, col] = True
        
        # Create a simple transform-like object (just store bounds and dimensions)
        class SimpleTransform:
            def __init__(self, bounds, width, height):
                self.bounds = bounds
                self.width = width
                self.height = height
                self.pixel_width = (bounds[2] - bounds[0]) / width
                self.pixel_height = (bounds[3] - bounds[1]) / height
            
            def world_to_raster(self, x, y):
                # Convert world coords (x, y) to raster coords (col, row)
                col = (x - self.bounds[0]) / self.pixel_width
                row = (y - self.bounds[1]) / self.pixel_height
                return (col, row)
        
        transform = SimpleTransform(seg_bounds, raster_width, raster_height)
    
    return raster_mask, transform, seg_bounds, raster_pixel_size


def compute_weighted_density(all_coords, all_attention, all_centroids, patch_size,
                            segmentation_union, pixels_to_mm2, logger=None,
                            raster_mask=None, transform=None, seg_bounds=None):
    """Compute attention-weighted cell density (using rasterization for speed)."""
    if logger is None:
        logger = logging.getLogger('compute_attention_density')
    
    logger.info(f"--- Method 2: Attention-Weighted Cell Density (All Tiles) ---")
    
    # Create rasterized mask if not provided
    if raster_mask is None or transform is None or seg_bounds is None:
        raster_start = time.time()
        raster_mask, transform, seg_bounds, raster_pixel_size = create_segmentation_raster(
            segmentation_union, patch_size, logger
        )
        raster_time = time.time() - raster_start
        logger.info(f"Rasterization time: {raster_time:.2f} seconds")
    
    # Vectorize cell counting: create boolean masks for all patches at once using broadcasting
    patch_x = all_coords[:, 0:1]  # Shape: (n_patches, 1) - column vector
    patch_y = all_coords[:, 1:2]  # Shape: (n_patches, 1) - column vector
    
    cell_x = all_centroids[:, 0]  # Shape: (n_cells,) - row vector
    cell_y = all_centroids[:, 1]  # Shape: (n_cells,) - row vector
    
    # Broadcasting: (n_patches, 1) op (n_cells,) -> (n_patches, n_cells)
    # Check which cells fall in each patch
    x_in_patch = (cell_x >= patch_x) & (cell_x < patch_x + patch_size)
    y_in_patch = (cell_y >= patch_y) & (cell_y < patch_y + patch_size)
    cells_in_patches = np.sum(x_in_patch & y_in_patch, axis=1)  # Shape: (n_patches,)
    
    # Compute tissue areas using raster mask (much faster than polygon intersections)
    tissue_areas_mm2 = np.zeros(len(all_coords))
    
    patch_x_coords = all_coords[:, 0]
    patch_y_coords = all_coords[:, 1]
    
        # For each patch, sample the raster mask
    for i in range(len(all_coords)):
        x, y = patch_x_coords[i], patch_y_coords[i]
        patch_x_end = x + patch_size
        patch_y_end = y + patch_size
        
        # Check if patch overlaps with segmentation bounds
        if not (patch_x_end >= seg_bounds[0] and x <= seg_bounds[2] and
                patch_y_end >= seg_bounds[1] and y <= seg_bounds[3]):
            # Patch is completely outside segmentation bounds, skip it (tissue_area = 0)
            continue
        
        # Convert patch corners to raster coordinates
        if HAS_RASTERIO:
            # Use rasterio transform inverse - returns (col, row)
            # Note: rasterio uses top-left origin, Y increases downward
            col_start, row_start = ~transform * (x, y)
            col_end, row_end = ~transform * (patch_x_end, patch_y_end)
        else:
            # Use custom transform method - returns (col, row)
            col_start, row_start = transform.world_to_raster(x, y)
            col_end, row_end = transform.world_to_raster(patch_x_end, patch_y_end)
        
        # Get integer indices (rasterio returns col,row but numpy arrays use [row,col])
        # Use floor for start (inclusive) and ceil for end (exclusive for proper slicing)
        row_start = int(np.floor(row_start))
        row_end = int(np.ceil(row_end))
        col_start = int(np.floor(col_start))
        col_end = int(np.ceil(col_end))
        
        # Clamp to valid array bounds [0, shape)
        # For slicing, end index can equal shape (exclusive), but start must be < shape
        row_start = max(0, min(row_start, raster_mask.shape[0] - 1))
        row_end = max(row_start + 1, min(row_end, raster_mask.shape[0]))  # Ensure at least 1 row
        col_start = max(0, min(col_start, raster_mask.shape[1] - 1))
        col_end = max(col_start + 1, min(col_end, raster_mask.shape[1]))  # Ensure at least 1 col
        
        if row_end > row_start and col_end > col_start:
            # Sample the mask for this patch region
            # Note: numpy arrays use [row, col] indexing
            patch_mask_region = raster_mask[row_start:row_end, col_start:col_end]
            tissue_pixels = np.sum(patch_mask_region)
            total_pixels = patch_mask_region.size
            
            if total_pixels > 0:
                # Calculate tissue coverage ratio
                coverage_ratio = tissue_pixels / total_pixels
                # Convert to mm^2: patch_area * coverage_ratio * pixels_to_mm2
                patch_area_pixels2 = patch_size * patch_size
                tissue_area_mm2 = patch_area_pixels2 * coverage_ratio * pixels_to_mm2
                tissue_areas_mm2[i] = tissue_area_mm2
            # Debug first few patches
            if i < 3:
                logger.debug(f"Patch {i}: world=({x:.1f},{y:.1f})->({patch_x_end:.1f},{patch_y_end:.1f}), "
                           f"raster_coords=({col_start:.1f},{row_start:.1f})->({col_end:.1f},{row_end:.1f}), "
                           f"array_indices=[{row_start}:{row_end}, {col_start}:{col_end}], "
                           f"region_shape={patch_mask_region.shape}, tissue={tissue_pixels}/{total_pixels}")
    
    # Vectorized weighted sums using numpy operations
    weighted_cells_sum = np.sum(all_attention * cells_in_patches)
    weighted_area_sum = np.sum(all_attention * tissue_areas_mm2)
    
    # Debug: Check how many patches have tissue area
    patches_with_tissue = np.sum(tissue_areas_mm2 > 0)
    logger.debug(f"Patches with tissue area: {patches_with_tissue} out of {len(all_coords)}")
    logger.debug(f"Total tissue area sum (unweighted): {np.sum(tissue_areas_mm2):.4f} mm^2")
    logger.debug(f"Weighted area sum: {weighted_area_sum:.4f} mm^2")
    
    if weighted_area_sum > 0:
        density = weighted_cells_sum / weighted_area_sum
        logger.info(f"Weighted cells sum: {weighted_cells_sum:.2f}")
        logger.info(f"Weighted tissue area sum: {weighted_area_sum:.4f} mm^2")
        logger.info(f"Density (attention-weighted): {density:.2f} cells/mm^2")
        return density
    else:
        logger.warning(f"No weighted tissue area found (patches with tissue: {patches_with_tissue}/{len(all_coords)})")
        return None


def plot_segmentation_outline(features, ax):
    """Plot segmentation outline from GeoJSON features."""
    for feature in features:
        geometry = feature.get('geometry', {})
        geom_type = geometry.get('type', '')
        coords = geometry.get('coordinates', [])
        
        if geom_type == 'Polygon':
            if coords and len(coords) > 0:
                ring = np.array(coords[0])
                x_ring = ring[:, 0]
                y_ring = ring[:, 1]
                ax.plot(x_ring, y_ring, 'b-', linewidth=1.5, alpha=0.7)
                ax.fill(x_ring, y_ring, alpha=0.2, color='blue')
        elif geom_type == 'MultiPolygon':
            for polygon in coords:
                if polygon and len(polygon) > 0:
                    ring = np.array(polygon[0])
                    x_ring = ring[:, 0]
                    y_ring = ring[:, 1]
                    ax.plot(x_ring, y_ring, 'b-', linewidth=1.5, alpha=0.7)
                    ax.fill(x_ring, y_ring, alpha=0.2, color='blue')
        elif geom_type == 'LineString':
            ring = np.array(coords)
            x_ring = ring[:, 0]
            y_ring = ring[:, 1]
            ax.plot(x_ring, y_ring, 'b-', linewidth=1.5, alpha=0.7)


def plot_attention_patches(high_attention_coords, patch_size, ax):
    """Plot high-attention patches as semi-transparent squares."""
    for coord in high_attention_coords:
        x, y = coord[0], coord[1]
        rect = patches.Rectangle(
            (x, y), patch_size, patch_size,
            linewidth=0.5, edgecolor='red', facecolor='red', alpha=0.3
        )
        ax.add_patch(rect)


def plot_cell_centroids(cells, high_attention_coords, patch_size, type_map,
                        sampling_percentage, window_params, ax, logger=None):
    """Plot sampled cell centroids in high-attention zones."""
    if logger is None:
        logger = logging.getLogger('compute_attention_density')
    
    if not cells or high_attention_coords is None or patch_size is None:
        logger.warning("Cannot plot nuclei - missing required data")
        return
    
    centroids = np.array([cell['centroid'] for cell in cells])
    
    # Sample specified percentage
    sampling_ratio = sampling_percentage / 100.0
    sample_size = max(1, int(len(centroids) * sampling_ratio))
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(len(centroids), size=sample_size, replace=False)
    sampled_centroids = centroids[sample_indices]
    logger.info(f"Sampled {len(sampled_centroids)} centroids ({sampling_percentage}% of {len(centroids)})")
    
    # Find centroids inside high-attention zones
    inside_mask = np.zeros(len(sampled_centroids), dtype=bool)
    for patch_coord in high_attention_coords:
        patch_x, patch_y = patch_coord[0], patch_coord[1]
        x_inside = (sampled_centroids[:, 0] >= patch_x) & (sampled_centroids[:, 0] < patch_x + patch_size)
        y_inside = (sampled_centroids[:, 1] >= patch_y) & (sampled_centroids[:, 1] < patch_y + patch_size)
        inside_mask |= (x_inside & y_inside)
    
    inside_centroids = sampled_centroids[inside_mask]
    logger.info(f"Found {len(inside_centroids)} centroids inside high-attention zones "
          f"({len(inside_centroids)/len(sampled_centroids)*100:.1f}% of sampled)")
    
    # Filter by window if specified
    if window_params is not None:
        window_x, window_y, window_width, window_height = window_params
        window_x_end = window_x + window_width
        window_y_end = window_y + window_height
        window_mask = (
            (inside_centroids[:, 0] >= window_x) & (inside_centroids[:, 0] < window_x_end) &
            (inside_centroids[:, 1] >= window_y) & (inside_centroids[:, 1] < window_y_end)
        )
        inside_centroids = inside_centroids[window_mask]
        print(f"Filtered to {len(inside_centroids)} centroids within window")
    
    # Plot centroids
    if len(inside_centroids) > 0:
        # Build label
        label = 'Nuclei in high-attention zones'
        if type_map:
            # Try to get cell types from first cell
            if cells and 'type' in cells[0]:
                cell_types = set(cell.get('type') for cell in cells)
                type_names = [type_map.get(str(tid), 'Unknown') for tid in sorted(cell_types) if str(tid) in type_map]
                if type_names:
                    label = f'Nuclei ({", ".join(type_names)}) in high-attention zones'
        
        ax.scatter(inside_centroids[:, 0], inside_centroids[:, 1], 
                  c='green', s=0.1, alpha=0.6, label=label)
        ax.legend()


def setup_plot(figsize=(12, 12)):
    """Create and configure matplotlib figure and axes."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    return fig, ax


def finalize_plot(ax, args, base_filename, window_params=None, output_dir=None, save_visualization=True):
    """Set plot labels, title, window, and save."""
    # Set viewport/window if specified
    if window_params is not None:
        window_x, window_y, window_width, window_height = window_params
        ax.set_xlim(window_x, window_x + window_width)
        ax.set_ylim(window_y, window_y + window_height)
        print(f"\nZoomed to window: X=[{window_x}, {window_x + window_width}], "
              f"Y=[{window_y}, {window_y + window_height}]")
    
    # Build title
    title_parts = ['Segmentation Outline with High-Attention Patches']
    title_parts.append(f'(Threshold: {args.attention_threshold}th percentile)')
    if args.display_coverage:
        title_parts.append('with coverage ratios')
    if args.display_cells:
        title_parts.append('with nuclei')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('\n'.join(title_parts), fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    # Save visualization if requested
    if save_visualization:
        if output_dir is not None:
            # Save in visualization subdirectory
            viz_dir = Path(output_dir) / 'visualization'
            viz_dir.mkdir(parents=True, exist_ok=True)
            output_path = viz_dir / f"{base_filename}_visualization.png"
        else:
            # Save in current directory
            output_path = Path(f"{base_filename}_visualization.png")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    plt.close()  # Close figure to free memory


def print_coordinate_info(features, logger=None):
    """Print coordinate range information from features."""
    if logger is None:
        logger = logging.getLogger('compute_attention_density')
    
    all_coords = []
    for feature in features:
        geometry = feature.get('geometry', {})
        geom_type = geometry.get('type', '')
        coords = geometry.get('coordinates', [])
        
        if geom_type == 'Polygon':
            if coords and len(coords) > 0:
                all_coords.extend(coords[0])
        elif geom_type == 'MultiPolygon':
            for polygon in coords:
                if polygon and len(polygon) > 0:
                    all_coords.extend(polygon[0])
        elif geom_type == 'LineString':
            all_coords.extend(coords)
        elif geom_type == 'Point':
            all_coords.append(coords)
    
    if all_coords:
        coords_array = np.array(all_coords)
        x_coords = coords_array[:, 0]
        y_coords = coords_array[:, 1]
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        logger.info(f"Coordinate range: X=[{x_coords.min():.1f}, {x_coords.max():.1f}], "
              f"Y=[{y_coords.min():.1f}, {y_coords.max():.1f}]")
        logger.info(f"Dimensions: {x_range:.1f} x {y_range:.1f}")


def setup_logging(output_dir=None):
    """Setup logging to both console and file."""
    logger = logging.getLogger('compute_attention_density')
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) 
                      for h in logger.handlers)
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    
    # Add console handler if not present
    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if output_dir is provided and file handler doesn't exist
    if output_dir and not has_file:
        log_file = Path(output_dir) / 'processing.log'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def check_required_files_exist(base_filename):
    """Check if all required files exist for a given base_filename."""
    paths = get_file_paths(base_filename)
    required_files = {
        'geo_json': paths['geo_json'],
        'attention_h5': paths['attention_h5'],
        'cell_detection': paths['cell_detection']
    }
    
    missing_files = []
    for file_type, file_path in required_files.items():
        if not Path(file_path).exists():
            missing_files.append(f"{file_type}: {file_path}")
    
    return len(missing_files) == 0, missing_files


def find_available_cases_from_directory():
    """Find all available cases by scanning the cell_detection directory."""
    cell_detection_base = DATA_ROOTS["nuclei_dir"]
    available_cases = []
    
    if not Path(cell_detection_base).exists():
        return available_cases
    
    for case_dir in Path(cell_detection_base).iterdir():
        if case_dir.is_dir():
            base_filename = case_dir.name
            cell_detection_file = case_dir / "cell_detection.json.snappy"
            json_file = case_dir / "cell_detection.json"
            if cell_detection_file.exists() or json_file.exists():
                # Check if other required files exist
                files_exist, _ = check_required_files_exist(base_filename)
                if files_exist:
                    available_cases.append(base_filename)
    
    return sorted(available_cases)


def find_available_cases_from_csv(csv_path):
    """Extract base_file_name column from CSV and check which cases have required files."""
    try:
        df = pd.read_csv(csv_path)
        if 'base_file_name' not in df.columns:
            raise ValueError(f"CSV file must contain 'base_file_name' column")
        
        available_cases = []
        for base_filename in df['base_file_name'].dropna().unique():
            files_exist, _ = check_required_files_exist(str(base_filename))
            if files_exist:
                available_cases.append(str(base_filename))
        
        return sorted(available_cases)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return []


def initialize_results_csv(output_dir):
    """Initialize or load the results CSV file."""
    csv_path = Path(output_dir) / "density_metrics_results.csv"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if CSV already exists
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded existing results CSV with {len(df)} entries")
    else:
        # Create new CSV with headers
        df = pd.DataFrame(columns=[
            'base_file_name',
            'density_high_attn',
            'density_weighted',
            'status',
            'processing_time_s'
        ])
        df.to_csv(csv_path, index=False)
        print(f"Created new results CSV: {csv_path}")
    
    return csv_path, df


def update_results_csv(csv_path, base_filename, density_high_attn, density_weighted, 
                       status, processing_time):
    """Update the results CSV with a new entry."""
    try:
        # Read existing CSV
        df = pd.read_csv(csv_path)
        
        # Check if entry already exists
        existing_idx = df[df['base_file_name'] == base_filename].index
        
        new_row = {
            'base_file_name': base_filename,
            'density_high_attn': density_high_attn if density_high_attn is not None else '',
            'density_weighted': density_weighted if density_weighted is not None else '',
            'status': status,
            'processing_time_s': f"{processing_time:.2f}"
        }
        
        if len(existing_idx) > 0:
            # Update existing entry
            for idx in existing_idx:
                for col, val in new_row.items():
                    df.at[idx, col] = val
            print(f"Updated entry for {base_filename} in results CSV")
        else:
            # Append new entry
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"Added entry for {base_filename} to results CSV")
        
        # Save CSV
        df.to_csv(csv_path, index=False)
        return True
    except Exception as e:
        print(f"Error updating results CSV: {e}")
        return False


def process_single_case(base_filename, args, output_dir=None):
    """Process a single case and return results."""
    case_start_time = time.time()
    
    # Setup logger for this case
    logger = setup_logging(output_dir)
    logger.info(f"\n\nLoading case {base_filename}")
    
    try:
        # Get file paths
        paths = get_file_paths(base_filename)
        window_params = None
        if args.window_x is not None:
            window_params = (args.window_x, args.window_y, args.window_width, args.window_height)
        
        # Check if files exist
        files_exist, missing_files = check_required_files_exist(base_filename)
        if not files_exist:
            logger.warning(f"Skipping {base_filename} - missing files: {missing_files}")
            return {
                'base_filename': base_filename,
                'density_high_attn': None,
                'density_weighted': None,
                'status': 'failure',
                'processing_time': time.time() - case_start_time,
                'error': f"Missing files: {missing_files}"
            }
        
        # Load GeoJSON
        geo_json, features = load_geojson(paths['geo_json'])
        print_coordinate_info(features, logger)
        
        # Setup plot (only if visualization is requested)
        fig, ax = None, None
        if args.visualization:
            fig, ax = setup_plot()
            plot_segmentation_outline(features, ax)
        
        # Load attention data
        attention_data = load_attention_data(
            paths['attention_h5'], args.attention_threshold, window_params, logger
        )
        
        if attention_data:
            if args.visualization:
                plot_attention_patches(
                    attention_data['high_attention_coords'],
                    attention_data['patch_size_level0'],
                    ax
                )
        else:
            attention_data = {
                'all_attention': None,
                'all_coords': None,
                'high_attention_coords': None,
                'patch_size_level0': None
            }
        
        # Extract segmentation polygons
        segmentation_union = extract_segmentation_polygons(features, logger)
        
        # Compute and display coverage if requested
        if args.display_coverage and attention_data['high_attention_coords'] is not None and args.visualization:
            if segmentation_union is not None:
                compute_tissue_coverage(
                    attention_data['high_attention_coords'],
                    attention_data['patch_size_level0'],
                    segmentation_union,
                    ax
                )
        
        # Load cell detection data for density calculation
        cell_data_density = load_cell_detection_data(paths['cell_detection'], args.cell_type)
        
        # Compute density metrics
        density_results = None
        if (cell_data_density and cell_data_density['base_mpp'] and 
            attention_data['all_attention'] is not None and segmentation_union is not None):
            density_results = compute_cell_density_metrics(
                cell_data_density['cells'],
                attention_data['all_attention'],
                attention_data['all_coords'],
                attention_data['high_attention_coords'],
                attention_data['patch_size_level0'],
                segmentation_union,
                cell_data_density['base_mpp'],
                args.attention_threshold,
                window_params,
                args.filter_cells_by_segmentation,
                logger
            )
        
        # Load and plot cells for visualization if requested
        if args.display_cells and args.visualization:
            cell_data_viz = load_cell_detection_data(paths['cell_detection'], args.cell_type)
            
            if cell_data_viz:
                plot_cell_centroids(
                    cell_data_viz['cells'],
                    attention_data['high_attention_coords'],
                    attention_data['patch_size_level0'],
                    cell_data_viz['type_map'],
                    args.cell_sampling_percentage,
                    window_params,
                    ax,
                    logger
                )
        
        # Finalize and save plot if visualization is requested
        if args.visualization and fig is not None:
            finalize_plot(ax, args, base_filename, window_params, output_dir, args.visualization)
        
        processing_time = time.time() - case_start_time
        
        # Extract density values
        density_high_attn = density_results['density_high_attn'] if density_results else None
        density_weighted = density_results['density_weighted'] if density_results else None
        
        return {
            'base_filename': base_filename,
            'density_high_attn': density_high_attn,
            'density_weighted': density_weighted,
            'status': 'success',
            'processing_time': processing_time,
            'error': None
        }
        
    except Exception as e:
        processing_time = time.time() - case_start_time
        logger = logging.getLogger('compute_attention_density')
        logger.error(f"ERROR processing {base_filename}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'base_filename': base_filename,
            'density_high_attn': None,
            'density_weighted': None,
            'status': 'failure',
            'processing_time': processing_time,
            'error': str(e)
        }


def main():
    """Main function to orchestrate the visualization pipeline."""
    script_start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging (if output_analysis is provided, log to file)
    logger = setup_logging(args.output_analysis)
    
    # Handle batch processing
    if args.process_batch:
        # Initialize results CSV
        csv_path, df = initialize_results_csv(args.output_analysis)
        
        # Find cases to process
        if args.input_csv:
            print(f"\nReading cases from CSV: {args.input_csv}")
            cases_to_process = find_available_cases_from_csv(args.input_csv)
        else:
            print(f"\nFinding all available cases from cell_detection directory...")
            cases_to_process = find_available_cases_from_directory()
        
        print(f"Found {len(cases_to_process)} cases with all required files")
        
        if len(cases_to_process) == 0:
            print("No cases to process. Exiting.")
            return
        
        # Process each case
        for i, base_filename in enumerate(cases_to_process, 1):
            print(f"\n{'#'*80}")
            print(f"Processing case {i}/{len(cases_to_process)}: {base_filename}")
            print(f"{'#'*80}")
            
            # Process the case
            result = process_single_case(base_filename, args, args.output_analysis)
            
            # Update CSV
            update_results_csv(
                csv_path,
                result['base_filename'],
                result['density_high_attn'],
                result['density_weighted'],
                result['status'],
                result['processing_time']
            )
        
        # Print batch summary
        total_batch_time = time.time() - script_start_time
        print(f"\n{'='*80}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"Total cases processed: {len(cases_to_process)}")
        print(f"Total batch time: {total_batch_time:.2f} seconds ({total_batch_time/60:.2f} minutes)")
        print(f"Average time per case: {total_batch_time/len(cases_to_process):.2f} seconds")
        print(f"Results saved to: {csv_path}")
        print(f"{'='*80}")
        
    else:
        # Single case processing (original behavior)
        # If output_analysis is not set, default to saving visualization (original behavior)
        if not args.output_analysis:
            args.visualization = True
        
        result = process_single_case(args.base_filename, args, args.output_analysis)
        
        # Update CSV if output_analysis is specified
        if args.output_analysis:
            csv_path, _ = initialize_results_csv(args.output_analysis)
            update_results_csv(
                csv_path,
                result['base_filename'],
                result['density_high_attn'],
                result['density_weighted'],
                result['status'],
                result['processing_time']
            )
        
        # Print execution summary
        total_script_time = time.time() - script_start_time
        print(f"\n{'='*60}")
        print("EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total script execution time: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes)")
        if result['status'] == 'success' and result.get('density_high_attn') is not None:
            print(f"  - Density computation: {result['processing_time']:.2f} seconds")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
