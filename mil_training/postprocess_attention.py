"""
Attention Post-Processing Script.

This script combines MIL tile-level attention with UNI encoder token-level attention
to produce granular per-token attention maps for whole slide images.

Pipeline:
1. Load MIL attention from compute_attention.py output H5 files
2. Load WSI slides and compute UNI encoder attention (14x14) on-the-fly
3. Combine: combined_attention = MIL_attention * normalize(UNI_attention)
4. Save organized H5 files with combined attention per slide
5. Generate thumbnail visualization overlays
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from contextlib import contextmanager
import tempfile
import multiprocessing as mp

import torch
import torch.multiprocessing as torch_mp
import torch.nn.functional as F
import numpy as np
import h5py
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

# For HuggingFace login (some models need access)
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir):
    """Set up logging configuration."""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'postprocess_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


# =============================================================================
# Utility Functions
# =============================================================================

@contextmanager
def complete_stderr_suppression():
    """Complete stderr suppression for operations with lots of TIFF warnings."""
    original_stderr = os.dup(2)
    try:
        with tempfile.TemporaryFile() as temp_stderr:
            os.dup2(temp_stderr.fileno(), 2)
            yield
    finally:
        os.dup2(original_stderr, 2)
        os.close(original_stderr)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Post-process MIL attention with UNI encoder token-level attention'
    )
    
    # Required arguments
    parser.add_argument('--mil_attention_dir', type=str, required=True,
                        help='Directory with MIL attention H5 files from compute_attention.py')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='CSV file with sl_filepath (WSI paths) and slide_name columns')
    parser.add_argument('--thumbnail_dir', type=str, required=True,
                        help='Directory containing thumbnail images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for combined attention H5 files')
    
    # Optional arguments
    parser.add_argument('--visualization_dir', type=str, default=None,
                        help='Directory for thumbnail visualizations (default: {output_dir}/visualizations)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for UNI encoder when using single GPU (default: cuda:0)')
    parser.add_argument('--cuda_devices', type=str, default=None,
                        help='Comma-separated list of CUDA device indices for multi-GPU processing (e.g., "0,1,2"). '
                             'Overrides --device when specified.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for UNI attention extraction (default: 256)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip slides that already have output files')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Transparency for visualization overlay (default: 0.5)')
    parser.add_argument('--tile_size', type=int, default=256,
                        help='Default tile size in pixels (fallback if patch_size_level0 not in H5, default: 256)')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip generating visualizations')
    
    return parser.parse_args()


# =============================================================================
# UNI Attention Extraction
# =============================================================================

def get_attention_hook(attention_maps_list):
    """
    Create a forward hook to capture attention weights from the last transformer block.
    The hook extracts CLS token attention to patches and averages across heads.
    
    Args:
        attention_maps_list: List to store captured attention maps
        
    Returns:
        Hook function
    """
    def hook_fn(module, input, output):
        """
        Hook function that captures attention weights during forward pass.
        """
        try:
            # Get input tensor shape: (B, N, C) where N includes CLS token + patches
            x = input[0] if isinstance(input, tuple) else input
            if x is None or not isinstance(x, torch.Tensor):
                return
            B, N, C = x.shape
            
            # Get number of heads from module
            num_heads = getattr(module, 'num_heads', None)
            if num_heads is None:
                qkv_out = module.qkv.out_features if hasattr(module.qkv, 'out_features') else None
                if qkv_out:
                    num_heads = qkv_out // (3 * C)
                else:
                    num_heads = 16  # UNI typically uses 16 heads
            
            head_dim = C // num_heads
            
            # Compute QKV
            qkv = module.qkv(x)
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Compute attention scores
            scale = getattr(module, 'scale', None)
            if scale is None:
                scale = 1.0 / np.sqrt(float(head_dim))
            elif isinstance(scale, torch.Tensor):
                scale = scale.item()
            
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            
            # Extract CLS token attention to patches
            # attn shape: (B, num_heads, N, N)
            cls_attn = attn[:, :, 0:1, 1:]  # (B, num_heads, 1, N-1)
            
            num_patches_tokens = cls_attn.shape[-1]
            spatial_dim = int(np.sqrt(num_patches_tokens))
            
            if spatial_dim * spatial_dim != num_patches_tokens:
                if num_patches_tokens == 196:
                    spatial_dim = 14
                elif num_patches_tokens == 256:
                    spatial_dim = 16
            
            # Reshape to spatial dimensions
            cls_attn = cls_attn.squeeze(2)
            cls_attn = cls_attn.reshape(B, num_heads, spatial_dim, spatial_dim)
            
            # Average across heads
            cls_attn_avg = cls_attn.mean(dim=1)  # (B, spatial_dim, spatial_dim)
            
            # Store in list
            attention_maps_list.append(cls_attn_avg.detach().cpu().numpy())
            
        except Exception as e:
            logging.warning(f"Error extracting attention in hook: {e}")
    
    return hook_fn


def initialize_uni_encoder(device):
    """
    Initialize the UNI encoder model.
    
    Args:
        device: Device to load model on
        
    Returns:
        encoder: UNI encoder model
    """
    # Try to login to HuggingFace if token is available
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        try:
            login(token=token)
        except Exception as e:
            logging.warning(f"HuggingFace login failed: {e}")
    
    # Import trident encoder factory
    try:
        from trident.patch_encoder_models import encoder_factory
    except ImportError:
        raise ImportError("trident package not found. Please install it to use UNI encoder.")
    
    logging.info("Initializing UNI encoder...")
    encoder = encoder_factory(model_name="uni_v1")
    encoder.eval()
    encoder.to(device)
    logging.info("UNI encoder initialized successfully")
    
    return encoder


def extract_uni_attention_for_slide(slide_path, coords, encoder, device, batch_size=256, 
                                     tile_size=256, tile_size_level0=None):
    """
    Extract 14x14 UNI attention maps for all tiles in a slide.
    
    Args:
        slide_path: Path to WSI file
        coords: Array of tile coordinates (N, 2) at level 0
        encoder: UNI encoder model
        device: Device for computation
        batch_size: Batch size for processing
        tile_size: Tile size for extraction (256 for UNI)
        tile_size_level0: Actual patch footprint at level 0 (for 40x slides, this is 512)
                          If None, defaults to tile_size (assumes 20x slide)
        
    Returns:
        attention_maps: Array of attention maps (N, 14, 14)
    """
    import openslide
    
    # If tile_size_level0 not provided, assume same as tile_size (20x slide)
    if tile_size_level0 is None:
        tile_size_level0 = tile_size
    
    logging.info(f"Extracting UNI attention for {len(coords)} tiles...")
    logging.info(f"tile_size_level0: {tile_size_level0} (patch footprint at level 0)")
    
    # Open the slide
    with complete_stderr_suppression():
        slide = openslide.OpenSlide(slide_path)
    
    attention_maps_list = []
    
    # Get the last transformer block's attention module (layer 23)
    last_block = encoder.model.blocks[23]
    attention_module = last_block.attn
    
    # Register hook to capture attention
    captured_attention = []
    hook_handle = attention_module.register_forward_hook(get_attention_hook(captured_attention))
    
    try:
        encoder.eval()
        with torch.no_grad():
            for batch_start in range(0, len(coords), batch_size):
                batch_end = min(batch_start + batch_size, len(coords))
                batch_coords = coords[batch_start:batch_end]
                
                # Load patches
                # Read patches at tile_size_level0 (the actual footprint at level 0)
                # This handles both 20x slides (256px) and 40x slides (512px)
                patches = []
                for coord in batch_coords:
                    x, y = int(coord[0]), int(coord[1])
                    try:
                        with complete_stderr_suppression():
                            # Read at the actual footprint size at level 0
                            patch = slide.read_region((x, y), 0, (tile_size_level0, tile_size_level0))
                        patch_rgb = patch.convert('RGB')
                        patch_array = np.array(patch_rgb, dtype=np.uint8)
                        patches.append(patch_array)
                    except Exception as e:
                        logging.warning(f"Error loading patch at ({x}, {y}): {e}")
                        patches.append(np.zeros((tile_size_level0, tile_size_level0, 3), dtype=np.uint8))
                
                if not patches:
                    continue
                
                # Convert to tensor and normalize
                patches_array = np.stack(patches)
                patches_tensor = torch.from_numpy(patches_array).float()
                patches_tensor = patches_tensor.permute(0, 3, 1, 2)  # (B, C, H, W)
                
                # UNI model expects 224x224 input - resize from tile_size_level0
                # This handles both 256->224 (20x) and 512->224 (40x) cases
                patches_tensor = F.interpolate(patches_tensor, size=(224, 224), 
                                               mode='bilinear', align_corners=False)
                
                # Normalize with ImageNet stats
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                patches_tensor = patches_tensor / 255.0
                patches_tensor = (patches_tensor - mean) / std
                
                # Move to device and run forward pass
                patches_tensor = patches_tensor.to(device)
                _ = encoder(patches_tensor)
        
        # Concatenate all attention maps
        if captured_attention:
            attention_maps = np.concatenate(captured_attention, axis=0)
        else:
            attention_maps = np.zeros((len(coords), 14, 14), dtype=np.float32)
            
    finally:
        hook_handle.remove()
        slide.close()
    
    logging.info(f"Extracted attention maps shape: {attention_maps.shape}")
    return attention_maps


# =============================================================================
# Attention Combination
# =============================================================================

def normalize_attention_map(M):
    """
    Normalize attention map to [0, 1] range.
    
    Args:
        M: attention map (14, 14)
    
    Returns:
        Normalized attention map
    """
    M_min = M.min()
    M_max = M.max()
    if M_max > M_min:
        return (M - M_min) / (M_max - M_min)
    else:
        return np.zeros_like(M)


def combine_attentions(mil_attention, uni_attention_maps):
    """
    Combine MIL tile attention with UNI token attention.
    
    Formula: combined = MIL_attention * normalize(UNI_attention)
    
    Args:
        mil_attention: Array of tile attention weights (N,)
        uni_attention_maps: Array of token attention maps (N, 14, 14)
    
    Returns:
        combined_maps: Array of combined attention maps (N, 14, 14)
    """
    combined_maps = []
    
    for a_i, M_i in zip(mil_attention, uni_attention_maps):
        M_i_norm = normalize_attention_map(M_i)
        combined = a_i * M_i_norm
        combined_maps.append(combined)
    
    return np.array(combined_maps, dtype=np.float32)


# =============================================================================
# Data Loading
# =============================================================================

def load_mil_attention(mil_attention_path):
    """
    Load MIL attention weights and coordinates from H5 file.
    
    Args:
        mil_attention_path: Path to MIL attention H5 file
        
    Returns:
        attention: Array of tile attention weights (N,)
        coords: Array of tile coordinates (N, 2)
        prediction: Model prediction
        probability: Prediction probability
        patch_size_level0: Actual patch footprint at level 0 (accounts for magnification)
        target_magnification: Target magnification used for patch extraction
    """
    with h5py.File(mil_attention_path, 'r') as f:
        # Prefer normalized attention, fall back to raw
        if 'attention_normalized' in f:
            attention = f['attention_normalized'][:]
        elif 'attention' in f:
            attention = f['attention'][:]
        else:
            raise KeyError("No attention data found in H5 file")
        
        coords = f['coords'][:]
        prediction = f.attrs.get('prediction', None)
        probability = f.attrs.get('probability', None)
        
        # Get patch_size_level0 from coords attributes (crucial for 20x/40x handling)
        # This tells us the actual footprint of each patch at level 0
        coord_attrs = dict(f['coords'].attrs)
        patch_size_level0 = int(coord_attrs.get('patch_size_level0', 256))
        target_magnification = coord_attrs.get('target_magnification', None)
        
        logging.info(f"patch_size_level0: {patch_size_level0}, target_magnification: {target_magnification}")
    
    return attention, coords, prediction, probability, patch_size_level0, target_magnification


def load_slide_mapping(csv_path):
    """
    Load CSV and create mapping from slide base names to file paths.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        mapping: Dict mapping base_name -> sl_filepath
    """
    df = pd.read_csv(csv_path)
    
    if 'sl_filepath' not in df.columns:
        raise ValueError("CSV must contain 'sl_filepath' column")
    if 'slide_name' not in df.columns:
        raise ValueError("CSV must contain 'slide_name' column")
    
    mapping = {}
    for _, row in df.iterrows():
        slide_name = str(row['slide_name'])
        base_name = slide_name.replace('.svs', '').replace('.SVS', '')
        sl_filepath = str(row['sl_filepath'])
        if pd.notna(row['sl_filepath']) and sl_filepath:
            mapping[base_name] = sl_filepath
    
    logging.info(f"Loaded {len(mapping)} slide mappings from CSV")
    return mapping


# =============================================================================
# Output H5 Saving
# =============================================================================

def save_combined_attention_h5(output_path, coords, combined_attention, 
                                slide_name, mil_attention_path, tile_size=256):
    """
    Save combined attention to H5 file.
    
    Args:
        output_path: Output H5 file path
        coords: Tile coordinates (N, 2)
        combined_attention: Combined attention maps (N, 14, 14)
        slide_name: Slide name
        mil_attention_path: Source MIL attention file path
        tile_size: Tile size in pixels
    """
    with h5py.File(output_path, 'w') as f:
        # Save coordinates
        f.create_dataset('coords', data=coords)
        
        # Save combined attention
        f.create_dataset('combined_attention', data=combined_attention.astype(np.float32))
        
        # Save metadata as attributes
        f.attrs['slide_name'] = slide_name
        f.attrs['num_tiles'] = len(coords)
        f.attrs['mil_attention_path'] = mil_attention_path
        f.attrs['processing_timestamp'] = datetime.now().isoformat()
        f.attrs['tile_size'] = tile_size
        f.attrs['token_grid_size'] = 14
    
    logging.info(f"Saved combined attention to {output_path}")


# =============================================================================
# Visualization
# =============================================================================

def get_slide_dimensions(slide_path):
    """
    Get slide dimensions from WSI file.
    
    Args:
        slide_path: Path to WSI file
        
    Returns:
        (width, height) at level 0
    """
    try:
        import openslide
        with complete_stderr_suppression():
            slide = openslide.OpenSlide(slide_path)
            dims = slide.dimensions
            slide.close()
        return dims
    except Exception as e:
        logging.warning(f"Could not get slide dimensions: {e}")
        return None


def get_slide_dimensions_from_coords(coords, tile_size=256):
    """
    Estimate slide dimensions from coordinates.
    
    Args:
        coords: Array of tile coordinates (N, 2)
        tile_size: Tile size in pixels
        
    Returns:
        (width, height) estimated dimensions
    """
    if len(coords) == 0:
        return (0, 0)
    
    max_x = coords[:, 0].max() + tile_size
    max_y = coords[:, 1].max() + tile_size
    
    return (int(max_x), int(max_y))


def overlay_attention_on_thumbnail(thumbnail_path, combined_maps, coords, slide_dims,
                                   output_path, alpha=0.5, tile_size=256, token_grid_size=14):
    """
    Overlay combined attention heatmap on thumbnail image.
    
    Args:
        thumbnail_path: Path to thumbnail image
        combined_maps: Combined attention maps (N, 14, 14)
        coords: Tile coordinates (N, 2)
        slide_dims: (width, height) of slide at level 0
        output_path: Path to save output image
        alpha: Transparency of attention overlay
        tile_size: Tile size in pixels
        token_grid_size: Token grid size (14)
    """
    # Load thumbnail
    thumbnail = Image.open(thumbnail_path)
    thumb_width, thumb_height = thumbnail.size
    
    # Calculate scale factor from level 0 to thumbnail
    slide_width, slide_height = slide_dims
    scale_x = thumb_width / slide_width
    scale_y = thumb_height / slide_height
    
    logging.info(f"Thumbnail: {thumb_width}x{thumb_height}, Slide: {slide_width}x{slide_height}")
    logging.info(f"Scale: {scale_x:.6f} x {scale_y:.6f}")
    
    # Build attention at thumbnail resolution
    attention_full = np.zeros((thumb_height, thumb_width), dtype=np.float32)
    tile_thumb_w = max(1, int(round(tile_size * scale_x)))
    tile_thumb_h = max(1, int(round(tile_size * scale_y)))
    
    for combined_map, (x, y) in zip(combined_maps, coords):
        # Resize 14x14 map to thumbnail tile size
        tile_pil = Image.fromarray(combined_map.astype(np.float32))
        tile_resized = tile_pil.resize((tile_thumb_w, tile_thumb_h), Image.Resampling.BILINEAR)
        tile_arr = np.array(tile_resized, dtype=np.float32)
        
        # Placement on thumbnail
        thumb_x = int(round(x * scale_x))
        thumb_y = int(round(y * scale_y))
        x_end = min(thumb_x + tile_thumb_w, thumb_width)
        y_end = min(thumb_y + tile_thumb_h, thumb_height)
        tile_w = x_end - thumb_x
        tile_h = y_end - thumb_y
        
        if tile_w > 0 and tile_h > 0:
            attention_full[thumb_y:y_end, thumb_x:x_end] = tile_arr[:tile_h, :tile_w]
    
    # Normalize attention for visualization
    attn_min = attention_full.min()
    attn_max = attention_full.max()
    if attn_max > attn_min:
        attention_normalized = (attention_full - attn_min) / (attn_max - attn_min)
    else:
        attention_normalized = np.zeros_like(attention_full)
    
    # Create heatmap colormap
    if hasattr(plt, 'colormaps'):
        cmap = plt.colormaps['hot']
    else:
        cmap = cm.get_cmap('hot')
    
    attention_colored = cmap(attention_normalized)
    attention_rgb = (attention_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Blend attention with thumbnail
    overlay = Image.fromarray(attention_rgb)
    if thumbnail.mode != 'RGB':
        thumbnail = thumbnail.convert('RGB')
    blended = Image.blend(thumbnail, overlay, alpha=alpha)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Original thumbnail
    axes[0].imshow(thumbnail)
    axes[0].set_title('Original Thumbnail', fontsize=12)
    axes[0].axis('off')
    
    # Draw tissue region bounds
    if len(coords) > 0:
        from matplotlib.patches import Rectangle
        min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
        max_x, max_y = coords[:, 0].max() + tile_size, coords[:, 1].max() + tile_size
        thumb_min_x = int(min_x * scale_x)
        thumb_min_y = int(min_y * scale_y)
        thumb_max_x = int(max_x * scale_x)
        thumb_max_y = int(max_y * scale_y)
        
        rect = Rectangle((thumb_min_x, thumb_min_y), 
                          thumb_max_x - thumb_min_x, 
                          thumb_max_y - thumb_min_y,
                          linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        axes[0].add_patch(rect)
        
        rect2 = Rectangle((thumb_min_x, thumb_min_y), 
                           thumb_max_x - thumb_min_x, 
                           thumb_max_y - thumb_min_y,
                           linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        axes[1].add_patch(rect2)
    
    # Overlay
    axes[1].imshow(blended)
    axes[1].set_title(f'Combined Attention Overlay (alpha={alpha})', fontsize=12)
    axes[1].axis('off')
    
    # Add info text
    info_text = f'Slide: {slide_width}x{slide_height} | Thumbnail: {thumb_width}x{thumb_height} | Tiles: {len(coords)} | Tile size (L0): {tile_size}px'
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved visualization to {output_path}")


# =============================================================================
# Multi-GPU Worker Functions
# =============================================================================

def worker_process(gpu_id, slide_tasks, args, visualization_dir, return_dict, worker_idx):
    """
    Worker process that processes slides on a specific GPU.
    
    Args:
        gpu_id: CUDA device index
        slide_tasks: List of (slide_name, slide_path) tuples to process
        args: Parsed command line arguments
        visualization_dir: Directory for visualizations
        return_dict: Multiprocessing dict to store results
        worker_idx: Worker index for logging
    """
    try:
        # Set up device
        device = torch.device(f'cuda:{gpu_id}')
        
        # Set CUDA device for this process
        torch.cuda.set_device(gpu_id)
        
        logging.info(f"[Worker {worker_idx}] Starting on GPU {gpu_id} with {len(slide_tasks)} slides")
        
        # Initialize encoder for this GPU
        encoder = initialize_uni_encoder(device)
        
        successful = 0
        failed = 0
        
        for slide_name, slide_path in slide_tasks:
            success = process_slide(
                slide_name=slide_name,
                mil_attention_dir=args.mil_attention_dir,
                slide_path=slide_path,
                thumbnail_dir=args.thumbnail_dir,
                output_dir=args.output_dir,
                visualization_dir=visualization_dir,
                encoder=encoder,
                device=device,
                batch_size=args.batch_size,
                default_tile_size=args.tile_size,
                alpha=args.alpha,
                skip_visualization=args.skip_visualization
            )
            
            if success:
                successful += 1
            else:
                failed += 1
        
        return_dict[worker_idx] = {'successful': successful, 'failed': failed}
        logging.info(f"[Worker {worker_idx}] Completed: {successful} successful, {failed} failed")
        
    except Exception as e:
        logging.error(f"[Worker {worker_idx}] Error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return_dict[worker_idx] = {'successful': 0, 'failed': len(slide_tasks)}


def distribute_slides_to_workers(slide_tasks, num_workers):
    """
    Distribute slides evenly across workers.
    
    Args:
        slide_tasks: List of (slide_name, slide_path) tuples
        num_workers: Number of workers (GPUs)
        
    Returns:
        List of lists, each containing tasks for one worker
    """
    # Distribute slides round-robin for better load balancing
    worker_tasks = [[] for _ in range(num_workers)]
    for i, task in enumerate(slide_tasks):
        worker_tasks[i % num_workers].append(task)
    return worker_tasks


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_slide(slide_name, mil_attention_dir, slide_path, thumbnail_dir,
                  output_dir, visualization_dir, encoder, device, 
                  batch_size=256, default_tile_size=256, alpha=0.5, skip_visualization=False):
    """
    Process a single slide to create combined attention.
    
    Args:
        slide_name: Base slide name (without extension)
        mil_attention_dir: Directory with MIL attention H5 files
        slide_path: Path to WSI file
        thumbnail_dir: Directory with thumbnail images
        output_dir: Output directory for H5 files
        visualization_dir: Directory for visualizations
        encoder: UNI encoder model
        device: Device for computation
        batch_size: Batch size for UNI extraction
        default_tile_size: Default tile size (used only if patch_size_level0 not in H5)
        alpha: Visualization overlay alpha
        skip_visualization: Whether to skip visualization
        
    Returns:
        bool: Success flag
    """
    try:
        # File paths
        mil_attention_path = os.path.join(mil_attention_dir, f'{slide_name}.h5')
        output_h5_path = os.path.join(output_dir, f'{slide_name}.h5')
        thumbnail_path = os.path.join(thumbnail_dir, f'{slide_name}.jpg')
        visualization_path = os.path.join(visualization_dir, f'{slide_name}_combined_attention.png')
        
        # Check required files exist
        if not os.path.exists(mil_attention_path):
            logging.error(f"MIL attention file not found: {mil_attention_path}")
            return False
        
        if not os.path.exists(slide_path):
            logging.error(f"Slide file not found: {slide_path}")
            return False
        
        logging.info(f"Processing {slide_name}...")
        
        # Load MIL attention (includes patch_size_level0 for proper magnification handling)
        mil_attention, coords, prediction, probability, patch_size_level0, target_mag = load_mil_attention(mil_attention_path)
        logging.info(f"Loaded MIL attention: {len(mil_attention)} tiles")
        
        # Use patch_size_level0 from H5 file (accounts for 20x vs 40x slides)
        # This is the actual footprint of each patch at level 0
        tile_size_level0 = patch_size_level0 if patch_size_level0 else default_tile_size
        logging.info(f"Using tile_size_level0: {tile_size_level0} (for visualization placement)")
        
        # For UNI attention extraction, we always use 256px patches (extracted at target mag)
        # The patch_size_level0 is only used for visualization placement
        extraction_tile_size = 256  # UNI was trained on 256px patches
        
        # Extract UNI attention
        uni_attention_maps = extract_uni_attention_for_slide(
            slide_path, coords, encoder, device, batch_size, 
            tile_size=extraction_tile_size, 
            tile_size_level0=tile_size_level0
        )
        
        # Combine attentions
        logging.info("Combining MIL and UNI attention...")
        combined_attention = combine_attentions(mil_attention, uni_attention_maps)
        
        # Save combined attention H5
        save_combined_attention_h5(
            output_h5_path, coords, combined_attention,
            slide_name, mil_attention_path, tile_size_level0
        )
        
        # Generate visualization if thumbnail exists and not skipped
        if not skip_visualization:
            if os.path.exists(thumbnail_path):
                # Get slide dimensions
                slide_dims = get_slide_dimensions(slide_path)
                if slide_dims is None:
                    slide_dims = get_slide_dimensions_from_coords(coords, tile_size_level0)
                
                overlay_attention_on_thumbnail(
                    thumbnail_path, combined_attention, coords, slide_dims,
                    visualization_path, alpha, tile_size_level0
                )
            else:
                logging.warning(f"Thumbnail not found, skipping visualization: {thumbnail_path}")
        
        logging.info(f"Successfully processed {slide_name}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing {slide_name}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    visualization_dir = args.visualization_dir or os.path.join(args.output_dir, 'visualizations')
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(args.output_dir)
    
    # Parse CUDA devices
    cuda_devices = None
    if args.cuda_devices:
        try:
            cuda_devices = [int(x.strip()) for x in args.cuda_devices.split(',')]
            # Validate that devices exist
            num_gpus = torch.cuda.device_count()
            for dev_id in cuda_devices:
                if dev_id >= num_gpus:
                    raise ValueError(f"CUDA device {dev_id} not available. Only {num_gpus} GPU(s) detected.")
        except ValueError as e:
            logging.error(f"Invalid --cuda_devices format: {e}")
            sys.exit(1)
    
    print("=" * 80)
    print("ATTENTION POST-PROCESSING")
    print("=" * 80)
    logging.info(f"MIL attention directory: {args.mil_attention_dir}")
    logging.info(f"CSV path: {args.csv_path}")
    logging.info(f"Thumbnail directory: {args.thumbnail_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Visualization directory: {visualization_dir}")
    if cuda_devices:
        logging.info(f"Multi-GPU mode: using CUDA devices {cuda_devices}")
    else:
        logging.info(f"Single-GPU mode: {args.device}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Log file: {log_file}")
    print("=" * 80)
    
    # Load slide mapping from CSV
    slide_mapping = load_slide_mapping(args.csv_path)
    
    # Get list of MIL attention H5 files
    mil_h5_files = [f for f in os.listdir(args.mil_attention_dir) if f.endswith('.h5')]
    logging.info(f"Found {len(mil_h5_files)} MIL attention H5 files")
    
    # Build list of slides to process (filtering skipped/missing)
    slide_tasks = []
    skipped = 0
    missing = 0
    
    for h5_file in mil_h5_files:
        slide_name = h5_file.replace('.h5', '')
        
        # Check if output already exists
        output_h5_path = os.path.join(args.output_dir, f'{slide_name}.h5')
        if args.skip_existing and os.path.exists(output_h5_path):
            logging.info(f"Skipping {slide_name} (output exists)")
            skipped += 1
            continue
        
        # Get slide path from mapping
        if slide_name not in slide_mapping:
            logging.warning(f"Slide {slide_name} not found in CSV mapping, skipping")
            missing += 1
            continue
        
        slide_path = slide_mapping[slide_name]
        slide_tasks.append((slide_name, slide_path))
    
    logging.info(f"Slides to process: {len(slide_tasks)}, Skipped (existing): {skipped}, Missing from CSV: {missing}")
    
    if len(slide_tasks) == 0:
        logging.info("No slides to process.")
        print("=" * 80)
        return
    
    # Process slides
    successful = 0
    failed = 0
    
    if cuda_devices and len(cuda_devices) > 1:
        # Multi-GPU processing
        logging.info(f"Starting multi-GPU processing with {len(cuda_devices)} workers...")
        
        # Set multiprocessing start method to spawn (required for CUDA)
        try:
            torch_mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # Distribute slides across workers
        worker_tasks = distribute_slides_to_workers(slide_tasks, len(cuda_devices))
        
        # Create manager for shared dict
        manager = mp.Manager()
        return_dict = manager.dict()
        
        # Start worker processes
        processes = []
        for worker_idx, (gpu_id, tasks) in enumerate(zip(cuda_devices, worker_tasks)):
            if len(tasks) == 0:
                continue
            logging.info(f"Worker {worker_idx}: GPU {gpu_id}, {len(tasks)} slides")
            p = torch_mp.Process(
                target=worker_process,
                args=(gpu_id, tasks, args, visualization_dir, return_dict, worker_idx)
            )
            processes.append(p)
            p.start()
        
        # Wait for all workers to complete
        for p in processes:
            p.join()
        
        # Aggregate results
        for worker_idx, result in return_dict.items():
            successful += result.get('successful', 0)
            failed += result.get('failed', 0)
    
    else:
        # Single-GPU processing (original behavior)
        if cuda_devices:
            device = torch.device(f'cuda:{cuda_devices[0]}')
        else:
            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        logging.info(f"Using device: {device}")
        encoder = initialize_uni_encoder(device)
        
        for slide_name, slide_path in tqdm(slide_tasks, desc="Processing slides"):
            success = process_slide(
                slide_name=slide_name,
                mil_attention_dir=args.mil_attention_dir,
                slide_path=slide_path,
                thumbnail_dir=args.thumbnail_dir,
                output_dir=args.output_dir,
                visualization_dir=visualization_dir,
                encoder=encoder,
                device=device,
                batch_size=args.batch_size,
                default_tile_size=args.tile_size,
                alpha=args.alpha,
                skip_visualization=args.skip_visualization
            )
            
            if success:
                successful += 1
            else:
                failed += 1
    
    # Final summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Skipped (existing): {skipped}")
    logging.info(f"Missing from CSV: {missing}")
    logging.info(f"Output directory: {args.output_dir}")
    if not args.skip_visualization:
        logging.info(f"Visualization directory: {visualization_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

