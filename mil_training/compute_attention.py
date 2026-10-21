"""
Attention computation script for trained MIL models.

This script loads a trained model and computes attention scores for slides in a dataset.
It generates attention H5 files with raw and normalized attention scores, along with
predictions and probabilities for each slide.
"""

import torch
import h5py
import pandas as pd
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import json

# Add this package directory to Python path so `from modules import ...` works.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules
from modules import GatedMILClassifier, set_seed

import warnings
warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments for attention computation."""
    parser = argparse.ArgumentParser(description='Compute attention scores for slides using trained model')
    
    # Required parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--h5_directory', type=str, required=True,
                       help='Directory containing H5 feature files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for attention H5 files')
    
    # Optional parameters - Input data
    parser.add_argument('--input_csv_path', type=str, default=None,
                       help='Path to CSV file containing slide names to process (optional). If not provided, processes all H5 files in directory.')
    parser.add_argument('--slide_id_column', type=str, default='slide_name',
                       help='Column name in CSV to use as slide identifier (default: slide_name)')
    parser.add_argument('--diagnosis_column', type=str, default='diagnosis',
                       help='Column name in CSV containing diagnosis labels (default: diagnosis)')
    
    # Optional parameters - Model configuration
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
                       help='Device to use for computation (auto=auto-detect)')
    parser.add_argument('--override_class_names', type=str, default=None,
                       help='Comma-separated class names to use if missing in checkpoint (e.g., "negative,positive")')
    parser.add_argument('--strict_checkpoint', action='store_true',
                       help='Require class names in checkpoint; otherwise fail')
    
    # Optional parameters - Output options
    parser.add_argument('--save_predictions_csv', action='store_true',
                       help='Save predictions summary to CSV file')
    parser.add_argument('--predictions_output_dir', type=str, default=None,
                       help='Directory for predictions CSV (default: same as output_dir)')
    parser.add_argument('--no_save_normalized_attention', action='store_false', dest='save_normalized_attention',
                       default=True, help='Disable saving normalized attention scores (default: enabled)')
    parser.add_argument('--no_save_raw_attention', action='store_false', dest='save_raw_attention',
                       default=True, help='Disable saving raw attention scores (default: enabled)')
    
    # Optional parameters - Processing options
    parser.add_argument('--batch_processing', action='store_true',
                       help='Process multiple slides in batches (experimental)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip slides that already have attention files')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def load_trained_model(model_path, device, override_class_names=None, strict_checkpoint=False):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint
        device (torch.device): Device to load model on
        override_class_names (str, optional): Comma-separated class names
        strict_checkpoint (bool): Whether to require class names in checkpoint
    
    Returns:
        tuple: (model, model_config, class_names, optimal_threshold)
    """
    print(f"Loading trained model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model configuration
    model_config = checkpoint.get('model_config', {})
    class_names = checkpoint.get('class_names', None)
    
    # Handle missing class names gracefully
    if class_names is None:
        if override_class_names:
            class_names = [c.strip() for c in override_class_names.split(',') if c.strip()]
            print(f"⚠️  Class names missing in checkpoint. Using override class names: {class_names}")
        elif strict_checkpoint:
            raise ValueError("Class names not found in checkpoint and strict mode enabled. Provide --override_class_names or retrain with class info.")
        else:
            # Default to binary classification with conventional ordering
            class_names = ['negative', 'positive']
            print("⚠️  Class names not found in checkpoint. Defaulting to binary classes: ['negative', 'positive']")
    
    # Get model parameters
    d_model = model_config.get('d_model', 1024)
    num_classes = len(class_names)
    latent_dim = model_config.get('latent_dim', 256)
    dropout = model_config.get('dropout', 0.0)
    attention_dropout = model_config.get('attention_dropout', 0.0)
    use_layer_norm = model_config.get('use_layer_norm', False)
    use_batch_norm = model_config.get('use_batch_norm', False)
    use_skip_connection = model_config.get('use_skip_connection', False)
    
    print(f"Model configuration:")
    print(f"  - Input dimension: {d_model}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Class names: {class_names}")
    print(f"  - Latent dimension: {latent_dim}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Attention dropout: {attention_dropout}")
    print(f"  - Layer normalization: {use_layer_norm}")
    print(f"  - Batch normalization: {use_batch_norm}")
    print(f"  - Skip connections: {use_skip_connection}")
    
    # Create model instance
    model = GatedMILClassifier(
        d_model=d_model,
        num_classes=num_classes,
        latent_dim=latent_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        use_layer_norm=use_layer_norm,
        use_batch_norm=use_batch_norm,
        use_skip_connection=use_skip_connection
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Check for optimal threshold in checkpoint
    optimal_threshold = checkpoint.get('optimal_threshold', None)
    operating_point_method = checkpoint.get('operating_point_method', None)
    
    if optimal_threshold is not None:
        method_str = f" (method: {operating_point_method})" if operating_point_method else ""
        print(f"✓ Using optimal threshold: {optimal_threshold:.4f}{method_str}")
    else:
        print("✓ No optimal threshold found, using default 0.5 (argmax)")
    
    print("✓ Model loaded successfully")
    return model, model_config, class_names, optimal_threshold


def get_slide_list(h5_directory, input_csv_path=None, slide_id_column='slide_name'):
    """
    Get list of slides to process.
    
    Args:
        h5_directory (str): Directory containing H5 feature files
        input_csv_path (str, optional): Path to CSV file with slide names
        slide_id_column (str): Column name in CSV for slide identifier
    
    Returns:
        tuple: (list of base file names, optional dataframe with metadata)
    """
    if input_csv_path and os.path.exists(input_csv_path):
        print(f"Loading slide list from CSV: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        
        if slide_id_column not in df.columns:
            raise ValueError(f"Column '{slide_id_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Convert slide names to base file names (remove .svs extension if present)
        slide_names = df[slide_id_column].tolist()
        base_file_names = [str(name).replace('.svs', '').replace('.h5', '') for name in slide_names]
        
        print(f"Found {len(base_file_names)} slides in CSV")
        return base_file_names, df
    else:
        # Process all H5 files in directory
        print(f"Processing all H5 files in directory: {h5_directory}")
        import glob
        h5_files = glob.glob(os.path.join(h5_directory, '*.h5'))
        base_file_names = [os.path.splitext(os.path.basename(f))[0] for f in h5_files]
        print(f"Found {len(base_file_names)} H5 files")
        return base_file_names, None


def process_h5_attention(h5_path, model, output_dir, save_raw=True, save_normalized=True, 
                         model_path_attr=None, optimal_threshold=None, class_names=None):
    """
    Process a single H5 file to compute and save attention scores.
    
    Args:
        h5_path (str): Path to input H5 feature file
        model (nn.Module): Trained model
        output_dir (str): Output directory for attention H5 files
        save_raw (bool): Whether to save raw attention scores
        save_normalized (bool): Whether to save normalized attention scores
        model_path_attr (str, optional): Model path to save as attribute
        optimal_threshold (float, optional): Optimal threshold for binary classification
        class_names (list, optional): List of class names
    
    Returns:
        tuple: (prediction, probability, success_flag)
    """
    try:
        # Extract base filename for output
        base_filename = os.path.splitext(os.path.basename(h5_path))[0]
        output_h5_path = os.path.join(output_dir, f"{base_filename}.h5")
        
        # Read the features from the h5 file
        with h5py.File(h5_path, 'r') as f:
            # Load features
            F_features = f['features'][:]
            # Get coordinates and attributes
            coords = f['coords'][:]
            
            # Get all attributes from the original file
            coords_attrs = dict(f['coords'].attrs)
        
        # Convert F_features to tensor
        F_features = torch.from_numpy(F_features).float()
        # Add a batch dimension
        F_features = F_features.unsqueeze(0)
        
        # Move to device
        device = next(model.parameters()).device
        F_features = F_features.to(device)
        
        # Compute attention and predictions
        model.eval()
        with torch.no_grad():
            attention_scores = model.compute_attention(F_features)
            attention_scores = attention_scores.cpu().detach().numpy()
            attention_scores = attention_scores.squeeze()
            
            # Forward pass for predictions
            outputs = model(F_features)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # Use threshold-based prediction for binary classification if threshold is available
            if optimal_threshold is not None and class_names is not None and len(class_names) == 2:
                # Binary classification with custom threshold
                prob_positive = probs[0, 1]
                pred = 1 if prob_positive >= optimal_threshold else 0
            else:
                # Default behavior: argmax (works for multi-class and default threshold)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                pred = int(preds[0])
                # For binary classification, get positive class probability
                if probs.shape[1] == 2:
                    prob_positive = probs[0, 1]
                else:
                    prob_positive = probs[0, pred]
            
            pred = int(pred)
            prob = float(prob_positive)
        
        # Normalize attention scores (min-max normalization)
        if save_normalized:
            if attention_scores.max() > attention_scores.min():
                normalized_attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min())
            else:
                normalized_attention_scores = np.ones_like(attention_scores) / len(attention_scores)
        
        # Save to H5 file with same structure as features file
        with h5py.File(output_h5_path, 'w') as f:
            # Save coordinates (copy from original)
            coords_dset = f.create_dataset('coords', data=coords)
            
            # Copy all original attributes to coords dataset
            for key, value in coords_attrs.items():
                try:
                    coords_dset.attrs[key] = value
                except (TypeError, ValueError):
                    # Skip attributes that can't be serialized
                    pass
            
            # Save raw attention scores
            if save_raw:
                attention_dset = f.create_dataset('attention', data=attention_scores.astype(np.float32))
                attention_dset.attrs['name'] = coords_attrs.get('name', base_filename)
                attention_dset.attrs['savetodir'] = output_dir
                attention_dset.attrs['normalized'] = False
            
            # Save normalized attention scores
            if save_normalized:
                normalized_attention_dset = f.create_dataset('attention_normalized', 
                                                             data=normalized_attention_scores.astype(np.float32))
                normalized_attention_dset.attrs['name'] = coords_attrs.get('name', base_filename)
                normalized_attention_dset.attrs['savetodir'] = output_dir
                normalized_attention_dset.attrs['normalized'] = True
            
            # Save prediction and probability as attributes
            f.attrs['prediction'] = pred
            f.attrs['probability'] = prob
            if model_path_attr:
                f.attrs['model_path'] = model_path_attr
        
        return pred, prob, True
        
    except Exception as e:
        print(f"Error processing {os.path.basename(h5_path)}: {str(e)}")
        return None, None, False


def process_slides(h5_directory, base_file_names, model, output_dir, metadata_df=None,
                   slide_id_column='slide_name', diagnosis_column='diagnosis',
                   save_raw=True, save_normalized=True, skip_existing=False,
                   model_path_attr=None, optimal_threshold=None, class_names=None):
    """
    Process multiple slides and collect results.
    
    Args:
        h5_directory (str): Directory containing H5 feature files
        base_file_names (list): List of base file names to process
        model (nn.Module): Trained model
        output_dir (str): Output directory
        metadata_df (pd.DataFrame, optional): DataFrame with metadata
        slide_id_column (str): Column name for slide identifier
        diagnosis_column (str): Column name for diagnosis
        save_raw (bool): Whether to save raw attention
        save_normalized (bool): Whether to save normalized attention
        skip_existing (bool): Whether to skip existing files
        model_path_attr (str, optional): Model path for attributes
        optimal_threshold (float, optional): Optimal threshold for binary classification
        class_names (list, optional): List of class names
    
    Returns:
        pd.DataFrame: Results dataframe
    """
    successful_base_file_names = []
    predictions = []
    probabilities = []
    diagnosis_labels = []
    
    print(f"\nProcessing {len(base_file_names)} slides...")
    
    for base_file_name in tqdm(base_file_names, desc="Computing attention"):
        h5_path = os.path.join(h5_directory, f"{base_file_name}.h5")
        
        # Check if file exists
        if not os.path.exists(h5_path):
            print(f"⚠️  Warning: H5 file not found: {h5_path}")
            continue
        
        # Check if output already exists
        if skip_existing:
            output_h5_path = os.path.join(output_dir, f"{base_file_name}.h5")
            if os.path.exists(output_h5_path):
                print(f"⏭️  Skipping {base_file_name} (already exists)")
                continue
        
        # Process the file
        pred, prob, success = process_h5_attention(
            h5_path, model, output_dir, 
            save_raw=save_raw, save_normalized=save_normalized,
            model_path_attr=model_path_attr,
            optimal_threshold=optimal_threshold,
            class_names=class_names
        )
        
        if success:
            predictions.append(pred)
            probabilities.append(prob)
            successful_base_file_names.append(base_file_name)
                
            # Get diagnosis label if metadata is available
            if metadata_df is not None:
                # Try to match slide name (with or without .svs extension)
                svs_file_name = base_file_name + ".svs"
                matching_rows = metadata_df[metadata_df[slide_id_column] == svs_file_name]
                if len(matching_rows) == 0:
                    # Try without extension
                    matching_rows = metadata_df[metadata_df[slide_id_column] == base_file_name]
                
                if len(matching_rows) > 0 and diagnosis_column in matching_rows.columns:
                    diagnosis_labels.append(matching_rows[diagnosis_column].iloc[0])
                else:
                    diagnosis_labels.append("Unknown")
            else:
                diagnosis_labels.append("N/A")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'base_file_name': successful_base_file_names,
        'prediction': predictions,
        'probability': probabilities
    })
    
    if len(diagnosis_labels) > 0:
        results_df['diagnosis'] = diagnosis_labels
    
    # Convert predictions to class names if available
    if len(set(predictions)) <= 2:  # Binary classification
        results_df['prediction_label'] = results_df['prediction'].apply(
            lambda x: 'positive' if x == 1 else 'negative'
        )
    
    return results_df


def save_results(results_df, output_dir, predictions_output_dir=None):
    """
    Save results to CSV file.
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        output_dir (str): Main output directory
        predictions_output_dir (str, optional): Specific directory for predictions CSV
    """
    if predictions_output_dir is None:
        predictions_output_dir = output_dir
    
    os.makedirs(predictions_output_dir, exist_ok=True)
    output_path = os.path.join(predictions_output_dir, 'predictions.csv')
    results_df.to_csv(output_path, index=False)
    print(f"✓ Predictions summary saved to: {output_path}")


def save_processing_summary(args, results_df, output_dir):
    """
    Save processing summary and configuration.
    
    Args:
        args: Parsed arguments
        results_df (pd.DataFrame): Results dataframe
        output_dir (str): Output directory
    """
    summary = {
        'total_slides_processed': len(results_df),
        'successful_predictions': len(results_df),
        'configuration': {
            'model_path': args.model_path,
            'h5_directory': args.h5_directory,
            'output_dir': args.output_dir,
            'input_csv_path': args.input_csv_path,
            'device': str(args.device),
            'save_raw_attention': args.save_raw_attention,
            'save_normalized_attention': args.save_normalized_attention,
            'skip_existing': args.skip_existing
        },
        'statistics': {
            'mean_probability': float(results_df['probability'].mean()),
            'std_probability': float(results_df['probability'].std()),
            'min_probability': float(results_df['probability'].min()),
            'max_probability': float(results_df['probability'].max())
        }
    }
    
    if 'prediction' in results_df.columns:
        summary['statistics']['prediction_distribution'] = results_df['prediction'].value_counts().to_dict()
    
    summary_path = os.path.join(output_dir, 'processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Processing summary saved to: {summary_path}")


def main():
    """Main attention computation function."""
    args = parse_arguments()
    
    # Set seed for reproducibility
    set_seed(args.random_state)
    
    print("=" * 80)
    print("ATTENTION COMPUTATION")
    print("=" * 80)
    print("Computing attention scores for slides using trained model")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load trained model
    model, model_config, class_names, optimal_threshold = load_trained_model(
        args.model_path,
        device,
        override_class_names=args.override_class_names,
        strict_checkpoint=args.strict_checkpoint
    )
    
    # Get list of slides to process
    base_file_names, metadata_df = get_slide_list(
        args.h5_directory,
        args.input_csv_path,
        args.slide_id_column
    )
    
    if len(base_file_names) == 0:
        print("⚠️  No slides found to process. Exiting.")
        return
    
    # Process slides
    results_df = process_slides(
        h5_directory=args.h5_directory,
        base_file_names=base_file_names,
        model=model,
        output_dir=args.output_dir,
        metadata_df=metadata_df,
        slide_id_column=args.slide_id_column,
        diagnosis_column=args.diagnosis_column,
        save_raw=args.save_raw_attention,
        save_normalized=args.save_normalized_attention,
        skip_existing=args.skip_existing,
        model_path_attr=args.model_path,
        optimal_threshold=optimal_threshold,
        class_names=class_names
    )
    
    # Save results if requested
    if args.save_predictions_csv:
        save_results(results_df, args.output_dir, args.predictions_output_dir)
    
    # Save processing summary
    save_processing_summary(args, results_df, args.output_dir)
    
    # Final summary
    print(f"\n" + "=" * 80)
    print("ATTENTION COMPUTATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"✓ Processed {len(results_df)} slides successfully")
    print(f"✓ Attention H5 files saved to: {args.output_dir}")
    if args.save_predictions_csv:
        predictions_dir = args.predictions_output_dir or args.output_dir
        print(f"✓ Predictions summary saved to: {predictions_dir}")
    print(f"✓ Mean prediction probability: {results_df['probability'].mean():.4f}")
    if 'diagnosis' in results_df.columns:
        print(f"✓ Slides with diagnosis labels: {(results_df['diagnosis'] != 'Unknown').sum()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
