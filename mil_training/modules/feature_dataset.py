import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import glob
import os
import torch.nn.functional as F
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split

class FeatureDataset(Dataset):
    def __init__(self, h5_directory, master_df, allowed_slide_names=None, class_names=None):
        """
        Dataset for loading features from H5 files.

        Args:
            h5_directory (str): Directory containing H5 files
            master_df (pd.DataFrame): DataFrame containing diagnosis labels based on slide_name
            allowed_slide_names (set, optional): Set of slide names to include. If None, includes all slides in master_df.
            class_names (list, optional): Explicit label ordering to use for the diagnosis->index
                mapping. When provided (e.g. from a trained checkpoint during test evaluation),
                this ordering is used instead of sorted(unique(diagnosis)). This guarantees that
                test-set label indices match the class ordering the model was trained on, which
                matters for multiclass tasks where the test set may not contain every class.
        """
        # Get unique diagnoses and create label mapping
        if class_names is not None:
            unique_diagnoses = list(class_names)
            present = set(master_df['diagnosis'].unique())
            unknown = present - set(unique_diagnoses)
            if unknown:
                raise ValueError(
                    f"Test set contains diagnosis labels not present in the provided class_names "
                    f"{unique_diagnoses}: {sorted(unknown)}. Fix the test CSV label mapping."
                )
        else:
            unique_diagnoses = sorted(master_df['diagnosis'].unique())
        self.diagnosis_to_label = {diagnosis: idx for idx, diagnosis in enumerate(unique_diagnoses)}
        self.label_to_diagnosis = {idx: diagnosis for idx, diagnosis in enumerate(unique_diagnoses)}
        
        print(f"Found {len(unique_diagnoses)} unique diagnoses:")
        for diagnosis, label in self.diagnosis_to_label.items():
            count = (master_df['diagnosis'] == diagnosis).sum()
            print(f"  {diagnosis}: {count} slides (label: {label})")
        
        # Create a dictionary mapping slide_name to diagnosis label
        self.label_dict = dict(zip(master_df['slide_name'], 
                                 master_df['diagnosis'].map(self.diagnosis_to_label)))
        
        # Get all available H5 files
        all_h5_files = glob.glob(os.path.join(h5_directory, '*.h5'))
        if not all_h5_files:
            raise ValueError(f"No H5 files found in {h5_directory}")
        
        # Filter H5 files to only include those present in master_df
        self.h5_files = []
        available_slide_names = set(master_df['slide_name'])
        
        # If allowed_slide_names is provided, further filter by it
        if allowed_slide_names is not None:
            available_slide_names = available_slide_names.intersection(allowed_slide_names)
        
        for h5_file in all_h5_files:
            # Extract slide name from H5 file
            filename = os.path.basename(h5_file)
            slide_name = filename.replace('.h5', '.svs')
            
            # Only include if slide_name exists in available_slide_names
            if slide_name in available_slide_names:
                self.h5_files.append(h5_file)
        
        if not self.h5_files:
            raise ValueError(f"No H5 files found that match slide_names in master_df")
        
        print(f"Found {len(self.h5_files)} H5 files matching master_df out of {len(all_h5_files)} total H5 files")
        
        # Print class distribution for this dataset
        if allowed_slide_names is not None:
            filtered_df = master_df[master_df['slide_name'].isin(available_slide_names)]
            class_counts = filtered_df['diagnosis'].value_counts()
            print("Class distribution in this dataset:")
            for diagnosis in unique_diagnoses:
                count = class_counts.get(diagnosis, 0)
                print(f"  {diagnosis}: {count} slides")
    
    def __len__(self):
        return len(self.h5_files)
    
    def __getitem__(self, idx):
        h5_file = self.h5_files[idx]
        # extract the filename from the h5_file filename
        filename = os.path.basename(h5_file)
        # drop the h5 extension add .svs
        slide_name = filename.replace('.h5', '.svs')
        # get the diagnosis label for the slide_name
        diagnosis = self.label_dict[slide_name]
        
        with h5py.File(h5_file, 'r') as f:
            # Load features
            features = f['features'][:]
            features = torch.from_numpy(features)
            
            return features, diagnosis

def collate_fn(batch):
    """
    Custom collate function for dynamic padding within each batch.
    
    Args:
        batch: List of (features, diagnosis) tuples
        
    Returns:
        Tuple of (padded_features_batch, diagnosis_batch)
    """
    features_list, diagnosis_list = zip(*batch)
    
    # Find the maximum number of features in this batch
    max_features_in_batch = max(features.shape[0] for features in features_list)
    
    # Pad each feature tensor to max_features_in_batch
    padded_features = []
    for features in features_list:
        padded = F.pad(features, (0, 0, 0, max_features_in_batch - features.shape[0]))
        padded_features.append(padded)
    
    # Stack the padded features and diagnoses
    features_batch = torch.stack(padded_features)
    diagnosis_batch = torch.tensor(diagnosis_list)
    
    return features_batch, diagnosis_batch

def patient_level_split(master_df, val_split=0.2, test_split=0.1, random_state=42, 
                       stratify_by_slide_distribution=True, max_iterations=100):
    """
    Split the dataset at the patient level to avoid data leakage.
    
    Args:
        master_df (pd.DataFrame): DataFrame containing slide_name, MRN, and diagnosis columns
        val_split (float): Fraction of patients for validation
        test_split (float): Fraction of patients for test set
        random_state (int): Random seed for reproducible splits
        stratify_by_slide_distribution (bool): Whether to try to maintain slide-level diagnosis distribution
        max_iterations (int): Maximum iterations to try for balanced splits
        
    Returns:
        tuple: (train_slides, val_slides, test_slides) - sets of slide names for each split
    """
    # Analyze patient-level diagnosis patterns
    patient_diagnosis_stats = master_df.groupby('MRN')['diagnosis'].agg(['nunique', 'count']).reset_index()
    patient_diagnosis_stats.columns = ['MRN', 'unique_diagnoses', 'slide_count']
    
    # Identify patients with mixed diagnoses
    mixed_diagnosis_patients = patient_diagnosis_stats[patient_diagnosis_stats['unique_diagnoses'] > 1]['MRN'].tolist()
    consistent_diagnosis_patients = patient_diagnosis_stats[patient_diagnosis_stats['unique_diagnoses'] == 1]['MRN'].tolist()
    
    print(f"Total patients: {len(patient_diagnosis_stats)}")
    print(f"Patients with consistent diagnoses: {len(consistent_diagnosis_patients)}")
    print(f"Patients with mixed diagnoses: {len(mixed_diagnosis_patients)}")
    
    # Get slide-level diagnosis distribution
    slide_diagnosis_counts = master_df['diagnosis'].value_counts()
    total_slides = len(master_df)
    target_class_distribution = slide_diagnosis_counts / total_slides
    
    print(f"Slide-level diagnosis distribution:")
    print(slide_diagnosis_counts)
    print(f"Target class distribution:")
    for diagnosis, proportion in target_class_distribution.items():
        print(f"  {diagnosis}: {proportion:.3f}")
    
    # Get all unique patients
    all_patients = master_df['MRN'].unique()
    
    if not stratify_by_slide_distribution or len(mixed_diagnosis_patients) > len(consistent_diagnosis_patients) * 0.3:
        # If too many patients have mixed diagnoses, just do random splits
        print("Using random patient splits (no stratification due to mixed diagnoses)")
        
        # Handle special case where test_split=1.0 (all data should be test data)
        if test_split >= 1.0:
            train_patients = np.array([])
            val_patients = np.array([])
            test_patients = all_patients
        else:
            # First split: separate out test set
            train_val_patients, test_patients = train_test_split(
                all_patients,
                test_size=test_split,
                random_state=random_state
            )
            
            # Second split: separate train and validation
            if val_split > 0:
                val_size_adjusted = val_split / (1 - test_split)
                train_patients, val_patients = train_test_split(
                    train_val_patients,
                    test_size=val_size_adjusted,
                    random_state=random_state
                )
            else:
                train_patients = train_val_patients
                val_patients = np.array([])
            
    else:
        # Try to balance slide-level diagnosis distribution across classes
        print("Attempting to balance slide-level diagnosis distribution across all classes...")
        
        # Handle special case where test_split=1.0 (all data should be test data)
        if test_split >= 1.0:
            train_patients = np.array([])
            val_patients = np.array([])
            test_patients = all_patients
        else:
            best_split = None
            best_balance_score = float('inf')
            
            np.random.seed(random_state)
            
            for iteration in range(max_iterations):
                # Shuffle patients
                shuffled_patients = np.random.permutation(all_patients)
                
                # Split patients
                n_test = int(len(shuffled_patients) * test_split)
                n_val = int(len(shuffled_patients) * val_split) if val_split > 0 else 0
                
                test_patients = shuffled_patients[:n_test]
                val_patients = shuffled_patients[n_test:n_test+n_val] if n_val > 0 else np.array([])
                train_patients = shuffled_patients[n_test+n_val:]
            
                # Calculate slide-level diagnosis distribution for this split
                train_slides_temp = master_df[master_df['MRN'].isin(train_patients)]
                val_slides_temp = master_df[master_df['MRN'].isin(val_patients)] if len(val_patients) > 0 else pd.DataFrame()
                test_slides_temp = master_df[master_df['MRN'].isin(test_patients)]
                
                # Calculate balance score across all classes
                balance_score = 0
                
                # Calculate for train set
                if len(train_slides_temp) > 0:
                    train_class_dist = train_slides_temp['diagnosis'].value_counts() / len(train_slides_temp)
                    for diagnosis in target_class_distribution.index:
                        train_prop = train_class_dist.get(diagnosis, 0)
                        target_prop = target_class_distribution[diagnosis]
                        balance_score += abs(train_prop - target_prop)
                
                # Calculate for validation set
                if len(val_slides_temp) > 0:
                    val_class_dist = val_slides_temp['diagnosis'].value_counts() / len(val_slides_temp)
                    for diagnosis in target_class_distribution.index:
                        val_prop = val_class_dist.get(diagnosis, 0)
                        target_prop = target_class_distribution[diagnosis]
                        balance_score += abs(val_prop - target_prop)
                
                # Calculate for test set
                if test_split > 0 and len(test_slides_temp) > 0:
                    test_class_dist = test_slides_temp['diagnosis'].value_counts() / len(test_slides_temp)
                    for diagnosis in target_class_distribution.index:
                        test_prop = test_class_dist.get(diagnosis, 0)
                        target_prop = target_class_distribution[diagnosis]
                        balance_score += abs(test_prop - target_prop)
                
                if balance_score < best_balance_score:
                    best_balance_score = balance_score
                    best_split = (train_patients, val_patients, test_patients)
                    
                    # If we found a very good split, stop early
                    # For 8 classes, we expect higher balance scores, so adjust threshold
                    threshold = 0.4 if test_split > 0 else 0.3
                    if balance_score < threshold:
                        break
            
            train_patients, val_patients, test_patients = best_split
            print(f"Best balance score after {min(iteration+1, max_iterations)} iterations: {best_balance_score:.4f}")
    
    print(f"Train patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")
    
    # Convert patient splits to slide splits
    train_slides = set(master_df[master_df['MRN'].isin(train_patients)]['slide_name'])
    val_slides = set(master_df[master_df['MRN'].isin(val_patients)]['slide_name']) if len(val_patients) > 0 else set()
    test_slides = set(master_df[master_df['MRN'].isin(test_patients)]['slide_name'])
    
    print(f"Train slides: {len(train_slides)}")
    print(f"Validation slides: {len(val_slides)}")
    print(f"Test slides: {len(test_slides)}")
    
    # Report actual slide-level diagnosis distribution
    train_df = master_df[master_df['slide_name'].isin(train_slides)]
    val_df = master_df[master_df['slide_name'].isin(val_slides)] if val_slides else pd.DataFrame()
    test_df = master_df[master_df['slide_name'].isin(test_slides)] if test_slides else pd.DataFrame()
    
    print("Actual slide-level diagnosis distribution:")
    print(f"Train: {train_df['diagnosis'].value_counts().to_dict()}")
    if len(val_df) > 0:
        print(f"Val: {val_df['diagnosis'].value_counts().to_dict()}")
    if len(test_df) > 0:
        print(f"Test: {test_df['diagnosis'].value_counts().to_dict()}")
    
    # Show class proportions
    print("\nClass proportions:")
    if len(train_df) > 0:
        train_proportions = train_df['diagnosis'].value_counts() / len(train_df)
        print(f"Train proportions: {train_proportions.to_dict()}")
    if len(val_df) > 0:
        val_proportions = val_df['diagnosis'].value_counts() / len(val_df)
        print(f"Val proportions: {val_proportions.to_dict()}")
    if len(test_df) > 0:
        test_proportions = test_df['diagnosis'].value_counts() / len(test_df)
        print(f"Test proportions: {test_proportions.to_dict()}")
    
    # Verify no patient appears in multiple splits
    train_patient_set = set(master_df[master_df['slide_name'].isin(train_slides)]['MRN'])
    val_patient_set = set(master_df[master_df['slide_name'].isin(val_slides)]['MRN'])
    test_patient_set = set(master_df[master_df['slide_name'].isin(test_slides)]['MRN'])
    
    assert len(train_patient_set.intersection(val_patient_set)) == 0, "Patient overlap between train and val!"
    if test_slides:  # Only check test overlaps if test set exists
        assert len(train_patient_set.intersection(test_patient_set)) == 0, "Patient overlap between train and test!"
        assert len(val_patient_set.intersection(test_patient_set)) == 0, "Patient overlap between val and test!"
    
    print("✓ Verified no patient overlap between splits")
    
    return train_slides, val_slides, test_slides

def create_patient_level_datasets(h5_directory, master_df, val_split=0.2, test_split=0.1, random_state=42,
                                stratify_by_slide_distribution=True, max_iterations=100):
    """
    Create train, validation, and test datasets with patient-level splits.
    
    Args:
        h5_directory (str): Directory containing H5 files
        master_df (pd.DataFrame): DataFrame containing slide_name, MRN, and diagnosis columns
        val_split (float): Fraction of patients for validation
        test_split (float): Fraction of patients for test set
        random_state (int): Random seed for reproducible splits
        stratify_by_slide_distribution (bool): Whether to try to maintain slide-level diagnosis distribution
        max_iterations (int): Maximum iterations to try for balanced splits
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_slides, val_slides, test_slides = patient_level_split(
        master_df, val_split=val_split, test_split=test_split, random_state=random_state,
        stratify_by_slide_distribution=stratify_by_slide_distribution, max_iterations=max_iterations
    )
    
    train_dataset = FeatureDataset(h5_directory, master_df, allowed_slide_names=train_slides) if train_slides else None
    val_dataset = FeatureDataset(h5_directory, master_df, allowed_slide_names=val_slides) if val_slides else None
    test_dataset = FeatureDataset(h5_directory, master_df, allowed_slide_names=test_slides) if test_slides else None
    
    return train_dataset, val_dataset, test_dataset

def create_patient_level_dataloaders(h5_directory, master_df, batch_size=1, num_workers=4, 
                                   val_split=0.2, test_split=0.1, random_state=42,
                                   stratify_by_slide_distribution=True, max_iterations=100):
    """
    Create DataLoaders for train, validation, and test sets with patient-level splits.
    
    Args:
        h5_directory (str): Directory containing H5 files
        master_df (pd.DataFrame): DataFrame containing slide_name, MRN, and diagnosis columns
        batch_size (int): Number of H5 files to process in each batch
        num_workers (int): Number of worker processes for data loading
        val_split (float): Fraction of patients for validation
        test_split (float): Fraction of patients for test set
        random_state (int): Random seed for reproducible splits
        stratify_by_slide_distribution (bool): Whether to try to maintain slide-level diagnosis distribution
        max_iterations (int): Maximum iterations to try for balanced splits
        
    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    train_dataset, val_dataset, test_dataset = create_patient_level_datasets(
        h5_directory, master_df, val_split=val_split, test_split=test_split, random_state=random_state,
        stratify_by_slide_distribution=stratify_by_slide_distribution, max_iterations=max_iterations
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True  # Prevent single-sample batches for batch normalization
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    ) if val_dataset else None
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    ) if test_dataset else None
    
    return train_dataloader, val_dataloader, test_dataloader


def create_test_only_dataloader(h5_directory, test_df, batch_size=1, num_workers=4, class_names=None):
    """
    Create a DataLoader for test evaluation only, without requiring train/val splits.

    Args:
        h5_directory (str): Directory containing H5 files
        test_df (pd.DataFrame): DataFrame containing slide_name and diagnosis columns for test set
        batch_size (int): Number of H5 files to process in each batch
        num_workers (int): Number of worker processes for data loading
        class_names (list, optional): Explicit label ordering (e.g. from the trained checkpoint)
            so test label indices match the model's training class order. Important for multiclass.

    Returns:
        tuple: (test_dataloader, test_df, slide_identifiers)
    """
    print(f"Creating test-only dataloader from test CSV")

    # Create test dataset directly from all slides in test_df
    test_slide_names = set(test_df['slide_name'].unique())
    test_dataset = FeatureDataset(h5_directory, test_df, allowed_slide_names=test_slide_names,
                                  class_names=class_names)
    
    # Create DataLoader with custom collate function for variable-sized tensors
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Don't shuffle for evaluation
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Extract slide identifiers in the same order as the dataset (dataloader iteration order)
    slide_identifiers = [os.path.basename(f).replace('.h5', '.svs') for f in test_dataset.h5_files]
    
    print(f"✓ Test dataloader created with {len(test_dataset)} samples")
    
    return test_dataloader, test_df, slide_identifiers

def create_feature_dataloader(h5_directory, master_df, batch_size=1, num_workers=4):
    """
    Create a DataLoader for the FeatureDataset with dynamic padding (backward compatibility).
    
    Args:
        h5_directory (str): Directory containing H5 files
        master_df (pd.DataFrame): DataFrame containing diagnosis labels based on slide_name
        batch_size (int): Number of H5 files to process in each batch
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        DataLoader: DataLoader instance for the FeatureDataset with dynamic padding
    """
    dataset = FeatureDataset(h5_directory, master_df)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

if __name__ == "__main__":
    raise SystemExit(
        "This module is imported by train_with_balanced_options.py; "
        "it is not a standalone script."
    )

