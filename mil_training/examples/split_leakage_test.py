import os
import sys
import numpy as np
import pandas as pd
import random
from typing import List, Tuple
import importlib.util


def load_module_from_path(module_name: str, file_path: str):
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	module = importlib.util.module_from_spec(spec)
	assert spec is not None and spec.loader is not None
	spec.loader.exec_module(module)
	return module


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODULES_DIR = os.path.join(REPO_ROOT, "modules")

# Load split implementation directly from current module
feature_dataset_module = load_module_from_path(
	"feature_dataset",
	os.path.join(MODULES_DIR, "feature_dataset.py"),
)

# The current implementation works for both binary and multiclass
patient_level_split = getattr(feature_dataset_module, "patient_level_split")


def seed_all(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)


def generate_synthetic_master_df_multiclass(
		num_patients: int = 50,
		slides_per_patient_range: Tuple[int, int] = (1, 5),
		diagnoses: List[str] = None,
		mixed_patient_fraction: float = 0.1,
		seed: int = 42,
) -> pd.DataFrame:
	"""
	Generate a synthetic master_df for multi-class tasks with optional mixed-diagnosis patients.
	Columns: slide_name, MRN, diagnosis, DIAGNOSIS (uppercase copy for binary code compatibility)
	"""
	seed_all(seed)
	if diagnoses is None:
		diagnoses = ["A", "B", "C", "D"]

	rows = []
	for patient_idx in range(num_patients):
		mrn = f"P{patient_idx:04d}"
		n_slides = np.random.randint(slides_per_patient_range[0], slides_per_patient_range[1] + 1)

		is_mixed = np.random.rand() < mixed_patient_fraction
		if is_mixed and len(diagnoses) > 1:
			# Ensure at least two diagnoses
			patient_diags = np.random.choice(diagnoses, size=n_slides, replace=True)
			# force mix
			patient_diags[0] = diagnoses[0]
			patient_diags[-1] = diagnoses[-1]
		else:
			patient_diag = np.random.choice(diagnoses)
			patient_diags = np.array([patient_diag] * n_slides)

		for slide_idx in range(n_slides):
			slide_name = f"S_{mrn}_{slide_idx:03d}"
			rows.append({
				"slide_name": slide_name,
				"MRN": mrn,
				"diagnosis": patient_diags[slide_idx],
				"DIAGNOSIS": patient_diags[slide_idx],
			})

	return pd.DataFrame(rows)


def generate_synthetic_master_df_binary(
		num_patients: int = 50,
		slides_per_patient_range: Tuple[int, int] = (1, 5),
		positive_rate: float = 0.5,
		mixed_patient_fraction: float = 0.1,
		seed: int = 42,
) -> pd.DataFrame:
	"""
	Generate a synthetic master_df for binary tasks with optional mixed-diagnosis patients.
	Columns: slide_name, MRN, DIAGNOSIS (values: 'positive'/'negative'), diagnosis (lowercase copy for multiclass code compatibility)
	"""
	seed_all(seed)
	rows = []
	for patient_idx in range(num_patients):
		mrn = f"P{patient_idx:04d}"
		n_slides = np.random.randint(slides_per_patient_range[0], slides_per_patient_range[1] + 1)

		is_mixed = np.random.rand() < mixed_patient_fraction
		if is_mixed:
			patient_labels = np.random.choice(["positive", "negative"], size=n_slides, p=[positive_rate, 1 - positive_rate])
			# force mix
			patient_labels[0] = "positive"
			patient_labels[-1] = "negative"
		else:
			label = "positive" if np.random.rand() < positive_rate else "negative"
			patient_labels = np.array([label] * n_slides)

		for slide_idx in range(n_slides):
			slide_name = f"S_{mrn}_{slide_idx:03d}"
			rows.append({
				"slide_name": slide_name,
				"MRN": mrn,
				"DIAGNOSIS": patient_labels[slide_idx],
				"diagnosis": patient_labels[slide_idx],
			})

	return pd.DataFrame(rows)


def verify_no_patient_overlap(master_df: pd.DataFrame, train_slides: set, val_slides: set, test_slides: set) -> None:
	train_patient_set = set(master_df[master_df['slide_name'].isin(train_slides)]['MRN'])
	val_patient_set = set(master_df[master_df['slide_name'].isin(val_slides)]['MRN'])
	test_patient_set = set(master_df[master_df['slide_name'].isin(test_slides)]['MRN'])

	assert len(train_patient_set.intersection(val_patient_set)) == 0, "Patient overlap between train and val!"
	if len(test_slides) > 0:
		assert len(train_patient_set.intersection(test_patient_set)) == 0, "Patient overlap between train and test!"
		assert len(val_patient_set.intersection(test_patient_set)) == 0, "Patient overlap between val and test!"


def run_multiclass_trials(seeds: List[int]) -> None:
	print("\n=== Multi-class trials ===")
	for seed in seeds:
		print(f"\nSeed {seed}")
		master_df = generate_synthetic_master_df_multiclass(
			num_patients=80,
			slides_per_patient_range=(1, 6),
			diagnoses=["A", "B", "C"],
			mixed_patient_fraction=0.2,
			seed=seed,
		)
		train_slides, val_slides, test_slides = patient_level_split(
			master_df=master_df,
			val_split=0.2,
			test_split=0.1,
			random_state=seed,
			stratify_by_slide_distribution=True,
			max_iterations=200,
		)
		verify_no_patient_overlap(master_df, train_slides, val_slides, test_slides)
		print("OK: no patient overlap")


def run_binary_trials(seeds: List[int]) -> None:
	print("\n=== Binary trials ===")
	for seed in seeds:
		print(f"\nSeed {seed}")
		master_df = generate_synthetic_master_df_binary(
			num_patients=80,
			slides_per_patient_range=(1, 6),
			positive_rate=0.55,
			mixed_patient_fraction=0.2,
			seed=seed,
		)
		train_slides, val_slides, test_slides = patient_level_split(
			master_df=master_df,
			val_split=0.2,
			test_split=0.1,
			random_state=seed,
			stratify_by_slide_distribution=True,
			max_iterations=200,
		)
		verify_no_patient_overlap(master_df, train_slides, val_slides, test_slides)
		print("OK: no patient overlap")


def main() -> None:
	seeds = [0, 1, 2, 3, 4]
	run_multiclass_trials(seeds)
	run_binary_trials(seeds)
	print("\nAll trials completed.")


if __name__ == "__main__":
	main()
