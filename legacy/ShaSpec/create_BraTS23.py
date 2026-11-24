import os
import csv
import random

random.seed(42)
# Dataset path
data_dir = "/path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
all_cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

total_count = len(all_cases)
if total_count == 0:
    raise ValueError("No cases found in the specified directory.")

train_count = int(0.7 * total_count)
val_count = int(0.1 * total_count)
test_count = total_count - train_count - val_count

# Group by patient ID (everything except the last four characters)
# (e.g., "BraTS-GLI-00703-000" -> patient ID: "BraTS-GLI-00703")
patient_dict = {}
for case in all_cases:
    parts = case.split('-')
    patient_id = '-'.join(parts[:-1])  # "BraTS-GLI-00703"
    if patient_id not in patient_dict:
        patient_dict[patient_id] = []
    patient_dict[patient_id].append(case)

# Separate multi-scan and single-scan patients
multi_scan_patients = {p: c for p, c in patient_dict.items() if len(c) > 1}
single_scan_patients = {p: c for p, c in patient_dict.items() if len(c) == 1}

train_set = []
val_set = []
test_set = []

# Add all multi-scan patient cases to training
for p, c in multi_scan_patients.items():
    train_set.extend(c)

multi_count = len(train_set)
if multi_count > train_count:
    raise ValueError("Cannot achieve the desired 70% training split because multi-scan patients alone exceed 70% of total cases.")

single_scan_cases = [c[0] for c in single_scan_patients.values()]
random.shuffle(single_scan_cases)

needed_for_train = train_count - multi_count

if needed_for_train > 0:
    to_train = single_scan_cases[:needed_for_train]
    train_set.extend(to_train)
    single_scan_cases = single_scan_cases[needed_for_train:]

if len(single_scan_cases) < val_count:
    raise ValueError("Not enough cases left for the desired validation split.")

val_set = single_scan_cases[:val_count]
remaining_after_val = single_scan_cases[val_count:]

if len(remaining_after_val) < test_count:
    raise ValueError("Not enough cases left for the desired test split.")

test_set = remaining_after_val[:test_count]

assert len(train_set) == train_count, f"Train set size mismatch: {len(train_set)} vs {train_count}"
assert len(val_set) == val_count, f"Val set size mismatch: {len(val_set)} vs {val_count}"
assert len(test_set) == test_count, f"Test set size mismatch: {len(test_set)} vs {test_count}"

train_set.sort()
val_set.sort()
test_set.sort()

# Write out the CSV files
with open("train.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case"])
    for case in train_set:
        writer.writerow([os.path.join(data_dir, case)])

with open("val.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case"])
    for case in val_set:
        writer.writerow([os.path.join(data_dir, case)])

with open("test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case"])
    for case in test_set:
        writer.writerow([os.path.join(data_dir, case)])

print("CSV files (train.csv, val.csv, test.csv) have been generated successfully.")