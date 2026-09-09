"""
SIH26012 Phase 1 Automated Verification Runner.
Executes test suite, runs pipeline, generates visualizations, and prints Phase 1 verification summary.
"""

import os
import sys
import json
import time
import subprocess
import yaml
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_command(cmd_list, description):
    print(f"\n>>> Running: {description} <<<")
    print(f"Command: {' '.join(cmd_list)}")
    t0 = time.time()
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"FAILED (Exit Code {res.returncode}) in {elapsed:.2f}s:")
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"Step '{description}' failed!")
    else:
        print(f"PASSED in {elapsed:.2f}s")
        if res.stdout:
            print(res.stdout)
    return res


def main():
    print("==================================================")
    print(" SIH26012 — PHASE 1 COMPREHENSIVE VERIFICATION")
    print("==================================================")
    
    python_bin = sys.executable
    pytest_bin = os.path.join(os.path.dirname(python_bin), "pytest")

    # Step 1: Run initial pre-flight tests (files, CRS, schema)
    print("\n--- Step 1: Pre-Flight Integrity Tests ---")
    run_command([pytest_bin, "tests/test_data_integrity.py", "-v"], "Data Integrity Tests")

    # Step 2: Run Data Preparation Pipeline
    print("\n--- Step 2: Execute Data Preparation Pipeline ---")
    run_command([python_bin, "scripts/prepare_data.py"], "Prepare Data Pipeline")

    # Step 3: Run Full Post-Generation Test Suite
    print("\n--- Step 3: Execute Full Post-Generation Test Suite ---")
    run_command([pytest_bin, "tests/", "-v"], "Full Pytest Suite")

    # Step 4: Generate Visualizations
    print("\n--- Step 4: Generate Sample Visualizations ---")
    run_command([python_bin, "scripts/generate_visualizations.py"], "Visualizations Generator")

    # Step 5: Compile Final Report
    print("\n==================================================")
    print(" PHASE 1 VERIFICATION REPORT")
    print("==================================================")
    
    with open("configs/dataset_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    summary_path = os.path.join(config["paths"]["metadata_dir"], "dataset_summary.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    print(f"1. Number of source images:          {summary['total_source_images']} (9 TIFFs)")
    print(f"2. Train / Val / Test tile counts:    Train: {summary['tile_counts']['train']} | Val: {summary['tile_counts']['val']} | Test: {summary['tile_counts']['test']}")
    print(f"3. Number of generated patches:      {summary['total_patches_generated']} total (Train: {summary['split_patch_counts']['train']}, Val: {summary['split_patch_counts']['val']}, Test: {summary['split_patch_counts']['test']})")
    print(f"4. Number of useful patches:         {summary['useful_boundary_patches']} ({summary['useful_patches_percentage']}%)")
    print(f"5. Image dimensions:                 {config['patch_extraction']['patch_size'][0]} x {config['patch_extraction']['patch_size'][1]} x 3 (RGB)")
    print(f"6. Label dimensions:                 {config['patch_extraction']['patch_size'][0]} x {config['patch_extraction']['patch_size'][1]} (Binary 0/1)")
    print(f"7. Coordinate Reference System (CRS):{summary['crs']} (Amersfoort / RD New)")
    print(f"8. Empty vs Non-Empty patches:       Empty/Background: {summary['empty_patches']} ({summary['empty_patches_percentage']}%) | Boundary-Containing: {summary['total_patches_generated'] - summary['empty_patches']} ({100 - summary['empty_patches_percentage']}%)")
    print(f"9. Errors:                           None. All tests passed successfully.")
    print("==================================================")


if __name__ == "__main__":
    main()
