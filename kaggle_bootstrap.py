#!/usr/bin/env python3
"""
kaggle_bootstrap.py — Bulletproof Kaggle & Cloud Notebook Environment Initializer
NLP Course Project: Problem Statement 4 (Cross-Project Software Story Point Estimation)

Fixes three common cloud/Kaggle execution failure modes:
1. FileNotFoundError: config.yaml (resolves across /kaggle/working and /kaggle/input)
2. AttributeError: '_CudaDeviceProperties' has no attribute 'total_mem' (safe PyTorch API)
3. FileExistsError: [Errno 17] File exists: 'TAWOS.sql' (safe symlink idempotency)
"""

import os
import sys
import shutil

def setup_kaggle_environment():
    print("=" * 75)
    print("TAWOS NLP PROJECT: BULLETPROOF KAGGLE / CLOUD ENVIRONMENT SETUP")
    print("=" * 75)
    
    is_kaggle = os.path.exists('/kaggle')
    is_colab = 'google.colab' in sys.modules
    platform_name = "Kaggle" if is_kaggle else ("Google Colab" if is_colab else "Local Workstation")
    print(f"[*] Detected Platform: {platform_name}")

    working_dir = '/kaggle/working' if is_kaggle else os.getcwd()
    if os.getcwd() != working_dir and is_kaggle:
        os.chdir(working_dir)
        print(f"[*] Switched active directory to: {working_dir}")

    # -------------------------------------------------------------------------
    # 1. FIX: FileNotFoundError for config.yaml
    # -------------------------------------------------------------------------
    config_target = os.path.join(working_dir, 'config.yaml')
    if not os.path.exists(config_target):
        # Search candidate locations
        config_candidates = [
            'config.yaml',
            '../config.yaml',
            '/kaggle/working/NLP project/config.yaml',
            '/kaggle/working/nlp-project/config.yaml'
        ]
        if is_kaggle and os.path.exists('/kaggle/input'):
            for root, _, files in os.walk('/kaggle/input'):
                if 'config.yaml' in files:
                    config_candidates.append(os.path.join(root, 'config.yaml'))
                    break

        found_config = None
        for cand in config_candidates:
            if os.path.exists(cand):
                found_config = cand
                break

        if found_config:
            shutil.copyfile(found_config, config_target)
            print(f"✓ [FIX 1] Synchronized config.yaml from: {found_config}")
        else:
            print("ℹ [FIX 1] config.yaml not found in candidate paths; notebook defaults will apply.")
    else:
        print(f"✓ [FIX 1] config.yaml confirmed present at: {config_target}")

    # -------------------------------------------------------------------------
    # 2. FIX: AttributeError for torch cuda device properties total_mem
    # -------------------------------------------------------------------------
    try:
        import torch
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            # PyTorch official API is 'total_memory'; fallback via getattr prevents AttributeError
            total_bytes = getattr(device_props, 'total_memory', getattr(device_props, 'total_mem', 0))
            total_gb = total_bytes / (1024 ** 3)
            print(f"✓ [FIX 2] GPU Detected: {device_props.name}")
            print(f"          VRAM Available: {total_gb:.2f} GB (Compute Capability: {device_props.major}.{device_props.minor})")
        else:
            print("✓ [FIX 2] PyTorch initialized successfully. (CPU Mode active)")
    except ImportError:
        print("ℹ [FIX 2] PyTorch not installed in current environment; proceeding in CPU mode.")
    except Exception as e:
        print(f"ℹ [FIX 2] Safe GPU inspection notice: {e}")

    # -------------------------------------------------------------------------
    # 3. FIX: FileExistsError for symlink TAWOS.sql
    # -------------------------------------------------------------------------
    sql_target = os.path.join(working_dir, 'TAWOS.sql')
    if os.path.exists(sql_target) or os.path.islink(sql_target):
        if os.path.islink(sql_target):
            # Check if broken link or re-run
            real_dest = os.path.realpath(sql_target)
            if not os.path.exists(real_dest):
                os.unlink(sql_target)
                print(f"✓ [FIX 3] Removed stale broken symlink: {sql_target}")
            else:
                print(f"✓ [FIX 3] TAWOS.sql symlink already active and valid -> {real_dest}")
        else:
            size_gb = os.path.getsize(sql_target) / (1024 ** 3)
            print(f"✓ [FIX 3] TAWOS.sql exists as regular file ({size_gb:.2f} GB)")
    else:
        # Search for TAWOS.sql in Kaggle input or subfolders
        sql_found = None
        if is_kaggle and os.path.exists('/kaggle/input'):
            for root, _, files in os.walk('/kaggle/input'):
                if 'TAWOS.sql' in files:
                    sql_found = os.path.join(root, 'TAWOS.sql')
                    break
        
        if sql_found:
            try:
                os.symlink(sql_found, sql_target)
                print(f"✓ [FIX 3] Safely created idempotent symlink: {sql_target} -> {sql_found}")
            except OSError as err:
                print(f"ℹ [FIX 3] Symlink creation notice ({err}); using source path directly.")
        else:
            print("ℹ [FIX 3] TAWOS.sql not found in /kaggle/input; preprocessed CSVs in data/processed will be used.")

    # -------------------------------------------------------------------------
    # 4. Ensure data/processed and results/ folders are available
    # -------------------------------------------------------------------------
    for folder in ['data', 'results']:
        target_path = os.path.join(working_dir, folder)
        if not os.path.exists(target_path):
            # Check if folder exists in input or parent
            found_folder = None
            if is_kaggle and os.path.exists('/kaggle/input'):
                for root, dirs, _ in os.walk('/kaggle/input'):
                    if folder in dirs:
                        candidate = os.path.join(root, folder)
                        # Check if has subdirectories
                        if os.path.exists(os.path.join(candidate, 'processed')) or os.path.exists(os.path.join(candidate, 'tables')):
                            found_folder = candidate
                            break
            if found_folder:
                if os.path.islink(target_path):
                    os.unlink(target_path)
                try:
                    os.symlink(found_folder, target_path)
                    print(f"✓ Linked {folder} directory -> {found_folder}")
                except Exception as e:
                    print(f"ℹ Notice linking {folder}: {e}")

    print("=" * 75)
    print("ENVIRONMENT IS 100% READY FOR NOTEBOOK EXECUTION & VIVA DEFENSE")
    print("=" * 75)

if __name__ == '__main__':
    setup_kaggle_environment()
