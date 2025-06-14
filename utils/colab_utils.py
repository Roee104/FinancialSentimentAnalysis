# utils/colab_utils.py
"""
Google Colab specific utilities
"""

import os
import sys
import subprocess
import gc
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def setup_colab_environment() -> Dict[str, Any]:
    """Setup Colab environment with optimal settings"""
    info = {
        'is_colab': is_colab(),
        'gpu_available': False,
        'gpu_name': None,
        'drive_mounted': False
    }

    if not info['is_colab']:
        logger.info("Not running in Google Colab")
        return info

    logger.info("🔧 Setting up Google Colab environment...")

    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        info['gpu_available'] = True
        info['gpu_name'] = torch.cuda.get_device_name(0)
        logger.info(f"✅ GPU available: {info['gpu_name']}")
    else:
        logger.info("ℹ️ No GPU available, using CPU")

    # Set memory-efficient environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Garbage collection
    gc.collect()

    return info


def mount_google_drive(mount_point: str = '/content/drive') -> bool:
    """Mount Google Drive in Colab"""
    if not is_colab():
        logger.warning("Not in Colab, cannot mount Google Drive")
        return False

    try:
        from google.colab import drive
        drive.mount(mount_point)
        logger.info(f"✅ Google Drive mounted at {mount_point}")
        return True
    except Exception as e:
        logger.error(f"Failed to mount Google Drive: {e}")
        return False


def install_requirements(requirements_file: str = "requirements.txt"):
    """Install requirements optimized for Colab"""
    if not os.path.exists(requirements_file):
        logger.error(f"Requirements file not found: {requirements_file}")
        return

    logger.info("📦 Installing requirements...")

    # Read requirements
    with open(requirements_file, 'r') as f:
        requirements = f.readlines()

    # Filter out comments and empty lines
    requirements = [r.strip() for r in requirements
                    if r.strip() and not r.strip().startswith('#')]

    # Install in batches to avoid memory issues
    batch_size = 5
    for i in range(0, len(requirements), batch_size):
        batch = requirements[i:i+batch_size]
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + batch
        subprocess.run(cmd)

    logger.info("✅ Requirements installed")


def download_from_drive(file_id: str, output_path: str) -> bool:
    """Download file from Google Drive using gdown"""
    try:
        # Install gdown if needed
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"])

        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

        logger.info(f"✅ Downloaded file to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download from Drive: {e}")
        return False


def get_optimal_batch_size() -> int:
    """Get optimal batch size based on available resources"""
    if torch.cuda.is_available():
        # Get GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        if gpu_memory_gb >= 15:  # T4 or better
            return 100
        elif gpu_memory_gb >= 10:
            return 50
        else:
            return 25
    else:
        # CPU mode
        return 10


def monitor_resources() -> Dict[str, float]:
    """Monitor Colab resources"""
    info = {}

    # RAM usage
    try:
        import psutil
        ram = psutil.virtual_memory()
        info['ram_used_gb'] = ram.used / 1e9
        info['ram_total_gb'] = ram.total / 1e9
        info['ram_percent'] = ram.percent
    except ImportError:
        pass

    # GPU usage
    if torch.cuda.is_available():
        info['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / 1e9
        info['gpu_memory_total_gb'] = torch.cuda.get_device_properties(
            0).total_memory / 1e9
        info['gpu_utilization'] = torch.cuda.memory_allocated(
        ) / torch.cuda.get_device_properties(0).total_memory * 100

    return info


def save_to_drive(local_path: str, drive_path: str) -> bool:
    """Save file to Google Drive"""
    if not is_colab():
        logger.warning("Not in Colab, cannot save to Drive")
        return False

    try:
        import shutil
        drive_full_path = f"/content/drive/MyDrive/{drive_path}"

        # Create directory if needed
        os.makedirs(os.path.dirname(drive_full_path), exist_ok=True)

        # Copy file
        shutil.copy2(local_path, drive_full_path)
        logger.info(f"✅ Saved to Drive: {drive_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save to Drive: {e}")
        return False


def create_download_link(filepath: str):
    """Create download link in Colab"""
    if not is_colab():
        logger.warning("Not in Colab, cannot create download link")
        return

    try:
        from google.colab import files
        files.download(filepath)
        logger.info(f"✅ Download started for {filepath}")
    except Exception as e:
        logger.error(f"Failed to create download link: {e}")


def clear_memory():
    """Clear memory in Colab"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("✅ Memory cleared")


def setup_nltk_data():
    """Download required NLTK data"""
    import nltk

    required_data = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger']

    for data_name in required_data:
        try:
            nltk.data.find(f'tokenizers/{data_name}')
        except LookupError:
            logger.info(f"Downloading NLTK {data_name}...")
            nltk.download(data_name, quiet=True)


# Colab-specific pipeline settings
def get_colab_pipeline_config() -> Dict:
    """Get optimized pipeline configuration for Colab"""
    config = {
        'batch_size': get_optimal_batch_size(),
        'sentiment_batch_size': 8 if torch.cuda.is_available() else 4,
        'max_chunks': 20,  # Limit chunks per article
        'checkpoint_interval': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    logger.info(f"Colab pipeline config: {config}")
    return config


def run_with_memory_management(func, *args, **kwargs):
    """Run function with memory management"""
    try:
        # Clear memory before
        clear_memory()

        # Run function
        result = func(*args, **kwargs)

        # Clear memory after
        clear_memory()

        return result
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(
                "GPU out of memory! Clearing cache and retrying with smaller batch...")
            clear_memory()

            # Retry with smaller batch
            if 'batch_size' in kwargs:
                kwargs['batch_size'] = kwargs['batch_size'] // 2
                logger.info(f"Retrying with batch_size={kwargs['batch_size']}")
                return func(*args, **kwargs)
        raise e
