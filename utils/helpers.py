# utils/helpers.py
"""
General utility functions for the project
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Union, Optional
import hashlib
import pickle

logger = logging.getLogger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2):
    """Save data to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_jsonl(filepath: Union[str, Path]) -> List[Dict]:
    """Load JSONL file"""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line in {filepath}")
    return results


def save_jsonl(data: List[Dict], filepath: Union[str, Path]):
    """Save data to JSONL file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_file_hash(filepath: Union[str, Path]) -> str:
    """Get SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def timer(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def chunked_iterable(iterable: List, chunk_size: int):
    """Yield chunks from iterable"""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def format_number(num: float, decimals: int = 2) -> str:
    """Format number with thousands separator"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.{decimals}f}M"
    elif num >= 1_000:
        return f"{num/1_000:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator"""
    return numerator / denominator if denominator != 0 else default


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'error': 'psutil not installed'}


def cache_result(cache_file: Union[str, Path], expire_hours: float = 24):
    """Decorator to cache function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_path = Path(cache_file)

            # Check if cache exists and is fresh
            if cache_path.exists():
                cache_age = time.time() - cache_path.stat().st_mtime
                if cache_age < expire_hours * 3600:
                    logger.info(f"Loading cached result from {cache_path}")
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)

            # Compute result
            result = func(*args, **kwargs)

            # Save to cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)

            return result
        return wrapper
    return decorator


def log_execution_time(func):
    """Log execution time of function"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        logger.info(
            f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper


def validate_config(config: Dict, required_keys: List[str]) -> bool:
    """Validate configuration dictionary has required keys"""
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        logger.error(f"Missing required configuration keys: {missing_keys}")
        return False

    return True


def create_backup(filepath: Union[str, Path], backup_dir: Optional[Path] = None):
    """Create backup of file"""
    filepath = Path(filepath)

    if not filepath.exists():
        logger.warning(f"File {filepath} does not exist, skipping backup")
        return

    backup_dir = backup_dir or filepath.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = get_timestamp()
    backup_path = backup_dir / f"{filepath.stem}_{timestamp}{filepath.suffix}"

    import shutil
    shutil.copy2(filepath, backup_path)
    logger.info(f"Created backup: {backup_path}")


# Console output helpers
class ConsoleColors:
    """ANSI color codes for console output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print colored header"""
    print(f"\n{ConsoleColors.HEADER}{ConsoleColors.BOLD}{text}{ConsoleColors.ENDC}")
    print("=" * len(text))


def print_success(text: str):
    """Print success message"""
    print(f"{ConsoleColors.OKGREEN}✓ {text}{ConsoleColors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{ConsoleColors.WARNING}⚠ {text}{ConsoleColors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{ConsoleColors.FAIL}✗ {text}{ConsoleColors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"{ConsoleColors.OKBLUE}ℹ {text}{ConsoleColors.ENDC}")
