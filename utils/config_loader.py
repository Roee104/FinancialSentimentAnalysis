# utils/config_loader.py
"""
Configuration loader with YAML support and CLI overrides
"""

import argparse
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = deepcopy(base_dict)
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: Optional[Path] = None,
    defaults: Optional[Dict] = None,
    parse_cli: bool = True
) -> Dict[str, Any]:
    """
    Load configuration from defaults, YAML file, and CLI overrides

    Args:
        config_path: Path to YAML config file
        defaults: Default configuration dictionary
        parse_cli: Whether to parse command line arguments

    Returns:
        Merged configuration dictionary
    """
    # Start with defaults
    if defaults is None:
        from config.settings import DEFAULT_CONFIG
        defaults = DEFAULT_CONFIG

    config = deepcopy(defaults)

    # Load YAML if provided
    if config_path and config_path.exists():
        logger.info(f"Loading config from {config_path}")
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    config = deep_update(config, yaml_config)
                    logger.debug(
                        f"Loaded {len(yaml_config)} config keys from YAML")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    # Parse CLI arguments if requested
    if parse_cli:
        parser = argparse.ArgumentParser(
            description="Financial Sentiment Analysis Pipeline")

        # Add common arguments
        parser.add_argument("--config", type=str,
                            help="Path to YAML config file")
        parser.add_argument("--batch-size", type=int,
                            help="Processing batch size")
        parser.add_argument("--max-articles", type=int,
                            help="Maximum articles to process")
        parser.add_argument("--sentiment-batch-size", type=int,
                            help="Sentiment model batch size")
        parser.add_argument("--checkpoint-interval", type=int,
                            help="Checkpoint save interval")
        parser.add_argument("--log-level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                            help="Logging level")

        # Parse known args to allow for additional arguments
        args, _ = parser.parse_known_args()

        # Apply CLI overrides
        if args.batch_size:
            config['pipeline_config']['batch_size'] = args.batch_size
        if args.sentiment_batch_size:
            config['sentiment_config']['batch_size'] = args.sentiment_batch_size
        if args.checkpoint_interval:
            config['pipeline_config']['checkpoint_interval'] = args.checkpoint_interval
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))

    return config


def save_config(config: Dict[str, Any], path: Path):
    """Save configuration to YAML file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config to {path}")
