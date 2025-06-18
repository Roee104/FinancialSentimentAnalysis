# tests/test_path_handling.py
"""
Test that scripts can be imported without path errors
"""

import pytest
import sys
import importlib.util
from pathlib import Path


# Scripts to test
SCRIPTS = [
    "scripts/run_pipeline.py",
    "scripts/run_experiments.py", 
    "scripts/error_analysis.py"
]


@pytest.mark.parametrize("script_path", SCRIPTS)
def test_script_imports(script_path):
    """Test that script can be imported without FileNotFoundError"""
    script_path = Path(script_path)
    
    if not script_path.exists():
        pytest.skip(f"Script {script_path} not found")
    
    # Load the script as a module
    spec = importlib.util.spec_from_file_location(
        script_path.stem, 
        script_path
    )
    module = importlib.util.module_from_spec(spec)
    
    # This should not raise FileNotFoundError for hardcoded paths
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        # Scripts may call sys.exit() in if __name__ == "__main__"
        pass
    except FileNotFoundError as e:
        pytest.fail(f"Script {script_path} has hardcoded path issue: {e}")


def test_no_sys_path_hacks():
    """Verify scripts don't use sys.path.append hacks"""
    for script_path in SCRIPTS:
        script_path = Path(script_path)
        if not script_path.exists():
            continue
            
        with open(script_path, 'r') as f:
            content = f.read()
            
        # Check for common sys.path hacks
        assert "sys.path.append" not in content, f"{script_path} uses sys.path.append"
        assert "sys.path.insert" not in content, f"{script_path} uses sys.path.insert"


def test_config_imports():
    """Test that config can be imported from anywhere"""
    # This should work with PYTHONPATH=.
    from config.settings import DATA_DIR, LOGGING_CONFIG
    
    assert DATA_DIR.exists()
    assert isinstance(LOGGING_CONFIG, dict)