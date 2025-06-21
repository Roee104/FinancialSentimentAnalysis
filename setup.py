"""
Package configuration for Financial Sentiment Analysis.
Run `pip install -e .` from the repo root after editing.
"""

from pathlib import Path
from setuptools import find_packages, setup

# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
NAME = "financial-sentiment-analysis"
VERSION = "1.0.0"
PYTHON_REQUIRES = ">=3.9"

# -----------------------------------------------------------------------------
# Helper: read long-description from README (nice for PyPI / test-pypi)
# -----------------------------------------------------------------------------
this_dir = Path(__file__).parent
readme_path = this_dir / "README.md"
LONG_DESCRIPTION = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# -----------------------------------------------------------------------------
# Helper: parse requirements.txt
# -----------------------------------------------------------------------------
with open(this_dir / "requirements.txt", encoding="utf-8") as req_file:
    INSTALL_REQUIRES = [
        line.strip()
        for line in req_file
        if line.strip() and not line.startswith("#")
    ]

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
setup(
    name=NAME,
    version=VERSION,
    description="End-to-end NLP pipeline for financial news sentiment analysis.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=PYTHON_REQUIRES,
    packages=find_packages(exclude=["tests", "docs", "notebooks"]),
    include_package_data=True,              # ship CSV/YAML assets inside packages
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": [
            "pytest>=7.3",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.0",
            "coverage>=7.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
