from setuptools import setup, find_packages

with open("requirements.txt") as req:
    install_requires = [ln.strip() for ln in req if ln.strip() and not ln.startswith("#")]

setup(
    name="financial-sentiment-analysis",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=install_requires,
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
)
