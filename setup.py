"""Setup script for SG-MRM package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="sg-mrm",
    version="0.1.0",
    author="SG-MRM Development Team",
    author_email="your.email@example.com",
    description="Satellite-Ground Multiscale Radio Map simulation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SG-RM",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "geo": [
            "gdal>=3.0.0",
            "rasterio>=1.2.0",
            "geopandas>=0.9.0",
            "shapely>=1.7.0",
        ],
        "orbit": [
            "skyfield>=1.39",
            "sgp4>=2.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "sg-mrm=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml"],
    },
    zip_safe=False,
)
