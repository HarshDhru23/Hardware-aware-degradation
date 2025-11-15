#!/usr/bin/env python3
"""
Setup script for Hardware-aware Degradation Pipeline
ISRO Multi-Frame Super-Resolution (MFSR) Project
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="hardware-aware-degradation",
    version="1.0.0",
    author="ISRO MFSR Team",
    author_email="your.email@example.com",
    description="Hardware-aware degradation pipeline for satellite super-resolution training data generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HarshDhru23/Hardware-aware-degradation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
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
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "process-images=scripts.process_images:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml"],
    },
    zip_safe=False,
    keywords="satellite-imagery super-resolution image-processing degradation-pipeline",
    project_urls={
        "Bug Reports": "https://github.com/HarshDhru23/Hardware-aware-degradation/issues",
        "Source": "https://github.com/HarshDhru23/Hardware-aware-degradation",
        "Documentation": "https://github.com/HarshDhru23/Hardware-aware-degradation/blob/main/README.md",
    },
)