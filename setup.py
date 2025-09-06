#!/usr/bin/env python3
"""
VEP-eMMCSE: Verifiable, Expressive, and Post-Quantum enhanced 
Multi-source Multi-client Conjunctive Searchable Encryption

Setup script for installation and distribution.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "VEP-eMMCSE: A Post-Quantum Framework for Searchable Encryption"

# Read requirements from requirements.txt
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return ["cryptography>=41.0.7", "numpy>=1.24.3", "pqcrypto>=0.3.4"]

setup(
    name="vep-emmcse",
    version="1.0.0",
    author="Seyed Hossein Ahmadpanah",
    author_email="h.ahmadpanah@iau.ac.ir",
    description="Verifiable, Expressive, and Post-Quantum enhanced Multi-source Multi-client Searchable Encryption",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ahmadpanah/vep-emmcse",
    project_urls={
        "Bug Tracker": "https://github.com/ahmadpanah/vep-emmcse/issues",
        "Documentation": "https://vep-emmcse.readthedocs.io/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security :: Cryptography",
        "Topic :: Scientific/Engineering",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "experiments": [
            "matplotlib>=3.7.1",
            "jupyter>=1.0.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vep-emmcse-benchmark=vep_emmcse.experiments.performance_eval:main",
            "vep-emmcse-setup=vep_emmcse.utils.setup:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vep_emmcse": ["data/*.json", "experiments/configs/*.yaml"],
    },
    zip_safe=False,
)