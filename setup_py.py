"""
Setup configuration for 4D-CT Strain Analysis package
"""

from setuptools import setup, find_packages
import os

# Read README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="4dct-strain-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description="A comprehensive Python toolkit for analyzing respiratory motion and strain from 4D-CT imaging data",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/4dct-strain-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/4dct-strain-analysis/issues",
        "Documentation": "https://github.com/yourusername/4dct-strain-analysis/wiki",
        "Source Code": "https://github.com/yourusername/4dct-strain-analysis",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "4dct-strain-analysis=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    keywords=[
        "medical imaging",
        "4D-CT",
        "strain analysis",
        "respiratory motion",
        "lung imaging",
        "image processing",
        "biomedical engineering"
    ],
    zip_safe=False,
)
