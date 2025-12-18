"""
GeoSAE: A 3D Stratigraphic Modeling Method Driven by Geological Constraint

Implementation based on:
Yang, Y.; Zhou, J.; Ruan, M.; Xiao, H.; Hua, W.; Wei, W.
GeoSAE: A 3D Stratigraphic Modeling Method Driven by Geological Constraint.
Appl. Sci. 2025, 15, 1185. https://doi.org/10.3390/app15031185
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="geosae",
    version="1.0.0",
    author="BoundaryRSS",
    description="GeoSAE: A 3D Stratigraphic Modeling Method Driven by Geological Constraint",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/weihaog1/BoundaryRSS-GeoSAE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
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
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
