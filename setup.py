"""Setup script for max_info_atlases package."""

from setuptools import setup, find_packages, Extension
import numpy as np

# Try to import Cython
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

# Define Cython extensions
ext_modules = []
if USE_CYTHON:
    extensions = [
        Extension(
            "max_info_atlases.cython.ConnectedComponentEntropy",
            ["src/max_info_atlases/cython/ConnectedComponentEntropy.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-std=c99"],
        )
    ]
    ext_modules = cythonize(extensions, compiler_directives={'language_level': "3"})

setup(
    name="max_info_atlases",
    version="2.0.0",
    description="Cell type clustering and percolation analysis pipeline",
    author="Roy Wollman",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "igraph>=0.9",
        "pyyaml>=5.4",
        "click>=8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "cython>=0.29",
        ],
        "anndata": [
            "anndata>=0.8",
            "scanpy>=1.9",
        ],
    },
    setup_requires=[
        "numpy>=1.20",
        "cython>=0.29",
    ],
    entry_points={
        "console_scripts": [
            "max-info=max_info_atlases.cli.main:cli",
        ],
    },
)
