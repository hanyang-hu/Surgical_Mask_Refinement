"""Setup script for surgical mask refinement project."""

from setuptools import setup, find_packages

setup(
    name="surgical_mask_refinement",
    version="0.1.0",
    description="Latent diffusion model for surgical instrument segmentation refinement",
    author="ECE285 Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "transformers>=4.30.0",
        "diffusers>=0.21.0",
        "accelerate>=0.20.0",
        "einops>=0.6.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "isort",
        ],
    },
)
