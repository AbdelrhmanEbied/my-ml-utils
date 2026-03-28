from setuptools import setup, find_packages

setup(
    name="my_ml_utils",
    version="0.1.0",
    description="A collection of PyTorch utilities for ML workflows",
    author="AbdelrhmanEbied",
    
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "matplotlib",
        "tqdm",
        "requests",
        "pandas",
        "scikit-learn",
        "python-dotenv", 
    ],
)
