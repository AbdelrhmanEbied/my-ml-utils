from setuptools import setup, find_packages

setup(
    name="MyFunctions",
    version="0.1.0",
    packages=find_packages(),
    description="Custom Deep Learning and Utility functions",
    author="Your Name",
    install_requires=[
        "torch",
        "torchvision",
        "tqdm",
        "matplotlib",
        "numpy",
        "requests",
    ],
    python_requires=">=3.7",
)