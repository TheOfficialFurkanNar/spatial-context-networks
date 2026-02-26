from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="scn",
    version="0.1.0",
    author="Furkan Nar",
    author_email="furkannar168@hotmail.com",
    description="Geometric Semantic Routing in Neural Architectures",
    url="https://github.com/TheOfficialFurkanNar/spatial-context-networks",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest", "matplotlib", "numpy"],
    },
)
