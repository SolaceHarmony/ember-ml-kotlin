from setuptools import setup, find_packages

setup(
    name="ember-ml",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "torch>=1.10.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "audio": ["librosa>=0.8.0"],
        "mlx": ["mlx>=0.23.2"],
        "dev": ["pandas-stubs>=1.2.0", "mypy>=0.910"],
    },
    author="SolaceDev Team",
    author_email="sydney@solace.ofharmony.ai",
    description="A backend-agnostic neural network library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SolaceHarmony/ember-ml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.8",
)
