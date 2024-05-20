import setuptools

# Load the long_description from README.md
with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyckan",  # Updated package name
    version="0.0.5",
    author="NeuralMaelstrom",  # Update author name if necessary
    author_email="your_email@example.com",  # Update author email if necessary
    description="Kolmogorov Arnold Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeuralMaelstrom/pyckan",  # Update repository URL
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch>=2.2.2',
        'numpy>=1.24.4',
        'scikit_learn>=1.1.3',
        'setuptools>=65.5.0',
        'sympy>=1.11.1',
        'matplotlib>=3.6.2',
        'tqdm>=4.66.2'
    ],
    python_requires='>=3.6',
)
