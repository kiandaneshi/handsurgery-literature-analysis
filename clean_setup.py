"""
Setup script for Hand Surgery Literature Analysis Pipeline
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("pyproject.toml", "r", encoding="utf-8") as fh:
    # Extract dependencies from pyproject.toml
    import re
    content = fh.read()
    deps_match = re.search(r'dependencies = \[(.*?)\]', content, re.DOTALL)
    if deps_match:
        deps_text = deps_match.group(1)
        # Extract package names from the dependency list
        dependencies = []
        for line in deps_text.split('\n'):
            line = line.strip()
            if line and line.startswith('"') and line.endswith('",'):
                pkg = line[1:-2]  # Remove quotes and comma
                dependencies.append(pkg)
    else:
        dependencies = []

setup(
    name="hand-surgery-literature-analysis",
    version="1.0.0",
    author="Clinical Research Team",
    author_email="research@example.com",
    description="BioBERT-powered clinical entity extraction and ML prediction for hand surgery outcomes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clinical-research/hand-surgery-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "biopython>=1.81",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0", 
        "pandas>=2.0.0",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0",
        "seaborn>=0.12.0",
        "streamlit>=1.25.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "xgboost>=1.7.0",
        "nltk>=3.8",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "jupyter>=1.0.0"
        ],
        "gpu": [
            "torch[cuda]>=2.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "hand-surgery-analysis=clean_run_pipeline:main",
            "hand-surgery-app=clean_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
    },
    keywords=[
        "medical-ai",
        "biobert", 
        "clinical-nlp",
        "hand-surgery",
        "literature-analysis",
        "pubmed",
        "machine-learning",
        "healthcare"
    ],
    project_urls={
        "Bug Reports": "https://github.com/clinical-research/hand-surgery-analysis/issues",
        "Source": "https://github.com/clinical-research/hand-surgery-analysis",
        "Documentation": "https://github.com/clinical-research/hand-surgery-analysis/blob/main/README.md",
    },
)