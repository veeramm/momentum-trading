#!/usr/bin/env python
"""
Momentum Trading Application Setup Script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="momentum-trading",
    version="1.0.0",
    author="Mahesh Veeramani",
    author_email="veeramm@hotmail.com",
    description="A comprehensive momentum trading application for medium and long-term strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/momentum-trading",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
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
            "pytest>=7.0",
            "pytest-asyncio>=0.20",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "docs": [
            "sphinx>=6.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "momentum-backtest=scripts.backtest:main",
            "momentum-trade=scripts.live_trading:main",
            "momentum-update=scripts.data_update:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
