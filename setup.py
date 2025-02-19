from setuptools import setup, find_packages

setup(
    name="volatility-clustering-rl-execution",
    version="1.0.0",
    description="Institutional Order Flow Trading Strategy with Market Impact Prediction",
    author="Ben Pfeffer",
    author_email="benpfefferpro@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "torch>=1.9.0",
        "tqdm>=4.62.0",
        "python-dotenv>=0.19.0",
        "backtrader>=1.9.76.123",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "run-backtest=src.scripts.run_backtest:main",
            "process-features=src.data.process_features:main",
            "train-model=src.scripts.train_tcn:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
