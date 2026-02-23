from setuptools import setup, find_packages

setup(
    name="chess-ai",
    version="1.0.0",
    description="Moteur d'échecs IA — Minimax + Alpha-Bêta + Q-Learning (Projet M1)",
    author="[Vos Noms]",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "python-chess>=1.10.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "ipython>=8.0.0",
        "jupyter>=1.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov>=4.0"],
    },
)
