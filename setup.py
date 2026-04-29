# setup.py
# Allows the project to be installed in editable mode:
#   pip install -e .
# This fixes all import errors when running notebooks or tests.

from setuptools import setup, find_packages

setup(
    name="healthcare-rag",
    version="0.2.0",
    description="Healthcare RAG-Powered Medical Q&A Assistant",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[],  # requirements.txt is the source of truth
)