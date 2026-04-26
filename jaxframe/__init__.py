"""
JAXFrame: A JAX-based DataFrame library with pandas-like API.

JAXFrame provides DataFrames powered by JAX, enabling:
- JIT compilation for fast computation
- Automatic differentiation for machine learning
- GPU/TPU acceleration
- Familiar pandas-like interface
"""

from jaxframe.dataframe import DataFrame, Series, concat, read_csv

__version__ = "0.1.0"
__all__ = ["DataFrame", "Series", "concat", "read_csv"]
