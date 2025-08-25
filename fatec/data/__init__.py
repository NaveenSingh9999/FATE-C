"""
FATE-C Data Loading and Processing Module

Provides intelligent data loading capabilities for diverse formats:
- CSV files with automatic feature detection
- TXT files with NLP preprocessing  
- JSON datasets
- Image data (future)
- Time series data (future)
"""

from .loaders import CSVLoader, TXTLoader

__all__ = [
    'CSVLoader', 'TXTLoader'
]
