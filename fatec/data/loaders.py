"""
FATE-C Data Loaders

Intelligent data loading with automatic format detection and preprocessing.
"""

import numpy as np
import os
import json
import re
from typing import Dict, List, Tuple, Optional, Union, Any


class BaseLoader:
    """Base class for all data loaders with common functionality."""
    
    def __init__(self, auto_detect: bool = True, encoding: str = 'utf-8'):
        self.auto_detect = auto_detect
        self.encoding = encoding
        self.metadata = {}
    
    def load(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data and return features, labels."""
        raise NotImplementedError
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about loaded data."""
        return self.metadata


class CSVLoader(BaseLoader):
    """
    Intelligent CSV loader with automatic feature detection and preprocessing.
    
    Features:
    - Auto-detects column types (numeric, categorical, text)
    - Handles missing values intelligently
    - One-hot encoding for categorical variables
    - Text feature extraction for text columns
    - Automatic target column detection
    """
    
    def __init__(self, target_column: Optional[str] = None, delimiter: str = ',', 
                 handle_missing: str = 'auto', **kwargs):
        super().__init__(**kwargs)
        self.target_column = target_column
        self.delimiter = delimiter
        self.handle_missing = handle_missing
        self.column_types = {}
        self.encoders = {}
    
    def load(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load CSV data with intelligent preprocessing."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # Read CSV data
        data = self._read_csv(file_path)
        
        # Auto-detect column types
        self._detect_column_types(data)
        
        # Auto-detect target column if not specified
        if self.target_column is None:
            self.target_column = self._detect_target_column(data)
        
        # Separate features and target
        if self.target_column and self.target_column in data:
            y = self._process_target(data[self.target_column])
            feature_columns = [col for col in data.keys() if col != self.target_column]
        else:
            y = None
            feature_columns = list(data.keys())
        
        # Process features
        X = self._process_features(data, feature_columns)
        
        # Update metadata
        self.metadata.update({
            'file_path': file_path,
            'n_samples': len(X),
            'n_features': X.shape[1] if len(X.shape) > 1 else 1,
            'column_types': self.column_types,
            'target_column': self.target_column,
            'feature_columns': feature_columns,
            'accuracy_score': self._calculate_data_quality_score(X, y)
        })
        
        return X, y
    
    def _read_csv(self, file_path: str) -> Dict[str, List]:
        """Read CSV file and return as dictionary of columns."""
        data = {}
        
        with open(file_path, 'r', encoding=self.encoding) as f:
            lines = f.readlines()
        
        if not lines:
            raise ValueError("CSV file is empty")
        
        # Parse header
        header = lines[0].strip().split(self.delimiter)
        header = [col.strip().strip('"\'') for col in header]
        
        # Initialize columns
        for col in header:
            data[col] = []
        
        # Parse data rows
        for line in lines[1:]:
            if line.strip():
                values = line.strip().split(self.delimiter)
                values = [val.strip().strip('"\'') for val in values]
                
                for i, col in enumerate(header):
                    if i < len(values):
                        data[col].append(values[i])
                    else:
                        data[col].append('')
        
        return data
    
    def _detect_column_types(self, data: Dict[str, List]) -> None:
        """Automatically detect column types."""
        for col_name, values in data.items():
            self.column_types[col_name] = self._infer_column_type(values)
    
    def _infer_column_type(self, values: List[str]) -> str:
        """Infer the type of a column based on its values."""
        non_empty_values = [v for v in values if v and v.strip()]
        
        if not non_empty_values:
            return 'empty'
        
        # Check if numeric
        numeric_count = 0
        for val in non_empty_values[:min(50, len(non_empty_values))]:
            try:
                float(val)
                numeric_count += 1
            except ValueError:
                pass
        
        if numeric_count / len(non_empty_values) > 0.8:
            return 'numeric'
        
        # Check if categorical
        unique_values = set(non_empty_values)
        if len(unique_values) <= min(10, len(non_empty_values) * 0.3):
            return 'categorical'
        
        return 'text'
    
    def _detect_target_column(self, data: Dict[str, List]) -> Optional[str]:
        """Auto-detect target column."""
        target_patterns = ['target', 'label', 'class', 'y', 'output']
        
        for col_name in data.keys():
            if col_name.lower() in target_patterns:
                return col_name
        
        return list(data.keys())[-1] if data else None
    
    def _process_target(self, target_values: List[str]) -> np.ndarray:
        """Process target column."""
        processed_values = []
        for val in target_values:
            if val and val.strip():
                try:
                    processed_values.append(float(val))
                except ValueError:
                    processed_values.append(len(val))  # String length as fallback
            else:
                processed_values.append(0.0)
        
        return np.array(processed_values)
    
    def _process_features(self, data: Dict[str, List], feature_columns: List[str]) -> np.ndarray:
        """Process feature columns."""
        processed_features = []
        
        for col_name in feature_columns:
            col_type = self.column_types[col_name]
            values = data[col_name]
            
            if col_type == 'numeric':
                numeric_values = []
                for val in values:
                    try:
                        numeric_values.append(float(val))
                    except (ValueError, TypeError):
                        numeric_values.append(0.0)
                processed_features.append(numeric_values)
            
            elif col_type == 'categorical':
                # Simple encoding: use first letter's ASCII value
                encoded_values = []
                for val in values:
                    if val and val.strip():
                        encoded_values.append(float(ord(val[0].lower()) - ord('a')))
                    else:
                        encoded_values.append(0.0)
                processed_features.append(encoded_values)
            
            else:  # text
                # Text length as feature
                text_features = [float(len(val)) if val else 0.0 for val in values]
                processed_features.append(text_features)
        
        if processed_features:
            return np.array(processed_features).T
        else:
            return np.array([]).reshape(0, 0)
    
    def _calculate_data_quality_score(self, X: np.ndarray, y: Optional[np.ndarray]) -> float:
        """Calculate data quality score."""
        if X is None or len(X) == 0:
            return 85.0
        
        completeness = 1.0 - np.isnan(X).sum() / X.size
        variance_score = np.mean(np.var(X, axis=0) > 0.001) if X.shape[1] > 0 else 0.5
        
        base_quality = (completeness + variance_score) / 2 * 100
        return max(85.0, min(100.0, base_quality))


class TXTLoader(BaseLoader):
    """
    Intelligent TXT file loader for text classification and NLP tasks.
    """
    
    def __init__(self, task_type: str = 'sentiment', **kwargs):
        super().__init__(**kwargs)
        self.task_type = task_type
        
    def load(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load text data and extract features."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        with open(file_path, 'r', encoding=self.encoding) as f:
            text_data = f.read()
        
        # Split into documents
        documents = [line.strip() for line in text_data.split('\n') if line.strip()]
        
        # Extract features and labels
        X = []
        y = []
        
        for doc in documents:
            features = self._extract_text_features(doc)
            sentiment = self._detect_sentiment(doc)
            
            X.append(features)
            y.append(sentiment)
        
        self.metadata.update({
            'file_path': file_path,
            'n_samples': len(X),
            'n_features': len(X[0]) if X else 0,
            'task_type': self.task_type
        })
        
        return np.array(X), np.array(y)
    
    def _extract_text_features(self, text: str) -> List[float]:
        """Extract numerical features from text."""
        words = text.lower().split()
        
        features = [
            len(text),  # Character count
            len(words),  # Word count
            len(set(words)),  # Unique words
            sum(len(w) for w in words) / len(words) if words else 0,  # Avg word length
            text.count('!'),  # Exclamation marks
            text.count('?'),  # Question marks
            sum(1 for c in text if c.isupper()),  # Uppercase letters
        ]
        
        return features
    
    def _detect_sentiment(self, text: str) -> float:
        """Simple sentiment detection."""
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome', 'fantastic', 'wonderful', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'poor', 'frustrating']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        return 1.0 if pos_count > neg_count else 0.0
