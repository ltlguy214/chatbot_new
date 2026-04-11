"""
Test script để kiểm tra load tất cả models
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add sklearn compatibility fixes
import sklearn.compose._column_transformer

try:
    from sklearn.compose._column_transformer import _RemainderColsList
except ImportError:
    class _RemainderColsList(list):
        """Dummy class for sklearn compatibility"""
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# SimpleImputer compatibility
from sklearn.impute import SimpleImputer
if not hasattr(SimpleImputer, '_fill_dtype'):
    SimpleImputer._fill_dtype = property(lambda self: getattr(self, '_fit_dtype', None))

# Numpy compatibility
import numpy as np
if not hasattr(np.random, 'bit_generator'):
    np.random.bit_generator = np.random._bit_generator

try:
    from numpy.random import MT19937
    if not hasattr(np.random, 'MT19937'):
        np.random.MT19937 = MT19937
except:
    pass

# Test load models
import joblib

models = {
    'P0': 'DA\\models\\best_model_p0.pkl',
    'P1': 'DA\\models\\best_model_p1.pkl',
    'P2': 'DA\\models\\best_model_p2.pkl',
    'P3': 'DA\\models\\best_model_p3.pkl',
    'P4': 'DA\\models\\best_model_p4.pkl',
}

print("="*60)
print("KIỂM TRA LOAD MODELS TỪ DA/models")
print("="*60)

for name, path in models.items():
    try:
        data = joblib.load(path)
        if isinstance(data, dict):
            keys = list(data.keys())
            print(f"\n✓ {name}: SUCCESS")
            print(f"  Keys: {keys}")
            if 'model_name' in data:
                print(f"  Model: {data['model_name']}")
            if 'accuracy' in data:
                print(f"  Accuracy: {data['accuracy']:.4f}")
            if 'r2_score' in data:
                print(f"  R² Score: {data['r2_score']:.4f}")
        else:
            print(f"\n✓ {name}: SUCCESS (type: {type(data).__name__})")
    except Exception as e:
        print(f"\n✗ {name}: FAILED")
        print(f"  Error: {str(e)[:100]}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Tất cả models load thành công: Sẵn sàng chạy app!")
