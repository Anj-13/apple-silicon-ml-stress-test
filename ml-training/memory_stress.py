"""
Stress RAM usage
"""

import numpy as np
import psutil
import time

print(f"Total Ram: {psutil.virtual_memory().total / 1e9:.2f} GB")
print("Creating large arrays to stress memory")

arrays = []
for i in range(6):
    # Each array ~2GB
    arr = np.random.rand(250_000_000)
    arrays.append(arr)
    print(f"Array {i+1}: {arr.nbytes / 1e9:.2f} GB")
    print(f"Total RAM used: {psutil.virtual_memory().used / 1e9:.2f} GB")

print("\n Press any key to release memory...")
input()

# Release
del arrays
print(f"RAM after release: {psutil.virtual_memory().used / 1e9:.2f} GB")