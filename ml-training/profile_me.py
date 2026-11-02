import numpy as np
import time

def slow_function(n):
    # Slow
    result = []
    for i in range(n):
        result.append(np.sin(i) * np.cos(i) ** 2)
    return np.array(result)

def medium_function(n):
    # Vectorized(faster)
    x = np.arange(n)
    return np.sin(x) * np.cos(x) ** 2

def fast_function(n):
    # NumPy with broadcasting
    x = np.arange(n)
    return np.sum(x) * np.power(np.cos(x), 2)

if __name__ == "__main__":
    N = 10_000_000

    print("Running slow function")
    start = time.time()
    slow_function(N // 10) # 1M
    print(f"Slow: {time.time()-start:.2f}s")

    print("Running medium function")
    start = time.time()
    medium_function(N)
    print(f"Medium: {time.time()-start:.2f}s")

    print("Running fast function")
    start = time.time()
    fast_function(N)
    print(f"Fast: {time.time()-start:.2f}s")