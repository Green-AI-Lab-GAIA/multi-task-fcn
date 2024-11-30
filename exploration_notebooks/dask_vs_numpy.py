import numpy as np
import dask.array as da
import time

# Define the large array shape
shape = (55000, 22000, 17)

# Function to measure time for NumPy operations
def numpy_operations():
    # Initialize a large NumPy array
    arr = np.random.rand(*shape).astype(np.float16)

    # Max calculation
    start_time = time.time()
    max_value = np.max(arr)
    max_time = time.time() - start_time

    # Matrix division
    start_time = time.time()
    divided_array = arr / 2
    division_time = time.time() - start_time

    # Filtering with mask
    start_time = time.time()
    mask = arr > 0.5
    filtered_array = arr[mask]
    filter_time = time.time() - start_time

    # Writing on the array
    start_time = time.time()
    arr[:, :, 0] = 0  # Set first channel to zero as a test
    write_time = time.time() - start_time

    return max_time, division_time, filter_time, write_time

# Function to measure time for Dask operations
def dask_operations():
    # Initialize a large Dask array
    arr = da.random.random(shape, chunks=(1000, 1000, 17))

    # Max calculation
    start_time = time.time()
    max_value = arr.max().compute()
    max_time = time.time() - start_time

    # Matrix division
    start_time = time.time()
    divided_array = (arr / 2).compute()
    division_time = time.time() - start_time

    # Filtering with mask
    start_time = time.time()
    mask = arr > 0.5
    filtered_array = arr[mask].compute()
    filter_time = time.time() - start_time

    # Writing on the array
    start_time = time.time()
    arr[:, :, 0] = 0  # Set first channel to zero as a test
    arr.compute()  # Execute the write
    write_time = time.time() - start_time

    return max_time, division_time, filter_time, write_time

# Run and print results
print("NumPy Times (seconds):")
numpy_times = numpy_operations()
print(f"Max calculation: {numpy_times[0]:.4f}")
print(f"Matrix division: {numpy_times[1]:.4f}")
print(f"Filtering with mask: {numpy_times[2]:.4f}")
print(f"Writing on the array: {numpy_times[3]:.4f}")

print("\nDask Times (seconds):")
dask_times = dask_operations()
print(f"Max calculation: {dask_times[0]:.4f}")
print(f"Matrix division: {dask_times[1]:.4f}")
print(f"Filtering with mask: {dask_times[2]:.4f}")
print(f"Writing on the array: {dask_times[3]:.4f}")
