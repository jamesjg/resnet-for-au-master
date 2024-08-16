import json
import os
import numpy as np

# Load the JSON file
with open('result/raf_db_result.json', 'r') as f:
    data = json.load(f)

# Load the numpy file
npy_data = np.load('raf_au_preds_new.npy')

# Check if the JSON and numpy files have the same length
assert len(data) == len(npy_data), "length data: {} != length npy_data: {}".format(len(data), len(npy_data))
diff_sum = 0
max_diff = 0
max_diff_index = 0
for i in range(len(data)):
    npy_au = npy_data[i]
    json_au = data[i]['au']
    diff = abs(npy_au - json_au)
    diff_sum += np.sum(diff)/24
    if np.sum(diff) / 24 >  max_diff:
        max_diff = np.sum(diff) / 24
        max_diff_index = i
    print(f"Difference at index {i}: {np.sum(diff)/24}")
print(f"Total difference: {diff_sum}")
print(f"Max difference: {max_diff}")
print(f"Max difference index: {max_diff_index}")