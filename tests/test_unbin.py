import numpy as np
import unittest

# Step 1: Create a small softmax array
softmaxes = np.array([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3]])

# Step 2: Create a small bin edges array
bin_left_edges = np.array([1, 2, 3])

# Step 3: Multiply the softmax array with the bin edges array
result = softmaxes * bin_left_edges

# Step 4: Sum the result along the appropriate axis
predictions = np.sum(result, axis=1)

# Print the result
print(predictions)

# Step 5: Compare the result with the expected output
# The expected output is [2.5, 2.1] because:
# For the first softmax array: 0.1*1 + 0.3*2 + 0.6*3 = 2.5
# For the second softmax array: 0.2*1 + 0.5*2 + 0.3*3 = 2.1
expected_output = np.array([2.5, 2.1])
assert np.allclose(predictions, expected_output), "The test failed"


if __name__ == '__main__':
    unittest.main()
