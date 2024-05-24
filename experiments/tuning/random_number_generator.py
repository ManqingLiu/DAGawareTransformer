
# randomly generate a number between 0 and 99

import random

# Set the seed and generate a random number
random.seed(4784)
print(random.randint(0, 99)) # 62

# Set a different seed and generate another random number
random.seed(297)
print(random.randint(0, 99)) # 67

# Set yet another different seed and generate a third random number
random.seed(2087)
print(random.randint(0, 99)) # 16