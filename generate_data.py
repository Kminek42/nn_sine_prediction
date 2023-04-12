import math
import random

length = 10**5
points = 5
filename = "training_set.txt"

file = open(filename, "w")
for i in range(length):
    a = random.uniform(0.1, 0.4)
    b = random.uniform(0, 1)
    for j in range(points):
        output = math.sin(2 * math.pi * (a * j / points + b))
        file.write(str(output) + " ")
    file.write("\n")

file.close()
