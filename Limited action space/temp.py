import numpy as np

p = 0
n = 0

for i in range(1000):
    state = np.random.choice([0, 1], p=[0.7, 0.3])
    if state == 1:
        n += 1
        p += 0.3
    else:
        state = np.random.choice([0, 1], p=[0.8, 0.2])
        if state == 1:
            n += 1
            p += 0.7 * 0.2

print(p/n)
print(n/1000)