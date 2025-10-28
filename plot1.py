import matplotlib.pyplot as plt
import numpy as np

sizes = [256, 512, 1024]
h2d = [0.0566, 0.2062, 1.1299]
compute = [0.1270, 0.4110, 2.2023]
d2h = [0.0978, 0.2961, 0.8483]

plt.bar(sizes, h2d, label='H2D Transfer')
plt.bar(sizes, compute, bottom=h2d, label='Compute Kernel')
plt.bar(sizes, d2h, bottom=np.add(h2d, compute), label='D2H Transfer')

plt.xlabel('Matrix Size (N)')
plt.ylabel('Time (ms)')
plt.title('Figure 3: GPU Time Breakdown')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

