from matplotlib import pyplot as plt
import numpy as np

labels = ['Sequential', 'Strategy 1', 'Strategy 2 w. 4 threads', 'Strategy 2 w. 11 threads']
x = np.arange(len(labels))   # [0,1,2]
width = 0.35

speedup_d3 = [1, 1.831720135, 2.1278, 2.701581343]
speedup_d5 = [1, 2.659, 2.82627, 4.1124]

plt.figure(figsize=(8,4)) 

bars1 = plt.bar(
    x - width/2, speedup_d3, width,
    label='Depth = 3',
    color='#4C72B0'
)

bars2 = plt.bar(
    x + width/2, speedup_d5, width,
    label='Depth = 5',
    color='#C44E52'
)

plt.ylabel('Speedup')
plt.title('Speedup Comparison under Different Depths')
plt.xticks(x, labels)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 數值標籤
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            h,
            f'{h:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

plt.tight_layout()
plt.show()
