import re
import matplotlib.pyplot as plt

line_re = re.compile(
    r"time=([\d\.]+)s.*eval_avg_score=([\d\.]+)"
)

def load_curve(path):
    xs, ys = [], []
    with open(path, 'r') as f:
        for line in f:
            m = line_re.search(line)
            if not m:
                continue
            t = float(m.group(1))
            score = float(m.group(2))
            xs.append(t)
            ys.append(score)
    return xs, ys

logs = {
    "1 proc": "log_1.txt",
    "4 proc": "log_4.txt",
}

for label, fname in logs.items():
    t, s = load_curve(fname)
    plt.plot(t, s, label=label)

plt.xlabel("Wall-clock time (s)")
plt.ylabel("Global avg score")
plt.title("TD-learning 2048: training speed vs #MPI processes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
