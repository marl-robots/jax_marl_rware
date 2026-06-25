import numpy as np
import matplotlib.pyplot as plt


def plot_lorenz_curve(x_axis, y_axis):
    # x: [N]
    # y: [E, N]

    E = y_axis.shape[0]

    plt.figure(figsize=(6, 6))

    for e in range(E):
        plt.plot(x_axis, y_axis[e], label=f"Episode {e}")

    # קו שוויון (45°)
    plt.plot([0, 1], [0, 1], "k--", label="Equality line")

    plt.xlabel("Fraction of agents")
    plt.ylabel("Fraction of cumulative contribution")
    plt.title("Lorenz Curve per Episode")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_action_histogram_bar(hist):
    """
    hist: shape (N, bins)
    Plots per-agent histogram as bar charts.
    """
    hist = np.array(hist)
    N, B = hist.shape

    plt.figure(figsize=(12, 4 * N))

    for i in range(N):
        plt.subplot(N, 1, i + 1)
        plt.bar(range(B), hist[i], color="blue", alpha=0.7, edgecolor="black")
        plt.title(f"Action Histogram – Agent {i}")
        plt.xlabel("Bin")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def plot_action_histogram_line(hist):
    """
    hist: shape (N, bins)
    Plots per-agent normalized histogram as line curves.
    """
    hist = np.array(hist)
    N, B = hist.shape
    print(B)
    # normalize to probability density
    p = hist / (hist.sum(axis=1, keepdims=True) + 1e-12)

    xs = np.linspace(0, B - 1, B)

    plt.figure(figsize=(12, 4 * N))

    for i in range(N):
        plt.subplot(N, 1, i + 1)
        plt.plot(xs, p[i], linewidth=2, color="blue")
        plt.fill_between(xs, p[i], alpha=0.3, color="blue")
        plt.title(f"Normalized Action Distribution – Agent {i}")
        plt.xlabel("Normalized Action Range")
        plt.ylabel("Density")

    plt.tight_layout()
    plt.show()


def plot_action_entropy(entropy):
    """
    entropy: shape (N,)
    """
    entropy = np.array(entropy)
    N = len(entropy)

    plt.figure(figsize=(8, 4))
    plt.bar(range(N), entropy)
    plt.title("Action Entropy per Agent")
    plt.xlabel("Agent")
    plt.ylabel("Entropy (nats)")
    plt.tight_layout()
    plt.show()


def plot_jsd_matrix(jsd):
    """
    jsd: shape (N, N)
    """
    jsd = np.array(jsd)
    N = jsd.shape[0]

    plt.figure(figsize=(6, 5))
    plt.imshow(jsd, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="JSD")

    plt.title("Pairwise Jensen-Shannon Divergence")
    plt.xlabel("Agent")
    plt.ylabel("Agent")

    # Add values inside cells
    for i in range(N):
        for j in range(N):
            plt.text(j, i, f"{jsd[i, j]:.2f}", ha="center", va="center", color="white")

    plt.tight_layout()
    plt.show()
