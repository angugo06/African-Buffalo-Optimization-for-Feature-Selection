import matplotlib.pyplot as plt

def plot_convergence(history, out_path: str, title: str = "ABO Convergence (best score)"):
    plt.figure()
    plt.plot(list(range(len(history))), history)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best score")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
