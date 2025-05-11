import torch
import math
import matplotlib.pyplot as plt


def plot_positional_encoding(dim: int, max_len: int = 50):
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))

    # Apply sin to even indices in the array; 2i
    pe[:, 0::2] = torch.sin(position * div_term)

    # Apply cos to odd indices in the array; 2i+1
    pe[:, 1::2] = torch.cos(position * div_term)

    plt.figure(figsize=(12, 6))
    for i in range(dim):
        plt.plot(pe[:, i].numpy(), label=f"dim {i}")

    plt.title(f"Positional Encoding for {dim} dimensions")
    plt.xlabel("Position")
    plt.ylabel("Encoding value")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Example usage
    plot_positional_encoding(dim=8, max_len=50)
