import numpy as np
import matplotlib.pyplot as plt


def main():
    tmp_file_path = '../examples/samples_output.npz'
    data = np.load(tmp_file_path)
    y_s = data['y_s']
    outputs = data['output']

    # Apply threshold - see ./dataloaders/dataloaders.py
    threshold = 0.85
    values_outer = 1.0
    y_s_thr = np.where(y_s < threshold, 0.0, values_outer)
    outputs_thr = np.where(outputs < threshold, 0.0, values_outer)

    # Difference map
    diff = outputs_thr - y_s_thr

    # Plot
    plt.figure(figsize=(12, 4))

    # Ground truth / y_s
    plt.subplot(1, 3, 1)
    plt.title("Label (y_s)")
    plt.imshow(y_s_thr)
    plt.colorbar()

    # Model output
    plt.subplot(1, 3, 2)
    plt.title("Model outputs")
    plt.imshow(outputs_thr)
    plt.colorbar()

    # Difference
    plt.subplot(1, 3, 3)
    plt.title("Difference (outputs - y_s)")
    plt.imshow(diff)
    plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()