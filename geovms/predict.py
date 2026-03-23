from inference import InferenceModel
import numpy as np
import h5py


def load_sample():
    ''' Return a sample for the example.'''
    h5_path = "../examples/samples.h5"
    x_key = "X"
    y_key = "y"
    sample_index = 0

    # ----------------------------
    # Load data
    # ----------------------------
    with h5py.File(h5_path, "r") as f:
        X = f[x_key]  # (N, C, H, W)
        y = f[y_key]  # (N, H, W)

        N, C, H, W = X.shape
        assert y.shape == (N, H, W), f"y shape {y.shape} != expected {(N, H, W)}"
        assert 0 <= sample_index < N, f"sample_index out of range: {sample_index}"

        x_s = X[sample_index]  # (C, H, W)
        y_s = y[sample_index]  # (H, W)

        # Optional: clean NaNs and clamp labels to [0, 1]
        x_s = np.nan_to_num(x_s, nan=0.0)
        y_s = np.clip(np.nan_to_num(y_s, nan=0.0), 0.0, 1.0)

        return x_s, y_s


def go_inference(im, patch):
    _patch_tensor = np.array(patch).copy()
    predict_labels = im(_patch_tensor)
    return predict_labels


if __name__ == "__main__":
    import argparse
    import yaml
    from types import SimpleNamespace

    def convert_dict_to_namespace(d):
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = convert_dict_to_namespace(value)
        return SimpleNamespace(**d)

    parser = argparse.ArgumentParser(description='Prediction with GeoFormer')
    parser.add_argument('--config', default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    args = convert_dict_to_namespace(config)

    # load a sample
    x_s, y_s = load_sample()
    x_s = x_s[np.newaxis, :, :, :]

    args.lr = float(args.lr)
    args.decay_epochs = int(args.decay_epochs)
    args.decay_rate = float(args.decay_rate)
    im = InferenceModel(args)
    outputs = go_inference(im, x_s)
    tmp_file_path = '../examples/samples_output.npz'
    np.savez(tmp_file_path, y_s=y_s, output=outputs[0])
    print(r"Data saved at %s." % tmp_file_path, "To visualize, run plot_prediction.py.")


# python predict.py --config ./configs/config.yaml
