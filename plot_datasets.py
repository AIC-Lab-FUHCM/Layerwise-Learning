from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "figures"


DATASETS = {
    "ecg": {
        "filename": "chfdb_chf01_275.pkl",
        "features": [0, 1],
        "label": 2,
    },
    "gesture": {
        "filename": "ann_gun_CentroidA.pkl",
        "features": [0, 1],
        "label": 2,
    },
    "pd": {
        "filename": "power_data.pkl",
        "features": [0],
        "label": 1,
    },
}


TRAIN_RANGE = {
    "ecg": None,
    "gesture": None,
    "pd": None,
}

TEST_RANGE = {
    "ecg": None,
    "gesture": None,
    "pd": None,
}


SHOW_PLOTS = True


def anomaly_intervals(labels):
    """Tìm tất cả các đoạn anomaly dạng [start, end)."""
    mask = (
        np.asarray(labels).reshape(-1) == 1
    ).astype(np.int8)

    changes = np.diff(np.r_[0, mask, 0])
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
    ]


def select_range(data, selected_range):
    if selected_range is None:
        start, end = 0, len(data)
    else:
        start, end = selected_range

    end = min(end, len(data))

    if not 0 <= start < end:
        raise ValueError(
            f"Khoảng ({start}, {end}) không hợp lệ; "
            f"chuỗi có {len(data)} timestep."
        )

    return data[start:end], start, end


def plot_dataset(dataset_name, filename, config):
    
    data_dir = (
        PROJECT_DIR
        / "dataset"
        / dataset_name
        / dataset_name
        / "labeled"
    )

    train_path = data_dir / "train" / filename
    test_path = data_dir / "test" / filename

    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy: {path}"
            )

    train_table = pd.DataFrame(
        pd.read_pickle(train_path)
    )
    test_table = pd.DataFrame(
        pd.read_pickle(test_path)
    )

    # ECG/Gesture: hai kênh; PD: một kênh.
    train_data = (
        train_table[config["features"]]
        .to_numpy(dtype=float)
    )
    test_data = (
        test_table[config["features"]]
        .to_numpy(dtype=float)
    )

    labels = (
        test_table[config["label"]]
        .to_numpy()
        .reshape(-1)
    )

    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError(
            f"{filename}: nhãn phải là 0 hoặc 1."
        )

    train, train_start, train_end = select_range(
        train_data,
        TRAIN_RANGE[dataset_name]
    )

    test, test_start, test_end = select_range(
        test_data,
        TEST_RANGE[dataset_name]
    )

    visible_labels = labels[test_start:test_end]

    print(f"\nDataset: {dataset_name} | File: {filename}")
    print(
        "Các đoạn anomaly [start, end):",
        anomaly_intervals(labels)
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 4.2),
        sharey=True
    )

    fig.suptitle(
        f"{dataset_name.upper()} — {Path(filename).stem}"
    )

    panels = [
        (
            axes[0], train, train_start, train_end,
            "Training time series"
        ),
        (
            axes[1], test, test_start, test_end,
            "Testing time series"
        ),
    ]

    colors = ["#1f77b4", "#ff7f0e"]

    for ax, data, start, end, caption in panels:
        time = np.arange(start, end)

        for channel in range(data.shape[1]):
            ax.plot(
                time,
                data[:, channel],
                color=colors[channel % len(colors)],
                linewidth=0.9
            )

        ax.set_xlim(start - 0.5, end - 0.5)
        ax.set_xlabel("Time (index)")
        ax.set_ylabel("Value")
        ax.tick_params(labelleft=True)
        ax.grid(False)

        ax.text(
            0.5, -0.29,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12
        )

    
    for start, end in anomaly_intervals(visible_labels):
        axes[1].axvspan(
            test_start + start - 0.5,
            test_start + end - 0.5,
            color="lightgreen",
            alpha=0.45,
            linewidth=0,
            zorder=0
        )

    if not np.any(visible_labels == 1):
        print("Đoạn test đang vẽ không có anomaly.")

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.27,
        top=0.86,
        wspace=0.30
    )

    # Lưu riêng vào figures/ecg, figures/gesture, figures/pd.
    save_dir = OUTPUT_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{Path(filename).stem}_train_test"

    fig.savefig(
        save_dir / f"{output_name}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        save_dir / f"{output_name}.pdf",
        bbox_inches="tight",
        facecolor="white"
    )

    print("Đã lưu PNG và PDF tại:", save_dir.resolve())

    if not SHOW_PLOTS:
        plt.close(fig)


def main():
    for dataset_name, config in DATASETS.items():
        if config["filename"] is None:
            train_dir = (
                PROJECT_DIR
                / "dataset"
                / dataset_name
                / dataset_name
                / "labeled"
                / "train"
            )

            filenames = [
                path.name
                for path in sorted(train_dir.glob("*.pkl"))
            ]

            if not filenames:
                raise FileNotFoundError(
                    f"Không có file .pkl trong: {train_dir}"
                )
        else:
            filenames = [config["filename"]]

        for filename in filenames:
            plot_dataset(
                dataset_name,
                filename,
                config
            )

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()