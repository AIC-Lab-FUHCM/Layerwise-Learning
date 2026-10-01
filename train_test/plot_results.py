from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def point_adjustment(labels, predicted):
    labels = np.asarray(labels).reshape(-1)

    adjusted = np.asarray(
        predicted,
        dtype=np.int64
    ).reshape(-1).copy()

    if len(labels) != len(adjusted):
        raise ValueError(
            "Nhãn thật và dự đoán phải cùng số timestep."
        )

    
    changes = np.diff(
        np.r_[0, (labels == 1).astype(int), 0]
    )

    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    for start, end in zip(starts, ends):
        if np.any(adjusted[start:end] == 1):
            adjusted[start:end] = 1

    return adjusted


def plot_test_results(
        result,
        save_dir,
        method_name,
        plot_range=None,
        save_test=False,
        show=True
):
    target = np.asarray(result["target"])
    reconstruction = np.asarray(result["reconstruction"])
    scores = np.asarray(result["score"]).reshape(-1)
    labels = np.asarray(result["label"]).reshape(-1)
    threshold = float(result["threshold"])

    
    if target.ndim != 2 or target.shape != reconstruction.shape:
        raise ValueError(
            "Target và reconstruction phải cùng shape [T, F]."
        )

    n = len(scores)

    if n == 0 or len(target) != n or len(labels) != n:
        raise ValueError(
            f"Lệch số timestep: target={len(target)}, "
            f"score={n}, label={len(labels)}."
        )

    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Nhãn anomaly phải là 0 hoặc 1.")

    if not np.isfinite(threshold) or not np.all(np.isfinite(scores)):
        raise ValueError("Score hoặc threshold chứa NaN/inf.")

    expected_scores = np.linalg.norm(
        reconstruction - target,
        axis=1
    )

    if not np.allclose(
            scores,
            expected_scores,
            rtol=1e-5,
            atol=1e-6
    ):
        raise ValueError(
            "Target, reconstruction và score chưa khớp timestep."
        )

    
    predicted_before = (scores >= threshold).astype(int)

    predicted_after = point_adjustment(
        labels,
        predicted_before
    )

    
    start, end = (
        (0, n) if plot_range is None else plot_range
    )
    end = min(end, n)

    if not 0 <= start < end:
        raise ValueError(
            "Khoảng thời gian cần vẽ không hợp lệ."
        )

    target = target[start:end]
    reconstruction = reconstruction[start:end]
    scores = scores[start:end]
    labels = labels[start:end]
    predicted_before = predicted_before[start:end]
    predicted_after = predicted_after[start:end]

    time = np.arange(start, end)

    
    changes = np.diff(
        np.r_[0, (labels == 1).astype(int), 0]
    )

    intervals = list(zip(
        np.flatnonzero(changes == 1),
        np.flatnonzero(changes == -1)
    ))

    colors = plt.get_cmap("tab10").colors

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def shade_anomalies(ax):
        for left, right in intervals:
            ax.axvspan(
                start + left - 0.5,
                start + right - 0.5,
                color="lightgreen",
                alpha=0.4,
                linewidth=0,
                zorder=0
            )

    def save_figure(fig, name):
        fig.savefig(
            save_dir / f"{name}.png",
            dpi=300,
            bbox_inches="tight"
        )
        fig.savefig(
            save_dir / f"{name}.pdf",
            bbox_inches="tight"
        )

    
    if save_test:
        fig_test, ax = plt.subplots(figsize=(12, 3.5))

        for channel in range(target.shape[1]):
            ax.plot(
                time,
                target[:, channel],
                color=colors[channel % len(colors)],
                linewidth=0.9,
                label=f"Channel {channel + 1}"
            )

        shade_anomalies(ax)
        ax.set(
            title="Test series — ground truth",
            xlabel="Time (index)",
            ylabel="Value"
        )
        ax.set_xlim(start - 0.5, end - 0.5)
        ax.legend(loc="upper right")

        fig_test.tight_layout()
        save_figure(fig_test, "test_series")

    
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        sharex=True
    )

    fig.suptitle(method_name.upper())

    
    for channel in range(target.shape[1]):
        color = colors[channel % len(colors)]

        axes[0].plot(
            time,
            target[:, channel],
            color=color,
            linewidth=0.9,
            label=f"Test {channel + 1}"
        )

        axes[0].plot(
            time,
            reconstruction[:, channel],
            color=color,
            linestyle="--",
            linewidth=0.9,
            label=f"Reconstruction {channel + 1}"
        )

    shade_anomalies(axes[0])
    axes[0].set(
        title="Test and reconstruction",
        ylabel="Value"
    )
    axes[0].legend(loc="upper right", ncol=2)

    
    axes[1].plot(
        time,
        scores,
        color="black",
        linewidth=0.9,
        label="Anomaly score"
    )

    axes[1].axhline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Threshold = {threshold:.4g}"
    )

    shade_anomalies(axes[1])
    axes[1].set(
        title="Anomaly score and threshold",
        ylabel="Score"
    )
    axes[1].legend(loc="upper right")

    
    
    prediction_rows = [
        (axes[2], predicted_before, "Before PA"),
        (axes[3], predicted_after, "After PA")
    ]

    for ax, prediction, stage in prediction_rows:
        ax.step(
            time,
            labels,
            where="mid",
            color="green",
            label="Ground truth"
        )

        ax.step(
            time,
            prediction,
            where="mid",
            color="red",
            linestyle="--",
            label=f"Prediction ({stage})"
        )

        ax.set(
            title=stage,
            ylabel="Label",
            yticks=[0, 1],
            ylim=(-0.1, 1.2)
        )

        ax.legend(loc="upper right")

    axes[3].set_xlabel("Time (index)")
    axes[3].set_xlim(start - 0.5, end - 0.5)

    fig.tight_layout()

    save_figure(
        fig,
        f"test_{method_name}"
    )

    print(
        f"Đã lưu hình gộp {method_name}:",
        save_dir.resolve()
    )

    if show:
        plt.show()