import matplotlib.pyplot as plt
from math import ceil, pi
from pathlib import Path


def radar_plot(
    labels: list[str], data: list[float], output_file: Path, title: str, max_value: float | None = None
) -> None:
    """Builds and save a radar plot.

    Args:
        labels (list): List of labels.
        data (list): List of values.
        output_file (Path): Path to the output file.
        title (str): Title of the plot.
        max_value (float, optional): Maximum value. Default: None.
    """
    num_variables = len(labels)
    cycled_data = data + data[:1]
    max_value = max_value or max(cycled_data)

    angles = [n / float(num_variables) * 2 * pi for n in range(num_variables)]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], labels, color="black", size=10)
    plt.yticks(
        [ceil(max_value / 2)],
        [str(ceil(max_value / 2))],
        color="grey",
        size=7,
    )
    plt.ylim(0, max_value)
    ax.plot(angles, cycled_data, linewidth=1, linestyle="solid")
    ax.fill(angles, cycled_data, "b", alpha=0.1)
    plt.title(title)
    plt.savefig(output_file)
    plt.clf()
