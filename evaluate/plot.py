import matplotlib.pyplot as plt
from .log import Log
from typing import Union, List, Tuple


def plot_rs(
    logs: Union[Log, List[Log]],
    π_labels: Union[Tuple[str], List[Tuple[str]], None] = None
):
    """
    Plots the average student reward and the teacher reward over time.

    params:
        logs -- the logs to plot
        π_labels -- the labels for the policies, each one is a tuple of the form
            (student policy label, teacher policy label)
    """
    if not isinstance(logs, list):
        logs = [logs]

        if π_labels is None:
            π_labels = [None] * len(logs)
        else:
            π_labels = [π_labels]

    # create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # plot the results
    for l, π_label in zip(logs, π_labels):
        # get the average student reward and teacher reward over time
        avg_sr = [s.avg_sr for s in l.history]
        t_r = [s.teacher_r for s in l.history]

        if π_label is None:
            π_label = (None, None)

        ax1.plot(avg_sr, label=π_label[0])
        ax2.plot(t_r, label=π_label[1])

    # set the labels
    ax1.set_ylabel("average student reward")
    ax1.set_xlabel("day")
    ax2.set_ylabel("teacher reward")
    ax2.set_xlabel("day")

    # set the legend
    ax1.legend()
    ax2.legend()

    # show the plot, with a tight layout
    fig.tight_layout()
    plt.show()
