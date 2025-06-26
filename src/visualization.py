import matplotlib.pyplot as plt
import seaborn as sns


def plot_label_distribution(
    df,
    column: str = "label",
    palette: str = "Set2",
    figsize: tuple = (8, 6),
    xlabel: str = None,
    ylabel: str = "Count",
    title: str = None,
    annotate_offset: float = 0.5,
):
    """
    Plots a bar chart showing the distribution of values in a selected DataFrame column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    column : str, default 'label'
        Name of the column whose distribution should be visualized.
    palette : str, default 'Set2'
        Seaborn color palette.
    figsize : tuple, default (8, 6)
        Plot size (width, height).
    xlabel : str, optional
        X-axis label. If None, the column name is used.
    ylabel : str, default 'Count'
        Y-axis label.
    title : str, optional
        Plot title. If None, 'Distribution of "<column>"' is used.
    annotate_offset : float, default 0.5
        Offset for text values above bars.
    """
    # Data preparation
    counts = df[column].value_counts(dropna=False)

    labels = counts.index.astype(str)
    values = counts.values

    # Plot settings
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    palette_colors = sns.color_palette(palette, n_colors=len(values))

    ax = sns.barplot(x=labels, y=values, palette=palette_colors, hue=labels)

    # Axis labels and title
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'Distribution of "{column}"')

    # Adding values above bars
    for idx, value in enumerate(values):
        ax.text(idx, value + annotate_offset, f"{value}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()
