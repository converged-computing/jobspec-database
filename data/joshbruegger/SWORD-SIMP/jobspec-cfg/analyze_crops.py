import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import time
from tqdm import tqdm


def read_label_files(folder, cropped):
    print(
        f"------------------------------------------\nReading label files from {folder}..."
    )
    classes = defaultdict(lambda: defaultdict(int))
    txt_files = [
        f
        for f in os.listdir(folder)
        if f.endswith(".txt") and not f.endswith("_classes.txt")
    ]

    for i, filename in enumerate(tqdm(txt_files, desc="Reading files")):
        painting_name = filename.rsplit(".", 1)[0].split("_", 1)[-1]
        if cropped:
            painting_name = "_".join(painting_name.rsplit("_", 1)[:-1])

        with open(os.path.join(folder, filename), "r") as file:
            for line in file:
                class_name = line.strip().split()[0]
                classes[class_name][painting_name] += 1

    return classes


def generate_freq_plots(
    freq_df1, df1, freq_df2, df2, output_path, painting_name, label1, label2
):
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(20, 0.5 * len(df1)), sharey=True
    )
    fig.tight_layout()

    freq_df1.plot(kind="barh", align="center", color="orange", ax=axes[0])
    freq_df2.plot(kind="barh", align="center", color="blue", ax=axes[1])

    axes[0].invert_xaxis()
    axes[0].set_xlabel(label1).set_fontweight("bold")
    axes[1].yaxis.set_tick_params(left=False, labelleft=False)
    axes[1].set_xlabel(label2).set_fontweight("bold")

    max_freq = max(freq_df1.max(), freq_df2.max()) + 1
    axes[0].set_xlim(max_freq, 0)
    axes[1].set_xlim(0, max_freq)

    for df, ax, side in zip([df1, df2], axes, [0.2, len(str(round(df2.max()))) * 0.6]):
        for i, v in enumerate(df):
            if v > 0 and (v / df.sum()) * 100 >= max_freq * 0.04:
                ax.text(
                    (v / df.sum()) * 100 - side,
                    i,
                    str(round(v)),
                    color="black",
                    va="center",
                )

    fig.supxlabel("Frequency (%)", x=0.57).set_fontweight("bold")
    fig.supylabel("Class", x=0.14, y=0.5).set_fontweight("bold")

    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.05, left=0.18, right=0.95)
    plt.savefig(output_path + f"freq_{painting_name}.png", bbox_inches="tight")
    plt.clf()


def generate_stacked(df, output_path):
    percentage_df = df.div(df.sum(axis=1), axis=0) * 100
    bar_plot = percentage_df.plot(
        kind="barh", stacked=True, legend=True, figsize=(10, 0.5 * len(df))
    )

    for patch in bar_plot.patches:
        width = patch.get_width()
        if width > 0:
            x = patch.get_x() + width / 2
            y = patch.get_y() + patch.get_height() / 2
            value = f"{width:.1f}%"
            y = y + patch.get_height() - 0.1 if len(value) >= width / 2 else y
            bar_plot.text(x, y, value, va="center", ha="center")

    plt.title("Percentage of classes for each painting")
    plt.ylabel("Class name")
    plt.legend(title="Paintings", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(output_path, bbox_inches="tight")
    plt.clf()


def generate_bar_chart(crop_classes, painting_classes, output_path):
    crop_df = pd.DataFrame(crop_classes).fillna(0).transpose()
    painting_df = pd.DataFrame(painting_classes).fillna(0).transpose()

    # Union operation
    all_classes = set(crop_df.index).union(painting_df.index)
    all_paintings = set(crop_df.columns).union(painting_df.columns)

    # Update both DataFrames
    crop_df = crop_df.reindex(index=all_classes, columns=all_paintings, fill_value=0)
    painting_df = painting_df.reindex(
        index=all_classes, columns=all_paintings, fill_value=0
    )

    # Sort index
    crop_df = crop_df.sort_index(ascending=False)
    painting_df = painting_df.sort_index(ascending=False)

    print("Generating stacked bar chart...")
    generate_stacked(crop_df, output_path + "stacked.png")

    print("Generating frequency plot for all paintings...")
    crop_freq_df = crop_df.sum(axis=1) / crop_df.sum(axis=1).sum() * 100
    painting_freq_df = painting_df.sum(axis=1) / painting_df.sum(axis=1).sum() * 100

    generate_freq_plots(
        crop_freq_df,
        crop_df.sum(axis=1),
        painting_freq_df,
        painting_df.sum(axis=1),
        output_path,
        "all",
        "Crops",
        "Painting",
    )

    for painting_name in crop_df.columns:
        print(f"Generating frequency plot for painting {painting_name}...")
        crop_freq_df = crop_df[painting_name] / crop_df[painting_name].sum() * 100
        painting_freq_df = (
            painting_df[painting_name] / painting_df[painting_name].sum() * 100
        )

        generate_freq_plots(
            crop_freq_df,
            crop_df[painting_name],
            painting_freq_df,
            painting_df[painting_name],
            output_path,
            painting_name,
            "Crops",
            "Painting",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("location", help="Location of the dataset")
    args = parser.parse_args()

    crop_classes = read_label_files(
        os.path.join(args.location, "cropped", "labels"), True
    )
    painting_classes = read_label_files(os.path.join(args.location, "labels"), False)

    output_path = os.path.join(args.location, "cropped", "analysis/")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    generate_bar_chart(crop_classes, painting_classes, output_path)


if __name__ == "__main__":
    main()
