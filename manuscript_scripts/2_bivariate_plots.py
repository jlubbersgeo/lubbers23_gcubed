import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aleutian_colors import create_aleutian_colors
import seaborn as sns
import mpl_defaults


# where all the figures get dumped
export_path = r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs"

data = pd.read_excel(
    r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Data\B4_training_data_cleaned.xlsx"
).set_index('volcano')
transformed_data = pd.read_excel(
    r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Data\B4_training_data_transformed_v2.xlsx"
).set_index('volcano')

volcanoes = data.index.unique().tolist()

dark_mode = False
color_dict = create_aleutian_colors()

#######################################################################
####################### MANUSCRIPT FIGURE 2 ###########################
#######################################################################
plot_pairs = [["Si_ppm", "K_ppm"], ["Sr", "Ba"], ["La/Yb", "Nb/U"]]
plot_labels = [["Si [wt%]", "K [wt%]"], ["Sr [ppm]", "Ba [ppm]"], ["La/Yb", "Nb/U"]]

fig, ax = plt.subplots(3, 2, figsize=(8, 12), layout="tight")
axes = ax.ravel()
for i in range(len(plot_pairs)):
    if i == 0:
        for c in color_dict:
            ax[i, 0].plot(
                data.loc[c, plot_pairs[i][0]] / 1e4,
                data.loc[c, plot_pairs[i][1]] / 1e4,
                **color_dict[c],
                ls="",
                label=c,
                ms=5,
            )
            ax[i, 1].plot(
                transformed_data.loc[c, plot_pairs[i][0]],
                transformed_data.loc[c, plot_pairs[i][1]],
                **color_dict[c],
                ls="",
                ms=5,
            )
        ax[i, 0].set_xlabel(plot_labels[i][0])
        ax[i, 0].set_ylabel(plot_labels[i][1])
        ax[i, 1].set_xlabel(f"clr[{plot_labels[i][0].replace(' [wt%]','')}]")
        ax[i, 1].set_ylabel(f"clr[{plot_labels[i][1].replace(' [wt%]','')}]")
    else:
        for c in color_dict:
            ax[i, 0].plot(
                data.loc[c, plot_pairs[i][0]],
                data.loc[c, plot_pairs[i][1]],
                **color_dict[c],
                ls="",
                ms=5,
            )
            ax[i, 1].plot(
                transformed_data.loc[c, plot_pairs[i][0]],
                transformed_data.loc[c, plot_pairs[i][1]],
                **color_dict[c],
                ls="",
                ms=5,
            )
        ax[i, 0].set_xlabel(plot_labels[i][0])
        ax[i, 0].set_ylabel(plot_labels[i][1])
        ax[i, 1].set_xlabel(f"clr[{plot_labels[i][0].replace(' [ppm]','')}]")
        ax[i, 1].set_ylabel(f"clr[{plot_labels[i][1].replace(' [ppm]','')}]")
fig.legend(
    loc="upper center",
    bbox_to_anchor=(0.54, 1.1),
    title="Volcano",
    title_fontsize=14,
    ncol=5,
    markerscale=1.5,
)

plt.savefig(
    "{}\\LTPE_concentration_vs_transform_panel.pdf".format(export_path),
    bbox_inches="tight",
)


if dark_mode is True:

    for a in axes:
        mpl_defaults.make_dark_bkgd_compatible(ax=a, bkgd_color="none")
        plt.savefig(
            "{}\\LTPE_concentration_vs_transform_panel_darkbkgd.pdf".format(
                export_path
            ),
            bbox_inches="tight",
        )
        plt.savefig(
            "{}\\LTPE_concentration_vs_transform_panel_darkbkgd.png".format(
                export_path
            ),
            bbox_inches="tight",
        )
plt.show()