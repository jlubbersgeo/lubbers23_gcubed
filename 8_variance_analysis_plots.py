""" 
plotting the results frome variance_analysis.py

"""

import sys

import cartopy.crs as ccrs  # for the subplot mosaic of maps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ternary
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from rich.console import Console
from rich.prompt import Prompt
from rich.theme import Theme

# custom plotting defaults
import mpl_defaults

custom_theme = Theme(
    {"main": "bold gold1", "path": "bold steel_blue1", "result": "magenta"}
)
console = Console(theme=custom_theme)

export_path = Prompt.ask("[bold gold1] Enter the path to where overall results should be exported[bold gold1]")
export_path = export_path.replace('"',"")

data_path = Prompt.ask("[bold gold1] Enter the folder path to where transformed data are stored[bold gold1]")
data_path = data_path.replace('"',"") 


training_df = pd.read_excel(
    f"{data_path}\B4_training_data_transformed_v2.xlsx"
).set_index('volcano')
trace_df = pd.read_excel(f"{data_path}\\vcs_rfe_variance_results.xlsx").set_index('eruption')
major_df = pd.read_excel(f"{data_path}\\vcs_major_variance_results.xlsx").set_index('eruption')
trace_df.insert(0,'type','trace_elements')
major_df.insert(0,'type','major_elements')


volcanoes = trace_df.loc[:,'Adagdak':'Veniaminof'].columns.tolist()
eruptions = trace_df.index.unique().tolist()

lats = training_df.loc[volcanoes]['latitude'].unique()
lons = training_df.loc[volcanoes]['longitude'].unique()


combined_df = pd.concat(
    [
        trace_df.loc[:, ["type", "Target", "Prediction"] + volcanoes],
        major_df.loc[:, ["type", "Target", "Prediction"] + volcanoes],
    ]
)


combined_df_melted = (
    combined_df.reset_index()
    .melt(id_vars=["eruption", "Target", "Prediction", "type"], value_vars=volcanoes)
    .set_index("eruption")
)

########################################################################
########################## MANUSCRIPT FIGURE 7 #########################
########################################################################

fig,ax = plt.subplots(3,1,figsize = (8,12))
axes = ax.ravel()
eruptions = ['GrayPtDacite','Churchill_C2','ELVC_C1']
labels = ["Gray point dacite", "C2", "C1"]
locations = [6,5,7]


for eruption, a,label,location in zip(eruptions,axes,labels,locations):
    sns.boxplot(
        data=combined_df_melted.loc[eruption, :],
        x="variable",
        y="value",
        hue="type",
        width=0.75,
        boxprops=dict(edgecolor="k", linewidth=1, alpha=1),
        medianprops=dict(color="k", lw=1),
        capprops=dict(linewidth=0),
        whiskerprops=dict(color="k", lw=0.5, linestyle="--"),
        showfliers=False,
        ax=a,
    )

    xticklabels = a.get_xticklabels()
    if a == axes[2]:

        a.set_xticklabels(xticklabels, rotation=90)
    else:
        a.set_xticklabels([])
    a.set_ylabel("Probability")
    a.set_xlabel("")
    a.minorticks_off()
    a.text(.65,.7,
        f'Volcano: {combined_df_melted.loc[eruption,"Target"].unique()[0]}\nEruption:{label}\nn = {trace_df.loc[eruption,:].shape[0]}',
        fontsize = 12,
        transform = a.transAxes
        
    )
    legend_elements = [Patch(facecolor='C0', edgecolor='k',
                        label='RFE Features'),
                Patch(facecolor='C1', edgecolor='k',
                        label='Major element concentrations')]
    a.legend([],frameon = False)

fig.legend(handles = legend_elements,bbox_to_anchor = (0.9,0.91),ncol = 2)
mpl_defaults.label_subplots(ax,location = 'upper left')
plt.savefig(
        "{}\\vcs_variance_comparison_panel.pdf".format(export_path),
        bbox_inches="tight",
    )

plt.show(block = False)

########################################################################
########################## MANUSCRIPT FIGURE 6 #########################
########################################################################
eruptions = trace_df.index.unique().tolist()

major_medians = []
major_stds = []

trace_medians = []
trace_stds = []

correct_volcano = []
trace_predictions = []
major_predictions = []

for eruption in eruptions:
    major_medians.append(major_df.loc[eruption,major_df.loc[eruption,'Target'].unique()].median().values[0])
    major_stds.append(major_df.loc[eruption,major_df.loc[eruption,'Target'].unique()].std().values[0])
    correct_volcano.append(major_df.loc[eruption,'Target'].unique().tolist()[0])


    
    trace_medians.append(trace_df.loc[eruption,trace_df.loc[eruption,'Target'].unique()].median().values[0])
    trace_stds.append(trace_df.loc[eruption,trace_df.loc[eruption,'Target'].unique()].std().values[0])

    trace_preds = pd.DataFrame(
        data=trace_df.loc[eruption, "Prediction"].value_counts(),
    ).reset_index()
    trace_preds.columns = ["predicted_class", "counts"]

    trace_prediction = list(
        trace_preds[trace_preds["counts"] == trace_preds["counts"].max()]["predicted_class"]
    )[0]

    major_preds = pd.DataFrame(
        data=major_df.loc[eruption, "Prediction"].value_counts(),
    ).reset_index()
    major_preds.columns = ["predicted_class", "counts"]

    major_prediction = list(
        major_preds[major_preds["counts"] == major_preds["counts"].max()]["predicted_class"]
    )[0]
    major_predictions.append(major_prediction)
    trace_predictions.append(trace_prediction)

    
major_medians = np.array(major_medians)
major_stds = np.array(major_stds)

trace_medians = np.array(trace_medians)
trace_stds = np.array(trace_stds)

df = pd.DataFrame(
    [
        eruptions,
        correct_volcano,
        major_predictions,
        trace_predictions,
        trace_medians,
        trace_stds,
        major_medians,
        major_stds,
    ]
).T
df.columns = [
    "eruption",
    "volcano",
    "major_prediction",
    "trace_prediction",
    "trace_median",
    "trace_std",
    "major_median",
    "major_std",
]
df.sort_values(by="volcano", inplace=True)
df.set_index("volcano", inplace=True)
df["x_maj"] = 0
df["x_trace"] = 0

spacing = 10
for i, volcano in zip(
    np.arange(0, len(volcanoes)) * spacing, df.index.unique().tolist()
):

    df.loc[volcano, "x_trace"] = i
    df.loc[volcano, "x_maj"] = i + spacing / 2

bar_vals = np.unique(df["x_trace"].to_numpy())
bar_vals = np.append(bar_vals, bar_vals.max() + spacing)


fig, ax = plt.subplots(figsize=(12, 4))
for volcano in df.index.unique().tolist():

    for i in range(df.loc[volcano, :].shape[0]):
        ax.plot(
            (df.loc[volcano, "x_maj"], df.loc[volcano, "x_trace"]),
            (df.loc[volcano, "major_median"], df.loc[volcano, "trace_median"]),
            "k-",
            lw=0.3,
        )

ax.plot(
    df["x_trace"],
    df["trace_median"],
    marker="o",
    ls="",
    label="RFE Features",
    ms=5,
    mew=0.5,
)
ax.plot(
    df["x_maj"],
    df["major_median"],
    marker="o",
    ls="",
    label="Major elements",
    ms=5,
    mew=0.5,
)


ax.set_xticks(np.arange(0, len(volcanoes)) * spacing + spacing / 3)
ax.minorticks_off()
ax.set_xticklabels(df.index.unique().tolist(), rotation=90)


for i, j in zip(bar_vals[0::2] - 2, bar_vals[1::2] - 2):
    ax.axvspan(xmin=i, xmax=j, color="gray", alpha=0.3)


for i in range(df[df["major_prediction"] != df.index].dropna().shape[0]):
    ax.plot(
        df[df["major_prediction"] != df.index].dropna()["x_maj"],
        df[df["major_prediction"] != df.index].dropna()["major_median"],
        ms=8,
        mfc="none",
        mec="r",
        marker="s",
        ls="",
    )

for i in range(df[df["trace_prediction"] != df.index].dropna().shape[0]):
    ax.plot(
        df[df["trace_prediction"] != df.index].dropna()["x_trace"],
        df[df["trace_prediction"] != df.index].dropna()["trace_median"],
        ms=8,
        mfc="none",
        mec="r",
        marker="s",
        ls="",

    )


ax.set_ylabel("Probability")
fig.legend(bbox_to_anchor=(0.9, 0.98), ncol=2, frameon=True)
ax.set_title("Target Prediction Probabilities", loc="left", fontsize=20)

plt.savefig(
    "{}\\vcs_variance_test_comparison_target.pdf".format(export_path), bbox_inches="tight"
)

plt.show(block = False)

########################################################################
########################## MANUSCRIPT FIGURE 8 #########################
########################################################################

def tern_points(right, top, left):
    """Tern_points takes 3 equal size 1D arrays or pandas series and organizes them into points to be plotted on a ternary
         with the following arrangement:(lower right,top,lower left).
             Inputs: 
             x = 1D array like (lower right vertex)
             y = 1D array like (top vertex)
             z = 1D array like (lower left vertex)
    """
    if isinstance(right, pd.Series):
        right = right.to_numpy()
    if isinstance(top, pd.Series):
        top = top.to_numpy()
    if isinstance(left, pd.Series):
        left = left.to_numpy()

    points = np.hstack([right[:, None], top[:, None], left[:, None]])

    return points


fig, ax = plt.subplots(3, 1, figsize=(4, 12))
scale = 1
tax_top = ternary.TernaryAxesSubplot(ax=ax[0])
tax_middle = ternary.TernaryAxesSubplot(ax=ax[1])
tax_bottom = ternary.TernaryAxesSubplot(ax=ax[2])

subset_cols = ["Veniaminof", "Makushin", "Aniakchak"]
colors = mpl_defaults.create_colorblind_palette(n=2)
for source, tax in zip(
    ["Aniakchak", "Veniaminof", "Makushin"], [tax_top, tax_middle, tax_bottom]
):

    major_subset = (
        major_df.reset_index()
        .set_index("Target")
        .loc[source, ["Makushin", "Aniakchak", "Veniaminof"]]
    )
    major_subset["sum"] = major_subset.sum(axis="columns")
    trace_subset = (
        trace_df.reset_index()
        .set_index("Target")
        .loc[source, ["Makushin", "Aniakchak", "Veniaminof"]]
    )
    trace_subset["sum"] = trace_subset.sum(axis="columns")

    major_points_to_plot = tern_points(
        major_subset[subset_cols[0]] / major_subset["sum"],
        major_subset[subset_cols[1]] / major_subset["sum"],
        major_subset[subset_cols[2]] / major_subset["sum"],
    )
    trace_points_to_plot = tern_points(
        trace_subset[subset_cols[0]] / trace_subset["sum"],
        trace_subset[subset_cols[1]] / trace_subset["sum"],
        trace_subset[subset_cols[2]] / trace_subset["sum"],
    )
    if source == 'Aniakchak':

        tax.scatter(major_points_to_plot, c="none", ec="C1", marker="o", s=15, label = 'Major Elements')
        tax.scatter(trace_points_to_plot, c="none", ec="C0", marker="o", s=15, label = 'RFE Features')
    else:
        tax.scatter(major_points_to_plot, c="none", ec="C1", marker="o", s=15, )
        tax.scatter(trace_points_to_plot, c="none", ec="C0", marker="o", s=15, )

    tax.right_corner_label(subset_cols[0], offset=0.25, rotation=-60)
    tax.top_corner_label(subset_cols[1], offset=0.25)
    tax.left_corner_label(subset_cols[2], offset=0.25, rotation=60)
    
    tax.gridlines(color="gray", multiple=0.2, linestyle="--",  dashes=(5, 5), zorder=0)
    tax.boundary(linewidth=1.5, zorder=0)
    # Set ticks
    tax.ticks(axis="lbr", multiple=0.2, tick_formats="%.1f", offset=0.025, linewidth=1)
    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
fig.set_facecolor("w")
fig.legend(loc = 'upper right', title = 'Model Features', title_fontsize = 14, bbox_to_anchor = (1.05,.95))
plt.savefig(
    "{}\\ani_veni_mak_variance_ternary.pdf".format(export_path), bbox_inches="tight"
)
plt.show(block = True)
