""" 
This script uses the model produced in 5_final_voting_classifier.py
and uses it to predict the source volcano for each sample analyzed in
the IODP cores U1417 and U1418. It organizes them by eruption (e.g.,
if they are the same age and composition) and plots the probabilistic
output in the form of box and whisker plots of each eruption.

It then plots the cumulative probability of each volcano in the model
for every observation in the IODP dataset to show which the most likely
volcanic sources are over the last 800ka.


"""
import pickle
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from cartopy.mpl.gridliner import Gridliner
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyrolite.plot import density, pyroplot
from rich.console import Console
from rich.prompt import Prompt
from rich.theme import Theme
from scipy.stats import ks_2samp, ttest_ind
from tqdm import tqdm

# custom plotting defaults
import mpl_defaults

custom_theme = Theme(
    {"main": "bold gold1", "path": "bold steel_blue1", "result": "magenta"}
)
console = Console(theme=custom_theme)

export_path = Prompt.ask(
    "[bold gold1] Enter the path to where overall results should be exported[bold gold1]"
)
export_path = export_path.replace('"', "")

data_path = Prompt.ask(
    "[bold gold1] Enter the folder path to where transformed data are stored[bold gold1]"
)
data_path = data_path.replace('"', "")


vcs_load = pickle.load(
    open(f"{data_path}\Aleutian_tuned_vcs_classifier_trace_deployment.sav", "rb")
)

iodp_data = pd.read_excel(f"{data_path}\IODP_data_transformed.xlsx").set_index(
    "IODP_sample"
)
iodp_samples = iodp_data.index.unique().tolist()
iodp_data.insert(
    0, "Site", [sample.split("-")[0] for sample in iodp_data.index.tolist()]
)
iodp_data.insert(
    1, "Hole", [sample.split("-")[1] for sample in iodp_data.index.tolist()]
)
iodp_ages = [iodp_data.loc[sample, "Age (yrs)"].unique()[0] for sample in iodp_samples]
iodp_ages = np.array(iodp_ages).astype("float")
myfeatures = [
    "Cs",
    "Nb",
    "Nb/U",
    "Sr/Y",
    "Th/La",
    "Si_ppm",
    "Ba",
    "Pb",
    "Sr",
    "Rb/Sm",
    "La/Yb",
    "K_ppm",
]


soft_predictions = pd.DataFrame(
    vcs_load.predict_proba(iodp_data.loc[:, myfeatures]), columns=vcs_load.classes_
)
soft_predictions.insert(
    0, "predicted_class", vcs_load.predict(iodp_data.loc[:, myfeatures])
)
soft_predictions.index = iodp_data.index
soft_predictions.index.name = "IODP_sample"
soft_predictions.insert(0, "Age (yrs)", iodp_data["Age (yrs)"])
samples = soft_predictions.index.unique()

training_data = pd.read_excel(
    f"{data_path}\B4_training_data_transformed_v2.xlsx"
).set_index("volcano")

volcanoes = training_data.index.unique().tolist()

sorted_locations = [
    "Churchill",
    "Hayes",
    "Redoubt",
    "Augustine",
    "Kaguyak",
    "Katmai",
    "Ugashik",
    "Aniakchak",
    "Black Peak",
    "Veniaminof",
    "Emmons Lake",
    "Fisher",
    "Makushin",
    "Okmok",
    "Adagdak",
    "Gareloi",
    "Semisopochnoi",
    "Davidof",
]
sorted_abbreviations = [
    "CU",
    "HA",
    "RD",
    "AU",
    "KG",
    "KT",
    "UG",
    "AN",
    "BP",
    "VE",
    "EM",
    "FI",
    "MA",
    "OK",
    "AD",
    "GA",
    "SS",
    "DV",
]

# grouping based on age such that these should
# be individual eruptions
groups = {
    "0.1": ["U1417D-1H-1W_3-5", "U1417D-1H-1W_3-4", "U1417B-2H-1W_2-4"],  # 100 yrs
    "18": ["U1418A-5H-5W_81-82"],  # 17.7ka
    "27": ["U1418D-8H-4W_30-31", "U1418A-8H-1W_51-52"],  # 27ka
    "57": ["U1417D-2H-5W_13-14"],  # 57ka
    "129": ["U1418A-24H-4W_66-69", "U1418E-12H-5W_29-31"],  # 129 ka
    "129-2": ["U1418A-24H-CCW_1-3", "U1418E-12H-5W_16-18"],  # 129ka
    "150": ["U1418D-22H-1W_103-104"],  # 150ka
    "164": [
        "U1417D-4H-4W_40-42",
        "U1417A-4H-3W_139-141",
        "U1417B-4H-5W_28-30",
    ],  # 164ka
    "166": ["U1417C-5H-2W_12-14"],  # 166ka
    "200": [
        "U1417D-5H-1W_147-148",
        "U1417C-5H-4W_139-141",
        "U1417D-4H-CCW_9-10",
        "U1417A-4H-7W_50-52",
        "U1417B-5H-1W_125-127",
        "U1417B-5H-1W_135-137",
        "U1417B-5H-1W_145-147",
        "U1417B-5H-2W_4-6",
        "U1417B-5H-2W_13-15",
    ],  # 200ka
    "280": ["U1417A-7H-5W_90-92"],  # 280ka
    "507": ["U1417A-9H-3W_45-47"],  # 507ka
    "529": ["U1417A-9H-4W_14-16"],  # 529ka
    "568": ["U1418F-29R-2W_20-22"],  # 568ka
    "570": ["U1418F-29R-2W_88-90", "U1418F-29R-4W_44-46"],  # 570ka
    "570-2": ["U1418F-29R-3W_34-36"],  # 570ka
    "591": ["U1417C-8H-5W_34-35", "U1417D-8H-5W_98-100"],  # 591ka
    "707": [
        "U1417C-9H-4W_139-140",
        "U1417A-10H-6W_4-5",
        "U1417A-10H-6W_9-10",
        "U1417D-9H-5W_95-97",
        "U1417B-9H-3W_40-42",
    ],  # 707ka
    "734": ["U1417B-10H-6W_42-44"],  # 734ka
    "749": [
        "U1417B-11H-2W_147-149",
        "U1417C-11H-6W_85-87",
        "U1417B-11H-4W_5-7",
        "U1417D-13H-4W_141-143",
    ],  # 749ka
}

############################################################
#################### FIGURE 10 ############################
############################################################
fig, ax = plt.subplots(5, 4, figsize=(12, 10), layout="constrained")
axes = ax.ravel()
for g, a, i in zip(groups, axes, range(len(axes))):
    df1 = pd.DataFrame(
        data=soft_predictions.loc[groups[g], "predicted_class"].value_counts(),
    ).reset_index()
    df1.columns = ["predicted_class", "counts"]

    best_soft_prediction = list(
        df1[df1["counts"] == df1["counts"].max()]["predicted_class"]
    )[0]

    box = a.boxplot(
        soft_predictions.loc[groups[g], sorted_locations],
        boxprops=dict(color="k", facecolor="lightgray", alpha=0.8),
        medianprops=dict(color="k", lw=1),
        capprops=dict(linewidth=0),
        whiskerprops=dict(color="k"),
        flierprops=dict(markeredgecolor="dimgray", alpha=0.7, markersize=3),
        showfliers=True,
        patch_artist=True,
    )
    for patch, marker, median, volcano in zip(
        box["boxes"], box["fliers"], box["medians"], sorted_locations
    ):
        if volcano == best_soft_prediction:
            patch.set_facecolor("darksalmon")
            marker.set_markerfacecolor("lightsalmon")
            # median.set_color("red")

        else:
            patch.set_facecolor("none")
            marker.set_markerfacecolor("none")
            marker.set_markeredgewidth(0.5)
    # if i % 3 == 0:

    #     a.set_ylabel("Probability")

    a.set_xticks(np.arange(1, len(sorted_locations) + 1))
    if i >= 16:
        a.set_xticklabels(sorted_locations, rotation=90)
    else:
        a.set_xticklabels(sorted_abbreviations, rotation=90, fontsize=8)
    a.minorticks_off()
    a.set_ylim(0, 1)

    if i in [0, 1, 5, 6, 7, 8, 9, 13, 14, 15, 17, 18, 19]:
        a.text(
            0.47,
            0.63,
            f"{best_soft_prediction}\nP = {np.round(soft_predictions.loc[groups[g], best_soft_prediction].median(),2)}",
            transform=a.transAxes,
            fontsize=12,
        )
        a.text(
            0.47,
            0.88,
            f"Age: {g} ka",
            transform=a.transAxes,
            fontsize=12,
        )
    else:
        a.text(
            0.03,
            0.63,
            f"{best_soft_prediction}\nP = {np.round(soft_predictions.loc[groups[g], best_soft_prediction].median(),2)}",
            transform=a.transAxes,
            fontsize=12,
        )
        a.text(
            0.03,
            0.88,
            f"Age: {g} ka",
            transform=a.transAxes,
            fontsize=12,
        )

    # a.grid(axis="x", ls="--", dashes=(5, 10))
fig.supylabel("Probability", fontsize=24)


plt.savefig(
    "{}\iodp_prediction_boxplot_panel_byeruption.pdf".format(export_path),
    bbox_inches="tight",
)
plt.show(block=False)

########################################################
################### FIGURE 11 ##########################
########################################################
lats = []
lons = []
for volcano in sorted_locations:
    lat = training_data.loc[volcano, "latitude"].mean()
    lon = training_data.loc[volcano, "longitude"].mean()
    if lon > 0:
        lon = (lon - 180) - 180
    lats.append(lat)
    lons.append(lon)

lats = np.array(lats)
lons = np.array(lons)
# bounding box for our map
extent = [175, 215, 50, 63]

# central longitude to help deal with int. dateline
center_longitude = 180

sct_params = {
    "ec": "k",
    "marker": "o",
    "cmap": "magma_r",
    "s": 100,
    "zorder": 10,
}

fig, ax = plt.subplots(
    figsize=(8, 6),
    subplot_kw={"projection": ccrs.AlbersEqualArea(central_longitude=center_longitude)},
    layout="constrained",
)

x = sorted_abbreviations
y = pd.DataFrame(soft_predictions.loc[:, sorted_locations].sum()).T

ax.set_extent(extent)
ax.coastlines(linewidth=1)


# add land and ocean features
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN)

# add thick border around edge
ax.spines["geo"].set_linewidth(2)

# add scatter at each location sized and colored by the probability
scatter = ax.scatter(
    x=lons,
    y=lats,
    c=y.to_numpy(),
    transform=ccrs.PlateCarree(),
    **sct_params,
)


# for lon,lat,label in zip(lons,lats, sorted_abbreviations):
#     ax.text(lon,lat, label, transform = ccrs.PlateCarree(),zorder = 10)

gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=1,
    linestyle="--",
    color="gray",
    alpha=0.5,
)
gl.xlocator = mticker.FixedLocator([-180, -170, -160, -150, -140])
gl.top_labels = False
gl.left_labels = False
gl.bottom_labels = False
gl.right_labels = False

cbar = fig.colorbar(
    scatter,
    shrink=0.3,
    ax=ax,
    ticks=[y.to_numpy().min(), y.to_numpy().max()],
    orientation="horizontal",
    anchor=(0.95, -20),
)
cbar.set_ticklabels(["Low", "High"])

ax.text(0.676, 0.15, "Cumulative Probability", fontsize=16, transform=ax.transAxes)
# u1417 lat: 56°57.5996′N lon: 147°6.5985′W
ax.plot(
    -147.11, 56.959, marker="*", ms=12, mec="k", mfc="k", transform=ccrs.PlateCarree()
)
ax.text(-147.11 - 3, 56.959 - 0.5, "U1417", transform=ccrs.PlateCarree(), fontsize=18)


# U1418 lat: 58°46.6095′N lon: 144°29.5777′W
ax.plot(
    -144.492, 58.777, marker="*", ms=12, mec="k", mfc="k", transform=ccrs.PlateCarree()
)
ax.text(-144.492 - 3, 58.777 - 0.5, "U1418", transform=ccrs.PlateCarree(), fontsize=18)
ax.text(-180, 57, "Bering Sea", transform=ccrs.PlateCarree(), fontsize=20)
ax.text(-155, 53, "Gulf of Alaska", transform=ccrs.PlateCarree(), fontsize=20)
ax.text(-158, 62, "Alaska", transform=ccrs.PlateCarree(), fontsize=20)
plt.savefig(
    "{}\\final_IODP_probababilities_map.pdf".format(export_path), bbox_inches="tight"
)
plt.show(block=False)

#############################################################
###################### FIGURE 12 ############################
######or at least the pair plots that were the foundation######

vars = [
    "Nb/U",
    "Sr/Y",
    "Th/La",
    "La/Yb",
    "Rb/Sm",
]
n = len(vars)
colorblind_colors = mpl_defaults.create_colorblind_palette(n=3)
train_color = "lightgray"
train_pred_color = colorblind_colors[0]
train_second_color = colorblind_colors[2]
iodp_color = colorblind_colors[1]

for g in tqdm(["27", "591", "570-2"]):
    df1 = pd.DataFrame(
        data=soft_predictions.loc[groups[g], "predicted_class"].value_counts(),
    ).reset_index()
    df1.columns = ["predicted_class", "counts"]

    best_soft_prediction = list(
        df1[df1["counts"] == df1["counts"].max()]["predicted_class"]
    )[0]

    second_best_prediction = None

    if df1.shape[0] > 1:
        second_best_prediction = df1.loc[1, "predicted_class"]

    fig, ax = plt.subplots(n, n, figsize=(3 * n, 3 * n))

    upper_indices = np.array(np.where(np.triu(ax, k=1))).T
    lower_indices = np.array(np.where(np.tril(ax, k=-1))).T
    diag_indices = np.array(np.where(np.eye(ax.shape[0], dtype=bool))).T
    all_indices = np.vstack([upper_indices, lower_indices, diag_indices])

    # plot the diagonals
    for coords in diag_indices:
        hist, bins = np.histogram(
            training_data.loc[:, vars[coords[0]]], bins=20, density=True
        )

        sns.histplot(
            data=training_data.loc[best_soft_prediction, :],
            x=vars[coords[0]],
            element="step",
            stat="density",
            bins=bins,
            color=train_pred_color,
            ax=ax[coords[0], coords[1]],
            alpha=0.5,
        )
        if second_best_prediction:
            sns.histplot(
                data=training_data.loc[second_best_prediction, :],
                x=vars[coords[0]],
                element="step",
                stat="density",
                bins=bins,
                color=train_second_color,
                ax=ax[coords[0], coords[1]],
                alpha=0.5,
            )

        sns.histplot(
            data=iodp_data.loc[groups[g], :],
            x=vars[coords[0]],
            element="step",
            stat="density",
            bins=bins,
            color=iodp_color,
            ax=ax[coords[0], coords[1]],
            alpha=0.5,
        )
        mpl_defaults.bottom_only_axes(ax[coords[0], coords[1]])

    # remove the upper triangle axes
    for coords in upper_indices:
        ax[coords[0], coords[1]].axis("off")
    ax[1, 3].axis("on")

    # plot the lower triangle axes
    for coords in lower_indices:
        # scatter plots for both groups
        ax[coords[0], coords[1]].plot(
            training_data.loc[:, vars[coords[1]]],
            training_data.loc[:, vars[coords[0]]],
            marker=".",
            ms=7,
            alpha=0.5,
            mfc=train_color,
            mec="none",
            ls="",
            label="all training data",
        )

        ax[coords[0], coords[1]].plot(
            training_data.loc[best_soft_prediction, vars[coords[1]]],
            training_data.loc[best_soft_prediction, vars[coords[0]]],
            marker=".",
            ms=7,
            alpha=0.5,
            mfc=train_pred_color,
            mec="none",
            ls="",
            label=f"{best_soft_prediction} training data",
        )
        if second_best_prediction:
            ax[coords[0], coords[1]].plot(
                training_data.loc[second_best_prediction, vars[coords[1]]],
                training_data.loc[second_best_prediction, vars[coords[0]]],
                marker=".",
                ms=7,
                alpha=0.5,
                mfc=train_second_color,
                mec="none",
                ls="",
                label=f"{second_best_prediction} training data",
            )

        ax[coords[0], coords[1]].plot(
            iodp_data.loc[groups[g], vars[coords[1]]],
            iodp_data.loc[groups[g], vars[coords[0]]],
            marker=".",
            ms=7,
            alpha=0.7,
            mfc=iodp_color,
            mec="none",
            ls="",
            label="sample data",
        )
        # pyrolite density contours for both groups
        training_data.loc[
            best_soft_prediction, [vars[coords[1]], vars[coords[0]]]
        ].pyroplot.density(
            ax=ax[coords[0], coords[1]],
            bins=100,
            contours=[0.67, 0.95],
            colors=["navy", "blue"],
            linestyles="--",
            linewidths=[1, 1.5],
            label_contours=False,
            zorder=10,
            extent=(
                ax[coords[0], coords[1]].get_xlim()[0],
                ax[coords[0], coords[1]].get_xlim()[1],
                ax[coords[0], coords[1]].get_ylim()[0],
                ax[coords[0], coords[1]].get_ylim()[1],
            ),
        )
        if second_best_prediction:
            training_data.loc[
                second_best_prediction, [vars[coords[1]], vars[coords[0]]]
            ].pyroplot.density(
                ax=ax[coords[0], coords[1]],
                bins=100,
                contours=[0.67, 0.95],
                colors=["darkgoldenrod", "darkorange"],
                linestyles="--",
                linewidths=[1, 1.5],
                label_contours=False,
                zorder=10,
                extent=(
                    ax[coords[0], coords[1]].get_xlim()[0],
                    ax[coords[0], coords[1]].get_xlim()[1],
                    ax[coords[0], coords[1]].get_ylim()[0],
                    ax[coords[0], coords[1]].get_ylim()[1],
                ),
            )

    # remove all the x and y labels that get auto populated
    # from pyrolite
    for coords in all_indices:
        ax[coords[0], coords[1]].set_xlabel("")
        ax[coords[0], coords[1]].set_ylabel("")

    # set the x and y labels on the left and bottom axes
    for xcoords, ycoords in zip(
        all_indices[all_indices[:, 0] == n - 1], all_indices[all_indices[:, 1] == 0]
    ):
        ax[xcoords[0], xcoords[1]].set_xlabel(vars[xcoords[1]])
        ax[ycoords[0], ycoords[1]].set_ylabel(vars[ycoords[0]])

    # remove the top left y label because its technically a diagonal plot
    ax[0, 0].set_ylabel("")

    # removing inner x and y tick labels so the figure reads cleaner
    for coords in all_indices:
        if coords[1] != 0:
            ax[coords[0], coords[1]].set_yticklabels("")
        if coords[0] != 4:
            ax[coords[0], coords[1]].set_xticklabels("")

    # custom legend
    legend_elements = [
        Patch(facecolor=train_color, edgecolor="none", label="All training data"),
        Patch(
            facecolor=train_pred_color,
            edgecolor="none",
            label=f"{best_soft_prediction} training data (Best)",
        ),
        Patch(
            facecolor=iodp_color,
            edgecolor="none",
            label=f"{g} ka\nprediction: {best_soft_prediction} ",
        ),
    ]
    if second_best_prediction:
        # custom legend
        legend_elements = [
            Patch(facecolor=train_color, edgecolor="none", label="All training data"),
            Patch(
                facecolor=train_pred_color,
                edgecolor="none",
                label=f"{best_soft_prediction} training data (Best)",
            ),
            Patch(
                facecolor=train_second_color,
                edgecolor="none",
                label=f"{second_best_prediction} training data (2nd Best)",
            ),
            Patch(
                facecolor=iodp_color,
                edgecolor="none",
                label=f"{g} ka\nprediction: {best_soft_prediction} ",
            ),
        ]

    # add the legend
    fig.legend(
        handles=legend_elements,
        bbox_to_anchor=[0.25, 0.9],
        title="Data type",
        title_fontsize=20,
        fontsize=12,
        frameon=True,
        shadow=True,
        facecolor="w",
    )
    # ax[0,2].text(0,0.2,f"{sample}\nAge: {soft_predictions.loc[sample,'Age (yrs)'].unique()[0]}",transform = ax[0,2].transAxes,fontsize = 20)
    # decrease the spacing between the plots

    ax[1, 3].boxplot(
        soft_predictions.loc[groups[g], sorted_locations],
        boxprops=dict(color="navy", facecolor="cornflowerblue", alpha=0.7),
        medianprops=dict(color="navy", lw=1),
        capprops=dict(linewidth=0),
        whiskerprops=dict(color="navy"),
        flierprops=dict(markeredgecolor="navy", alpha=0.5, markersize=3),
        showfliers=True,
        patch_artist=True,
    )
    fig.tight_layout()
    ax[1, 3].set_ylabel("Probability")
    ax[1, 3].set_xticklabels(sorted_locations, rotation=90)
    ax[1, 3].set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax[1, 3].set_yticklabels([0.2, 0.4, 0.6, 0.8, 1])
    ax[1, 3].set_ylim(0, 1)
    ax[1, 3].minorticks_off()
    ax[1, 3].set_title(
        f"n = {soft_predictions.loc[groups[g],:].shape[0]}",
        loc="left",
        fontsize=12,
    )
    mpl_defaults.left_bottom_axes(ax[1, 3])

    fig.set_facecolor("w")
    plt.savefig(
        "{}\\{}_prediction_pairplot.pdf".format(export_path, g), bbox_inches="tight"
    )
    if g == "570-2":
        plt.show(block=True)
    else:
        plt.show(block=False)
