"""
DERKACHEV ET AL 2018 SAMPLE PREDICTIONS
"""
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

export_path = Prompt.ask("[bold gold1] Enter the path to where overall results should be exported[bold gold1]")
export_path = export_path.replace('"',"")

data_path = Prompt.ask("[bold gold1] Enter the folder path to where transformed data are stored[bold gold1]")
data_path = data_path.replace('"',"") 



vcs_load = pickle.load(open(f"{data_path}\Aleutian_tuned_vcs_classifier_trace_deployment.sav",'rb'))

derk_data = pd.read_excel(f"{data_path}\Derkachev_test_data_transformed.xlsx").set_index('Eruption ID')
derk_samples = derk_data.index.unique().tolist()

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

soft_predictions = pd.DataFrame(vcs_load.predict_proba(derk_data.loc[:,myfeatures]), columns = vcs_load.classes_)
soft_predictions.insert(0,'predicted_class',vcs_load.predict(derk_data.loc[:,myfeatures]))
soft_predictions.index = derk_data.index
samples = soft_predictions.index.unique()
soft_predictions.insert(0,'type','individual')

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
    "DV"
]

training_data = pd.read_excel(
    f"{data_path}\B4_training_data_transformed_v2.xlsx").set_index('volcano')
volcanoes = training_data.index.unique().tolist()

mean_df = pd.DataFrame()
std_df = pd.DataFrame()
for sample in samples:
    m = pd.DataFrame(derk_data.loc[sample,'Date (yymmdd)':].mean(axis = 'rows'))
    mean_df = pd.concat([mean_df,m.T])
    s = pd.DataFrame(derk_data.loc[sample,'Date (yymmdd)':].std(axis = 'rows'))
    std_df = pd.concat([std_df,s.T])
    
mean_df.index = samples
mean_df.index.name = 'Eruption ID'
std_df.index = samples
std_df.index.name = 'Eruption ID'


mean_predictions = pd.DataFrame(vcs_load.predict_proba(mean_df.loc[:,myfeatures]), columns = vcs_load.classes_)
mean_predictions.insert(0,'predicted_class',vcs_load.predict(mean_df.loc[:,myfeatures]))
mean_predictions.index = mean_df.index

# generate the 1000 random comps for each sample
random_df = pd.DataFrame()
np.random.seed(0)
for i in range(1000):

    random_df = pd.concat(
        [
            random_df,
            pd.DataFrame(
                np.random.normal(loc=mean_df, scale=std_df),
                index=mean_df.index,
                columns=mean_df.columns,
            ),
        ]
    )

# make model predictions for synthetic dataset
random_predictions = pd.DataFrame(
    vcs_load.predict_proba(random_df.loc[:, myfeatures]), columns=vcs_load.classes_
)
random_predictions.insert(
    0, "predicted_class", vcs_load.predict(random_df.loc[:, myfeatures])
)
random_predictions.index = random_df.index
random_predictions.insert(0, "type", "monte carlo")


# reshape to be used in seaborn
combined_df = pd.concat([soft_predictions, random_predictions])
combined_df_melted = (
    combined_df.reset_index()
    .melt(id_vars=["Eruption ID", "predicted_class", "type"], value_vars=volcanoes)
    .set_index("Eruption ID")
)
combined_df_melted.sort_values(by="variable", inplace=True)

##########################################################
################# MANUSCRIPT FIGURE 13 ###################
##########################################################
fig, ax = plt.subplots(2,1, figsize = (6,8),layout = 'constrained')
axes = ax.ravel()
for a,sample in zip(axes,samples):
    sns.boxplot(
        data=combined_df_melted.loc[sample, :],
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
    a.legend([],frameon = False)

    xticklabels = a.get_xticklabels()

    a.set_xticklabels(xticklabels, rotation=90)
    a.set_ylabel("Probability")
    a.set_xlabel("")
    a.minorticks_off()

legend_elements = [Patch(facecolor='C1', edgecolor='k',
                        label='Monte Carlo'),
                Patch(facecolor='C0', edgecolor='k',
                        label='Individual')]
fig.legend(handles = legend_elements,bbox_to_anchor = (0.52,1.07),ncol = 2,frameon = True, title = 'Prediction Type')
mpl_defaults.label_subplots(ax,location = 'upper right',fontsize = 16)
plt.savefig(
    "{}\derkachev_prediction_boxplot_panel.pdf".format(export_path), bbox_inches="tight"
)
plt.show(block = False)

##########################################################
################### FIGURE 14 ############################
##########################################################
vars = [
    "Nb/U",
    "Th/La",
    "Ba/Nb",
    "K_ppm"
]

n = len(vars)
colorblind_colors = mpl_defaults.create_colorblind_palette(n=3)
train_color = "lightgray"
train_pred_color = colorblind_colors[0]
train_second_color = colorblind_colors[2]
iodp_color = colorblind_colors[1]


for sample in tqdm(derk_samples):

    df1 = pd.DataFrame(
        data=soft_predictions.loc[sample, "predicted_class"].value_counts(),
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
            data=derk_data.loc[sample, :],
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
    # ax[1,3].axis("on")

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
            mec='none',
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
            mec='none',
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
            derk_data.loc[sample, vars[coords[1]]],
            derk_data.loc[sample, vars[coords[0]]],
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

        ax[coords[0], coords[1]].errorbar(
            mean_df.loc[sample, vars[coords[1]]],
            mean_df.loc[sample, vars[coords[0]]],
            xerr=std_df.loc[sample, vars[coords[1]]],
            yerr=std_df.loc[sample, vars[coords[0]]],
            marker="o",
            ls="",
            mfc=iodp_color,
            mec="darkmagenta",
            ecolor=iodp_color,
            elinewidth = 1.5,
            ms = 5,
            zorder = 10
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

    # # making all the x limits for a given column the same


    # removing inner x and y tick labels so the figure reads cleaner
    for coords in all_indices:
        if coords[1] != 0:
            ax[coords[0], coords[1]].set_yticklabels("")
        if coords[0] != 3:
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
            label=f"{sample}\nprediction: {best_soft_prediction} ",
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
                label=f"{sample}\nprediction: {best_soft_prediction}",
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

    fig.tight_layout()
    ax[1, 3].set_ylabel("Probability")
    ax[1, 3].set_xticklabels(sorted_locations, rotation=90)
    ax[1, 3].set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax[1, 3].set_yticklabels([0.2, 0.4, 0.6, 0.8, 1])
    ax[1, 3].set_ylim(0, 1)
    ax[1, 3].minorticks_off()
    # mpl_defaults.left_bottom_axes(ax[1,3])

    fig.set_facecolor("w")

    # save the thing
    plt.savefig(
        "{}\\{}_prediction_pairplot.pdf".format(export_path, sample),
        bbox_inches="tight",
    )
    if sample == derk_samples[0]:

        plt.show(block = False)
    else:
        plt.show(block = True)

