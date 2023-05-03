""" 
CHECKING FOR NOVEL DATA WITHIN IODP CORE SAMPLES
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyrolite.plot import pyroplot, density
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter



import mpl_defaults


# where all the figures get dumped
export_path = r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs"




data = pd.read_excel(
    r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs\B4_training_data_transformed_v2.xlsx"
)
major_elements = data.loc[:, "Si_ppm":"P_ppm"].columns.tolist()
trace_elements = data.loc[:, "Ca":"U"].columns.tolist()
ratios = data.loc[:, "Sr/Y":"Rb/Cs"].columns.tolist()


iodp_data = pd.read_excel(r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs\IODP_data_transformed.xlsx").set_index('IODP_sample')
iodp_samples = iodp_data.index.unique().tolist()
iodp_data.insert(0,'drill_site', [sample[:5] for sample in iodp_data.index])

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
    "746": ["U1417B-11H-2W_147-149"],  # 746ka
    "749": ["U1417C-11H-6W_85-87", "U1417B-11H-4W_5-7"],  # 749ka
    "771": ["U1417D-13H-4W_141-143"],  # 771
}

group_list = list(groups.keys())

# populations of eruptions that have similar chemical composition
pops = [
    ["0.1", "507"],
    ["150", "200", "707"],
    ["570", "734"],
    ["27", "57"],
    ["18"],
    ["570-2"],
    ["129-2", "164"],
    ["280", "568", ],
    ["529","746", "749", "771"],
    ["591"],
]

##########################################################
###################### FIGURE S10 #######################
#########################################################
fig, ax = plt.subplots(5, 2, figsize=(6, 15), layout="constrained")
vars = ["Sr/Y", "K_ppm", "Nb/U", "Si_ppm", "Th/La", "Cs"]
xvar = "Rb/Sm"

volcanoes = ["Katmai", "Emmons Lake"]
pops = [["0.1"], ["27"]]

names = ["1912_pop", "27_pop"]

df = data.set_index("volcano")
dff = data.set_index("eruption")

train_color = "lightgray"
iodp_color = "darkslategray"

legend_elements = []
for volcano, pop, j in zip(volcanoes, pops, range(len(volcanoes))):
    eruptions = df.loc[volcano, "eruption"].unique()
    pop_samples = list(map(groups.get, pop))
    pop_samples = [item for sublist in pop_samples for item in sublist]
    colorblind_colors = mpl_defaults.create_colorblind_palette(n=len(eruptions))

    for var, i in zip(vars, range(ax.shape[0])):
        ax[i, j].plot(
            data[xvar],
            data[var],
            marker="o",
            ls="",
            ms=4,
            mfc=train_color,
            mew=0,
            alpha=0.7,
        )
        for eruption, color in zip(eruptions, colorblind_colors):
            ax[i, j].plot(
                dff.loc[eruption, xvar],
                dff.loc[eruption, var],
                marker="o",
                mfc="none",
                mec=color,
                ls="",
                ms=4,
                label = eruption
            )
            if i==0:
                ax[i,j].legend(loc = 'upper left',bbox_to_anchor = (0,1.4),title = f"{volcano} Eruptions")
                

        ax[i, j].plot(
            iodp_data.loc[pop_samples, xvar],
            iodp_data.loc[pop_samples, var],
            marker="o",
            ls="",
            ms=4,
            mec=iodp_color,
            mfc="none",
            mew=1,
        )
        if i == ax.shape[0] - 1:
            ax[i, j].set_xlabel(xvar)
        if j == 0:
            ax[i, j].set_ylabel(var)
plt.savefig("{}\IODP_train_compare_27ka-100yr_pops.pdf".format(export_path),bbox_inches = 'tight')

plt.show()

##########################################################
###################### FIGURE S9 #######################
#########################################################

vars = ["Nb/U", "Sr/Y", "Th/La", "La/Yb","Rb/Sm"]
n = len(vars)
train_color = "lightgray"
iodp_color = "darkslategray"

volcano = ["Augustine"]
df = data.set_index("volcano")
dff = data.set_index("eruption")
pops = [
    ["0.1", "507"],
    ["150", "200", "707"],
    ["570", "734"],
    ["27", "57"],
    ["18"],
    ["570-2"],
    ["129-2", "164"],
    ["280", "568", ],
    ["529","746", "749", "771"],
    ["591"],
]

colorblind_colors = mpl_defaults.create_colorblind_palette(n = 9)
colorblind_colors.append('black')
eruptions = df.loc[volcano, "eruption"].unique()


fig, ax = plt.subplots(n, n, figsize=(3 * n, 3 * n))

upper_indices = np.array(np.where(np.triu(ax, k=1))).T
lower_indices = np.array(np.where(np.tril(ax, k=-1))).T
diag_indices = np.array(np.where(np.eye(ax.shape[0], dtype=bool))).T
all_indices = np.vstack([upper_indices, lower_indices, diag_indices])

# plot the diagonals
for coords in diag_indices:
    # histograms where y vals are density
    hist, bins = np.histogram(data.loc[:, vars[coords[0]]], bins=20, density=True)

    mpl_defaults.bottom_only_axes(ax[coords[0], coords[1]])
    ax[coords[0], coords[1]].set_xlim(bins.min(), bins.max())

# remove the upper triangle axes
for coords in upper_indices:
    ax[coords[0], coords[1]].axis("off")

# plot the lower triangle axes
for coords in lower_indices:
    # scatter plots for both groups

    ax[coords[0], coords[1]].plot(
        data[vars[coords[1]]],
        data[vars[coords[0]]],
        marker="o",
        ls="",
        ms=4,
        mfc='none',
        mew=0,
        alpha=0.7,
    )

    
    ax[coords[0], coords[1]].plot(
        data[vars[coords[1]]],
        data[vars[coords[0]]],
        marker="o",
        ls="",
        ms=4,
        mfc = 'none',
        mec=train_color,
        mew=.5,
        alpha=0.5,
    )
    
    if all(coords == np.array([1, 0])):
        for i, samples, color in zip(range(len(pops)),pops, colorblind_colors):
            pop_samples = list(map(groups.get, samples))
            pop_samples = [item for sublist in pop_samples for item in sublist]

            ax[coords[0], coords[1]].plot(
                iodp_data.loc[pop_samples, vars[coords[1]]],
                iodp_data.loc[pop_samples, vars[coords[0]]],
                marker="^",
                ls="",
                ms=4,
                mec=color,
                mfc="none",
                mew=.75,
                label = f"pop. {i+1}: {samples}",
            )
    else:
        for i, samples, color in zip(range(len(pops)),pops, colorblind_colors):
            pop_samples = list(map(groups.get, samples))
            pop_samples = [item for sublist in pop_samples for item in sublist]

            ax[coords[0], coords[1]].plot(
                iodp_data.loc[pop_samples, vars[coords[1]]],
                iodp_data.loc[pop_samples, vars[coords[0]]],
                marker="^",
                ls="",
                ms=4,
                mec=color,
                mfc="none",
                mew=.75,
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

# making all the x limits for a given column the same
for i in range(n):
    col = all_indices[all_indices[:, 1] == i]
    for coords in col:
        ax[coords[0], coords[1]].set_xlim(ax[i, i].get_xlim())

# removing inner x and y tick labels so the figure reads cleaner
for coords in all_indices:
    if coords[1] != 0:
        ax[coords[0], coords[1]].set_yticklabels("")
    if coords[0] != 7:
        ax[coords[0], coords[1]].set_xticklabels("")


h, l = ax[1, 0].get_legend_handles_labels()


# custom legend
legend_elements = [
    Patch(facecolor=train_color, edgecolor="k", label=f"all training data"),
    # Patch(facecolor=iodp_color, edgecolor="black", label="\n".join(pop_samples)),
]
# add the legend
fig.legend(
    handles=legend_elements + h,
    bbox_to_anchor=[0.25, 0.78],
    title="Data type",
    title_fontsize=20,
    fontsize=12,
    frameon=True,
    shadow=True,
    facecolor="w",
    ncol=2,
    markerscale=2,
)
# decrease the spacing between the plots
fig.tight_layout()
fig.set_facecolor("w")

for coords in diag_indices:
    ax[coords[0],coords[1]].axis('off')

# save the thing

plt.savefig("{}\IODP_train_compare_pairplot_populations.pdf".format(export_path),bbox_inches = 'tight')
plt.show()



