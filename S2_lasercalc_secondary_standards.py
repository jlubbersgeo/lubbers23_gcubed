"""
This creates plots and tables to show the composition of
our measured LAICPMS secondary standards compared to their 
published values



"""

import glob as glob
import os
import re

# matplotlib defaults
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme

import mpl_defaults

custom_theme = Theme(
    {"main": "bold gold1", "path": "bold steel_blue1", "result": "magenta"}
)
console = Console(theme=custom_theme)

export_path = Prompt.ask(
    "[bold gold1] Enter the path to where figures should be exported[bold gold1]"
)
export_path = export_path.replace('"', "")

# path to raw laser data csv files
data_path = Prompt.ask(
    "[bold gold1] Enter the path to the supplementary data spreadsheet[bold gold1]"
)
data_path = data_path.replace('"', "")

data = pd.read_excel(data_path, sheet_name="LAICPMS_secondary_standards").set_index(
    "sample"
)


stds_data = pd.read_excel(data_path, sheet_name="LAICPMS_accepted_standards").set_index(
    "Standard"
)


##############################################################
################### MANUSCRIPT FIGURE S2 #####################
##############################################################
myanalytes = data.loc[:, "7Li":].columns.tolist()
myanalytes = [element for element in myanalytes if "_se" not in element]
# This has a lot of BDL values because of a high background on GSE glass
# and so we generally just threw this analyte out of the dataset...so
# not processed further.
myanalytes.remove("31P")


ratios_df = pd.DataFrame(0, index=data.index, columns=myanalytes)

secondary_standards = ["GSD-1G", "BCR-2G", "ATHO-G", "NIST-612"]
calib_std = "GSE-1G"
# calib_std_data = data.loc[calib_std, :]
standard_elements = [
    analyte for analyte in stds_data.columns.tolist() if not ("_std" in analyte)
]
std_conc_ratios = []
myanalytes_nomass = []

for i in range(len(myanalytes)):
    # strip the atomic number from our analyte data
    nomass = re.split("(\d+)", myanalytes[i])[2]
    # make it a list
    myanalytes_nomass.append(nomass)

    # if our element is in the list of standard elements take the ratio
    if nomass in standard_elements:
        std_conc_ratios.append(
            stds_data.loc[calib_std, nomass] / stds_data.loc[calib_std, "Si"]
        )

ratios_df = pd.DataFrame(0, index=data.index, columns=myanalytes)

for standard in secondary_standards:
    for analyte, nomass in zip(myanalytes, myanalytes_nomass):
        if type(data.loc[standard, :]) == pd.Series:
            r = data.loc[standard, analyte] / stds_data.loc[standard, nomass]
            ratios_df.loc[standard, analyte] = r
        else:
            r = data.loc[standard, analyte].to_numpy() / stds_data.loc[standard, nomass]
            ratios_df.loc[standard, analyte] = r


ratios_df = ratios_df.loc[secondary_standards, :]
ratios_df_long = ratios_df.reset_index().melt(id_vars="sample", value_vars=myanalytes)


n = len(secondary_standards)
fig, ax = plt.subplots(n, 1, figsize=(10, 3 * n), layout="constrained")
axes = ax.ravel()
for standard, a in zip(secondary_standards, axes):
    sns.boxplot(
        data=ratios_df_long[ratios_df_long["sample"] == standard],
        x="variable",
        y="value",
        boxprops={"facecolor": "whitesmoke", "lw": 0.5},
        medianprops={"lw": 1},
        capprops={"lw": 0},
        flierprops={
            "marker": "o",
            "ms": 3,
            "lw": 0.25,
            "mfc": "whitesmoke",
            "mec": "gray",
            "mew": 0.25,
        },
        whiskerprops={"lw": 1},
        ax=a,
    )

    a.axhline(1, c="k", ls="--", marker="", zorder=0, lw=1)
    a.axhspan(0.95, 1.05, alpha=0.2, fc="gray", zorder=0)
    # ax.set_ylim(0.5, 1.5)
    a.set_title(f"{standard}", fontsize=24, loc="left")

    a.minorticks_off()

    a.set_xticklabels(myanalytes, rotation=70)
    a.set_ylabel("measured/accepted", fontsize=20)
    a.set_xlabel("")
    labels = [
        "$^{{{}}}${}".format(
            re.findall(r"\d+", element)[0],
            element.replace(re.findall(r"\d+", element)[0], ""),
        )
        for element in myanalytes
    ]
    a.set_xticklabels(labels)

    a.minorticks_off()
# plt.legend([],[], frameon=False)
# fig.tight_layout()
plt.savefig(
    "{}\{}_accuracy_{}primary.pdf".format(
        export_path, secondary_standards[0], calib_std
    ),
    bbox_inches="tight",
)

################################################################
########## accuracy and precision values table #################
################################################################


means = []
stds = []
for standard in secondary_standards:
    means.append(ratios_df.loc[standard].mean())
    stds.append(ratios_df.loc[standard].std())

mean_df = pd.DataFrame(np.round(np.array(means), 2))
mean_df.index = secondary_standards
mean_df.columns = myanalytes


stds_df = pd.DataFrame(np.round(np.array(stds), 2))
stds_df.index = secondary_standards
stds_df.columns = myanalytes

vals = []
for j in range(mean_df.shape[0]):
    for i in range(mean_df.shape[1]):
        vals.append("{} ± {}".format(str(mean_df.iloc[j, i]), str(stds_df.iloc[j, i])))


vals_df = pd.DataFrame(np.array(vals).reshape(mean_df.shape))
vals_df.index = secondary_standards
vals_df.columns = myanalytes
vals_df = vals_df.T
vals_df.to_excel("{}\B4_proximal_lasersecondarystds_accuracy.xlsx".format(export_path))

vals_df.index.name = "Analyte"
console = Console()

output_table = vals_df.reset_index()
display_table = Table(title="LA-ICP-MS secondary standard measured values")
for column in output_table.columns.tolist():
    display_table.add_column(column)

rows = output_table.values.tolist()
rows = [[str(el) for el in row] for row in rows]
for row in rows:
    display_table.add_row(*row)

console.print(display_table)
plt.show()
