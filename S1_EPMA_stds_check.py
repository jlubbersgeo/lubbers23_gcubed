from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

simplefilter(action='ignore', category=RuntimeWarning)
from rich.console import Console
from rich.progress import track
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme

import mpl_defaults

# where all the figures get dumped
custom_theme = Theme(
    {"main": "bold gold1", "path": "bold steel_blue1", "result": "magenta"}
)
console = Console(theme=custom_theme)

export_path = Prompt.ask("[bold gold1] Enter the path to where spreadsheets should be exported[bold gold1]")
export_path = export_path.replace('"',"")


inpath = Prompt.ask("[bold gold1]Enter the path the the supplementary data file")
inpath = inpath.replace('"',"")

data = pd.read_excel(inpath, sheet_name="EPMA_secondary_standards").set_index("standard")
stds_data = pd.read_excel(inpath, sheet_name="EPMA_accepted_standards").set_index("standard")

standards = data.index.unique().tolist()
standards.remove("RLS-75")

df = pd.DataFrame()
for standard in standards:

    measured = data.loc[standard, "SiO2_norm":"K2O_norm"]
    if standard == "VG-2_cal":
        accepted = stds_data.loc["VG-2", "SiO2_norm":"K2O_norm"]

    else:
        accepted = stds_data.loc[standard, "SiO2_norm":"K2O_norm"]

    ratios = measured.div(accepted)
    df = pd.concat([df, ratios])

df.reset_index(inplace=True)




###########################################################################
############################# FIGURE S1 ##################################
########################################################################
elements = df.loc[:, "SiO2_norm":].columns.tolist()
rows = int(np.ceil(len(elements) / 3))
labels = [
    "SiO$_2$",
    "TiO$_2$",
    "Al$_2$O$_3$",
    "FeO$_T$",
    "MnO",
    "MgO",
    "CaO",
    "Na$_2$O",
    "K$_2$O",
]


fig, ax = plt.subplots(3, rows, figsize=(12, 12))
axes = ax.ravel()
for i in range(len(elements)):
    sns.boxplot(
        data=df,
        y=elements[i],
        x="standard",
        ax=axes[i],
        color="whitesmoke",
        width=0.5,
        boxprops={"linewidth": 0.75, "edgecolor": "k"},
        medianprops={"linewidth": 1, "color": "k"},
        capprops={"linewidth": 0},
        whiskerprops={"linewidth": 1, "color": "k"},
        showfliers=False,
    )

    sns.stripplot(
        data=df,
        y=elements[i],
        x="standard",
        hue="standard",
        linewidth=0.5,
        edgecolor="k",
        ax=axes[i],
        legend=False,
    )
    axes[i].set_title("{}".format(labels[i]), fontsize=20, loc="left")
    mpl_defaults.left_bottom_axes(axes[i])
    axes[i].minorticks_off()
    axes[i].set_ylabel("")

    axes[i].set_xlabel("")


fig.tight_layout()
fig.set_facecolor("w")

fig.supylabel(
    "measured / accepted",
    fontsize=20,
)

fig.tight_layout()

plt.savefig(r"{}\Menlo_stds_nolowtotals.pdf".format(export_path))

df_bystandard = df.set_index("standard")
means = []
stds = []
for standard in standards:
    means.append(df_bystandard.loc[standard].mean() * 100)
    stds.append(df_bystandard.loc[standard].std() * 100)

mean_df = pd.DataFrame(np.round(np.array(means), 1))
mean_df.index = standards
mean_df.columns = elements
mean_df = mean_df.T
mean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
mean_df.fillna('no value',inplace = True)

stds_df = pd.DataFrame(np.round(np.array(stds), 1))
stds_df.index = standards
stds_df.columns = elements
stds_df = stds_df.T
stds_df.replace([np.inf, -np.inf], np.nan, inplace=True)
stds_df.fillna('',inplace = True)


vals = []
for j in range(mean_df.shape[0]):
    
    for i in range(mean_df.shape[1]):
        vals.append("{} ± {}".format(str(mean_df.iloc[j,i]),str(stds_df.iloc[j,i])))


vals_df = pd.DataFrame(np.array(vals).reshape(mean_df.shape))
vals_df.index =elements
vals_df.columns = standards
vals_df = vals_df
vals_df.index.name = 'Analyte'

vals_df.to_excel('{}\B4_proximal_epmasecondarystandards_accuracy.xlsx'.format(export_path))

#######################################################################
#################### FIGURE AND OUTPUT SHOWING ########################
###### RELATIONSHIP BETWEEN CONCENTRATION AND EXTERNAL PRECISION #####
good_standards = [standard for standard in standards if standard != "KN18"]
mean_df = pd.DataFrame()
std_df = pd.DataFrame()
elements_notnormal = [element.replace("_norm", "") for element in elements]
for standard in good_standards:
    means = pd.DataFrame(data.loc[standard, elements].mean()).T
    means.columns = elements
    means.index = [standard]
    mean_df = pd.concat([mean_df, means])

    stds = pd.DataFrame(data.loc[standard, elements].std()).T
    stds.columns = [f"{element}" for element in elements]
    stds.index = [standard]
    std_df = pd.concat([std_df, stds])
mean_df.index.name = "Standard"
std_df.index.name = "Standard"

mean_df_melted = mean_df.reset_index().melt(
    id_vars=["Standard"], value_vars=elements, var_name="element", value_name="mean"
)
std_df_melted = std_df.reset_index().melt(
    id_vars=["Standard"], value_vars=elements, var_name="element", value_name="std"
)

melted = (
    pd.concat([mean_df_melted, std_df_melted], axis="columns").T.drop_duplicates().T
)
melted["rel_std"] = 100 * (melted["std"] / melted["mean"])
melted["log_mean"] = np.log(melted["mean"].to_numpy().astype("float"))
melted["log_rel_std"] = np.log(melted["rel_std"].to_numpy().astype("float"))

X = melted["log_mean"].to_numpy()
X = sm.add_constant(X)

y = melted["log_rel_std"].to_numpy()

results = sm.OLS(y, X).fit()

xx = np.linspace(-5, 5, 100)
xx = sm.add_constant(xx)
df = results.get_prediction(xx).summary_frame(alpha=0.5)

print(results.summary())

fig, ax = plt.subplots(2,1, figsize = (4,8),layout = 'constrained')

ax[0].plot(xx[:, 1], df["mean"], c="k", ls="--", label = 'best fit')
ax[0].fill_between(
    xx[:, 1], df["mean_ci_lower"], df["mean_ci_upper"], fc="gray", alpha=0.5
)
ax[0].fill_between(xx[:, 1], df["obs_ci_lower"], df["obs_ci_upper"], fc="gray", alpha=0.3)
sns.scatterplot(data=melted, x="log_mean", y="log_rel_std", hue="element", ec="k", ax = ax[0])
h, l = ax[0].get_legend_handles_labels()
ax[0].legend(h,[label.replace("_norm","") for label in l ],loc = 'upper right', ncol = 2, bbox_to_anchor = (1,1),fontsize = 8)

ax[0].set_xlabel("log[concentration]")
ax[0].set_ylabel("log[% 1$\sigma$]")

ax[0].text(
    0.05,
    0.1,
    f"y = {np.round(results.params[1],2)}$\pm$ {np.round(results.bse[1],2)}x + {np.round(results.params[0],2)}$\pm${np.round(results.bse[0],2)}",
    transform=ax[0].transAxes,
)

x_var = np.linspace(.01,85,100)
y_var = np.exp(-0.64*np.log(x_var) + 1.57)
sns.scatterplot(data=melted, x="mean", y="rel_std", hue="element", ec="k",ax = ax[1], legend = False)
ax[1].plot(x_var,y_var,'k--')

ax[1].set_xlabel("concentration [wt%]")
ax[1].set_ylabel("% 1$\sigma$")
ax[1].set_yscale('log')


plt.savefig(r"{}\Menlo_stds_externalerrors.pdf".format(export_path))
plt.show(block = False)




console = Console()

output_table = vals_df.reset_index()
display_table = Table(title="EPMA secondary standard measured values")
for column in output_table.columns.tolist():

    display_table.add_column(column)

rows = output_table.values.tolist()
rows = [[str(el) for el in row] for row in rows]
for row in rows:
    display_table.add_row(*row)

console.print(display_table)
plt.show(block = True)
