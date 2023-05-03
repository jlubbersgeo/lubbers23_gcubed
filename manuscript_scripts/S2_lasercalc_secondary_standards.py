"""
LASERCALC FOR SECONDARY STANDARDS ONLY
This uses GEOREM accepted values for secondary standards to "auto process"
them after they have been normalized to an internal standard

For the math here see the following 
[document](https://github.com/jlubbersgeo/laserTRAM-DB/blob/main/docs/LaserTRAM_DB_documentation.pdf)
"""

import re

# matplotlib defaults
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
# from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.tools.eval_measures import rmse
import glob as glob
from tqdm import tqdm
import os

import mpl_defaults

from rich.console import Console
from rich.progress import track
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme

outpath = r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs"

#path to raw laser data csv files
folder_path = r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Data\laser_data\B4 Caldera Project\LT_complete\proximal"
infiles = glob.glob('{}/*.xlsx'.format(folder_path))

for calib_std in ["GSE-1G","GSD-1G"]:
    
    file_df_list = []
    # Choosing our calibration standard


    for file in tqdm(infiles):

        # import published standards data
        stds_data = pd.read_excel(
            r"C:\Users\jlubbers\OneDrive - DOI\Research\Coding\laserTRAM-DB\tests\laicpms_stds_tidy.xlsx"
        ).set_index("Standard")


        data = (
            pd.read_excel(file).dropna().set_index("Spot"))

        data.insert(loc=0, column="index", value=np.arange(1, len(data) + 1))

        myspots = data.index.unique().tolist()
        pubstandards = stds_data.index.unique().tolist()
        # list of columns that pertain to analytes
        myanalytes = [
            analyte
            for analyte in data.columns.tolist()
            if not (
                "_se" in analyte
                or "_SE" in analyte
                or "norm" in analyte
                or "index" in analyte
                or "start" in analyte
                or "stop" in analyte
                or "long" in analyte
                or "timestamp" in analyte
            )
        ]
        myanalytes.remove(data["norm"].unique()[0])
        # list of columns that pertain to analyte uncertainties
        myuncertainties = [analyte + "_se" for analyte in myanalytes]

        stds_column = [[std for std in pubstandards if std in spot] for spot in myspots]

        stds_column = [["unknown"] if not l else l for l in stds_column]

        stds_column = [std for sublist in stds_column for std in sublist]

        # all standards that have more than 1 analysis and can
        # therefore be used as a calibration standard
        potential_standards = [
            std for std in np.unique(stds_column) if stds_column.count(std) > 1
        ]
        potential_standards.remove("unknown")

        # all of the samples in your input sheet that are NOT potential standards
        all_standards = list(np.unique(stds_column))
        all_standards.remove("unknown")

        # This now denotes whether or not something is a standard
        # or an unknown
        data["sample"] = stds_column

        data.reset_index(inplace=True)
        data.set_index("sample", inplace=True)


        # create a separate dataframe for our calibration standard data
        calib_std_data = data.loc[calib_std, :]

        # Calibration standard information
        # mean
        calib_std_means = calib_std_data.loc[:, myanalytes + myuncertainties].mean()
        # std deviation
        calib_std_stds = calib_std_data.loc[:, myanalytes + myuncertainties].std()
        # relative standard error
        calib_std_ses = 100 * (
            (calib_std_stds / calib_std_means) / np.sqrt(calib_std_data.shape[0])
        )

        # Get a list of all of the elements supported in the published standard datasheet
        # Get a second list for the same elements but their corresponding uncertainty columns
        standard_elements = [
            analyte for analyte in stds_data.columns.tolist() if not ("_std" in analyte)
        ]
        standard_uncertainties = [analyte + "_std" for analyte in standard_elements]

        calib_std_rmses = []
        calib_std_slopes = []
        calib_std_intercepts = []
        drift_check = []

        # For our calibration standard, calculate the concentration ratio of each analyte to the element used as the internal standard
        std_conc_ratios = []
        myanalytes_nomass = []

        for j in range(len(myanalytes)):

            # Getting regression statistics on analyte normalized ratios through time
            # for the calibration standard. This is what we use to check to see if it needs
            # to be drift corrected
            if "timestamp" in calib_std_data.columns.tolist():
                # get an array in time units based on timestamp column. This is
                # is in seconds
                x = np.array(
                    [np.datetime64(d, "m") for d in calib_std_data["timestamp"]]
                ).astype(np.float64)
                # x = np.cumsum(np.diff(x))
                # x = np.insert(x, 0, 0).astype(np.float64)

            else:

                x = calib_std_data["index"]

            # x = calib_std_data["index"]
            y = calib_std_data[myanalytes[j]]

            X = sm.add_constant(x)
            # Note the difference in argument order
            model = sm.OLS(y, X).fit()
            # now generate predictions
            ypred = model.predict(X)

            # calc rmse
            RMSE = rmse(y, ypred)

            calib_std_rmses.append(RMSE)

            if model.params.shape[0] < 2:
                calib_std_slopes.append(model.params[0])
                calib_std_intercepts.append(0)

            else:

                calib_std_slopes.append(model.params[1])
                calib_std_intercepts.append(model.params[0])

        # For our calibration standard, calculate the concentration ratio of each analyte to the element used as the internal standard
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
                    stds_data.loc[calib_std, nomass]
                    / stds_data.loc[
                        calib_std, re.split("(\d+)", calib_std_data["norm"].unique()[0])[2]
                    ]
                )




        # inputting an array rather than scalar value
        # int_std_oxide = int_std_concentrations.loc[:,'SiO2 wt%'].to_numpy()
        # int_std_concentration = oxide_to_ppm(int_std_oxide, calib_std_data["norm"].unique()[0])

        secondary_standards = all_standards.copy()
        secondary_standards.remove(calib_std)
        concentrations_list = []


        for standard in secondary_standards:
            drift_concentrations_list = []

            for j, analyte, slope, intercept, drift in zip(
                range(len(myanalytes)),
                myanalytes,
                calib_std_slopes,
                calib_std_intercepts,
                drift_check,
            ):

                if "True" in drift:
                    if "timestamp" in data.columns.tolist():
                        frac = (
                            slope
                            * np.array(
                                [
                                    np.datetime64(d, "m")
                                    for d in data.loc[standard, "timestamp"]
                                ]
                            ).astype(np.float64)
                            + intercept
                        )
                    else:

                        frac = slope * data.loc[standard, "index"] + intercept

                    drift_concentrations = (
                        (
                            stds_data.loc[
                                standard,
                                re.split("(\d+)", calib_std_data["norm"].unique()[0])[2],
                            ]
                        )
                        * (std_conc_ratios[j] / frac)
                        * data.loc[standard, analyte]
                    )

                    if type(drift_concentrations) == np.float64:
                        df = pd.DataFrame(np.array([drift_concentrations]), columns=[analyte])

                    else:
                        df = pd.DataFrame(drift_concentrations, columns=[analyte])

                    drift_concentrations_list.append(df)

            if len(drift_concentrations_list) > 0:

                drift_df = pd.concat(drift_concentrations_list, axis="columns")

                if drift_df.shape[0] == 1:
                    drift_df["sample"] = standard
                    drift_df.set_index("sample", inplace=True)

                concentrations = (
                    (
                        stds_data.loc[
                            standard, re.split("(\d+)", calib_std_data["norm"].unique()[0])[2]
                        ]
                    )
                    * (std_conc_ratios / calib_std_means[myanalytes])
                    * data.loc[standard, myanalytes]
                )

                for column in drift_df.columns.tolist():
                    if type(concentrations) == pd.Series:
                        concentrations.loc[column] = drift_df[column].to_numpy()[0]

                    else:
                        concentrations[column] = drift_df[column]

                if type(concentrations) == pd.Series:
                    concentrations = pd.DataFrame(concentrations).T
                    concentrations["sample"] = standard
                    concentrations.set_index("sample", inplace=True)

                concentrations_list.append(concentrations)
            else:
                concentrations = (
                    (
                        stds_data.loc[
                            standard, re.split("(\d+)", calib_std_data["norm"].unique()[0])[2]
                        ]
                    )
                    * (std_conc_ratios / calib_std_means[myanalytes])
                    * data.loc[standard, myanalytes]
                )
                concentrations_list.append(concentrations)


        # incorporate uncertainty in calibration standard
        calib_uncertainty = True


        stds_list = []
        unknowns_list = []
        # relative uncertainty in % of the concentration of the internal standard
        # in the unknown
        unknown_int_std_unc = 1

        # use RMSE of regression for elements where drift correction is applied rather than the standard error
        # of the mean of all the calibration standard normalized ratios
        for j in range(len(drift_check)):
            if "True" in drift_check[j]:
                calib_std_means[j] = 100 * calib_std_rmses[j] / calib_std_means[j]

        # creates a list of dataframes that hold the uncertainty information for each secondary standard.
        for standard, concentration in zip(secondary_standards, concentrations_list):

            # concentration of internal standard in unknown uncertainties
            t1 = (
                stds_data.loc[
                    standard,
                    "{}_std".format(re.split("(\d+)", calib_std_data["norm"].unique()[0])[2]),
                ]
                / stds_data.loc[
                    standard,
                    "{}".format(re.split("(\d+)", calib_std_data["norm"].unique()[0])[2]),
                ]
            ) ** 2

            # concentration of internal standard in calibration standard uncertainties
            t2 = (
                stds_data.loc[
                    calib_std,
                    "{}_std".format(re.split("(\d+)", calib_std_data["norm"].unique()[0])[2]),
                ]
                / stds_data.loc[
                    calib_std,
                    "{}".format(re.split("(\d+)", calib_std_data["norm"].unique()[0])[2]),
                ]
            ) ** 2

            # concentration of each analyte in calibration standard uncertainties
            std_conc_stds = []
            for i in range(len(myanalytes)):
                # strip the atomic number from our analyte data
                nomass = re.split("(\d+)", myanalytes[i])[2]

                # if our element is in the list of standard elements take the ratio
                if nomass in standard_elements:
                    std_conc_stds.append(
                        (
                            stds_data.loc[calib_std, "{}_std".format(nomass)]
                            / stds_data.loc[calib_std, nomass]
                        )
                        ** 2
                    )

            std_conc_stds = np.array(std_conc_stds)

            # Overall uncertainties

            if calib_uncertainty == True:

                stds_values = concentration * np.sqrt(
                    np.array(
                        t1
                        + t2
                        + std_conc_stds
                        + (calib_std_ses[myanalytes].to_numpy()[np.newaxis, :] / 100) ** 2
                        + (data.loc[standard, myuncertainties].to_numpy() / 100) ** 2
                    ).astype(np.float64)
                )

                stds_values.columns = myuncertainties

                stds_list.append(stds_values)
            else:
                stds_values = concentration * np.sqrt(
                    t1
                    + t2
                    + std_conc_stds
                    + (calib_std_ses[myanalytes].to_numpy()[np.newaxis, :] / 100) ** 2
                    + (data.loc[standard, myuncertainties].to_numpy() / 100) ** 2
                )
                stds_values.columns = myuncertainties
                stds_list.append(stds_values)


        final_standards_list = []
        final_unknowns_list = []
        # concatenates the concentrations and uncertainties dataframes such that there is now one dataframe for each
        # secondary standard that contains both the concentrations and concentrations of the uncertainties for each spot

        for concentration, standard, name in zip(
            concentrations_list, stds_list, secondary_standards
        ):
            df = pd.concat([concentration, standard], axis=1)
            df[df < 0] = "b.d.l."
            df.insert(loc=0, column="Spot", value=data.loc[name, "Spot"])
            if calib_std_data["norm"].unique()[0] == "43Ca":
                df.insert(loc=1, column="CaO", value=stds_data.loc[name, "CaO"] / 1e4)
                df.insert(
                    loc=2,
                    column="CaO_std%",
                    value=(stds_data.loc[name, "CaO_std"] / stds_data.loc[name, "CaO"]) * 100,
                )

            elif calib_std_data["norm"].unique()[0] == "29Si":
                df.insert(loc=1, column="SiO2", value=stds_data.loc[name, "SiO2"] / 1e4)
                df.insert(
                    loc=2,
                    column="SiO2_std%",
                    value=(stds_data.loc[name, "SiO2_std"] / stds_data.loc[name, "SiO2"]) * 100,
                )

            df.insert(loc = 1, column = "Day_run", value = np.array(data.loc[name,'timestamp'].dt.day_name()))
            df.insert(loc = 2, column = "timestamp", value = np.array(data.loc[name,'timestamp']))
            df.insert(loc = 3, column = 'file_name', value = os.path.basename(file))
            df.insert(loc = 4, column = "primary_standard", value = calib_std)


            final_standards_list.append(df)

        df_standards = pd.concat(final_standards_list)

        file_df_list.append(df_standards)

    all_standards_comps = pd.concat(file_df_list,axis = 'rows').reset_index().sort_values(by = 'timestamp')


    with pd.ExcelWriter(r"{}\AK_tephra_laser_std_comps_{}.xlsx".format(outpath,calib_std)) as writer:
        all_standards_comps.to_excel(writer, sheet_name="Sheet1",index = False)

    print(f"Congrats your standards are processed and saved here:\n {outpath}")

gse_data = pd.read_excel(f"{outpath}\AK_tephra_laser_std_comps_GSE-1G.xlsx").set_index('sample')
gsd_data = pd.read_excel(f"{outpath}\AK_tephra_laser_std_comps_GSD-1G.xlsx").set_index('sample')

##############################################################
################### MANUSCRIPT FIGURE S2 #####################
##############################################################
ratios_df = pd.DataFrame(0, index=gse_data.index, columns=myanalytes)
elements = gse_data.loc[:, "7Li":].columns.tolist()
elements = [element for element in elements if "_se" not in element]
secondary_standards = [
    'GSD-1G',
    'BCR-2G', 
    'ATHO-G', 
    'NIST-612'
]
calib_std = "GSE-1G"
for standard in secondary_standards:

    for analyte, nomass in zip(myanalytes, myanalytes_nomass):
        if type(gse_data.loc[standard, :]) == pd.Series:
            r = gse_data.loc[standard, analyte] / stds_data.loc[standard, nomass]
            ratios_df.loc[standard, analyte] = r
        else:

            r = (
                gse_data.loc[standard, analyte].to_numpy()
                / stds_data.loc[standard, nomass]
            )
            ratios_df.loc[standard, analyte] = r



ratios_df = ratios_df.loc[secondary_standards, :]
ratios_df_long = ratios_df.reset_index().melt(id_vars="sample", value_vars=myanalytes)

n = len(secondary_standards)
fig, ax = plt.subplots(n,1,figsize=(10, 3*n), layout = "constrained")
axes = ax.ravel()
for standard, a in zip(secondary_standards,axes):

    sns.boxplot(
        data=ratios_df_long[ratios_df_long['sample'] == standard],
        x="variable",
        y="value",
        boxprops = {'facecolor':'whitesmoke','lw':.5},
        medianprops = {"lw":1},
        capprops = {'lw':0},
        flierprops = {'marker': 'o', 'ms':3,'lw':.25,'mfc': 'whitesmoke','mec':'gray','mew':.25},
        whiskerprops = {"lw": 1},
        ax = a

    )

    a.axhline(1, c="k", ls="--", marker="", zorder=0, lw=1)
    a.axhspan(0.95, 1.05, alpha=0.2, fc="gray", zorder=0)
    # ax.set_ylim(0.5, 1.5)
    a.set_title(
        f"{standard}",
        fontsize=24,
        loc = 'left'
    )

    a.minorticks_off()

    a.set_xticklabels(myanalytes, rotation=70)
    a.set_ylabel("measured/accepted", fontsize=20)
    a.set_xlabel("")
    labels = [
        "$^{{{}}}${}".format(
            re.findall(r"\d+", element)[0],
            element.replace(re.findall(r"\d+", element)[0], ""),
        )
        for element in elements
    ]
    a.set_xticklabels(labels)


    a.minorticks_off()
# plt.legend([],[], frameon=False)
# fig.tight_layout()
plt.savefig("{}\{}_accuracy_{}primary.pdf".format(outpath,secondary_standards[0],calib_std),bbox_inches = 'tight')

################################################################
########## accuracy and precision values table #################
################################################################

ratios_df = pd.DataFrame(0, index=gse_data.index, columns=myanalytes)

secondary_standards = [
    'GSD-1G', 
    'BCR-2G', 
    'ATHO-G', 
    'NIST-612'
]
calib_std = "GSE-1G"
for standard in secondary_standards:

    for analyte, nomass in zip(myanalytes, myanalytes_nomass):
        if type(gse_data.loc[standard, :]) == pd.Series:
            r = gse_data.loc[standard, analyte] / stds_data.loc[standard, nomass]
            ratios_df.loc[standard, analyte] = r
        else:

            r = (
                gse_data.loc[standard, analyte].to_numpy()
                / stds_data.loc[standard, nomass]
            )
            ratios_df.loc[standard, analyte] = r



ratios_df = ratios_df.loc[secondary_standards, :]
ratios_df_long = ratios_df.reset_index().melt(id_vars="sample", value_vars=myanalytes)

means = []
stds = []
for standard in secondary_standards:
    means.append(ratios_df.loc[standard].mean())
    stds.append(ratios_df.loc[standard].std())

mean_df = pd.DataFrame(np.round(np.array(means),2))
mean_df.index = secondary_standards
mean_df.columns = myanalytes


stds_df = pd.DataFrame(np.round(np.array(stds),2))
stds_df.index = secondary_standards
stds_df.columns = myanalytes

vals = []
for j in range(mean_df.shape[0]):
    
    for i in range(mean_df.shape[1]):
        vals.append("{} ± {}".format(str(mean_df.iloc[j,i]),str(stds_df.iloc[j,i])))


vals_df = pd.DataFrame(np.array(vals).reshape(mean_df.shape))
vals_df.index = secondary_standards
vals_df.columns = myanalytes
vals_df = vals_df.T
vals_df.to_excel('{}\B4_proximal_lasersecondarystds_accuracy.xlsx'.format(outpath))

vals_df.index.name = 'Analyte'
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