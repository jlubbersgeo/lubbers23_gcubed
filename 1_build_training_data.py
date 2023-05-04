"""# Building the training dataset

_Jordan Lubbers<br>
U.S. Geological Survey Alaska Volcano Observatory_<br>

This takes the `major_trace_proximal_train_data` sheet from `Lubbers23_gcubed_supplementary_data.xlsx` 
spreadsheet and builds the training dataset for subsequent use in machine learning applications. 
To do this, we will largely be using functions from the `kinumaax.crunching` module. This does **not** 
do any feature engineering, but provides the foundation for which that may be explored by providing 
separate tidy and transformed spreadsheets of the data. The general workflow is:

1. remove strings (e.g., b.d.l. in laser data) and replace with `NaN`
2. Remove observations with `NaN`
3. Add in uncertainties for EPMA. For each LAICPMS analysis on a tephra chip we are using the tephra chip average and standard deviation major element composition. Where only one EPMA analysis exists per chip we use the relationship between concentration of an analyte and its external precision (reproducibility) to come up with a value for its uncertainty. See below and in `EPMA_stds_check.ipynb` for more details.
4. Convert wt% oxide concentrations to ppm elemental concentrations
5. Remove negative concentrations (required for log-transforms)
6. Transform the data using the centered logratio transform
7. Add in ratios
8. Export cleaned untransformed data as well as cleaned transformed data
"""


import time
from warnings import simplefilter

import numpy as np
import pandas as pd
import pyrolite.comp
from rich.console import Console
from rich.prompt import Prompt
from rich.theme import Theme

from kinumaax.source.kinumaax import crunching as kc

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)


custom_theme = Theme(
    {"main": "bold gold1", "path": "bold steel_blue1", "result": "magenta"}
)
console = Console(theme=custom_theme)

export_path = Prompt.ask("[bold gold1] Enter the path to where spreadsheets should be exported[bold gold1]")
export_path = export_path.replace('"',"")


spreadsheet_path = Prompt.ask("[bold gold1]Enter the path the the supplementary data file")
spreadsheet_path = spreadsheet_path.replace('"',"")

t0 = time.time()
console.print("\nWORKING ON PROXIMAL DATA\n",style = "main")

data = pd.read_excel(spreadsheet_path, sheet_name="major_trace_proximal_train_data")
shape_original = data.shape

# gets a list of trace elements measured
# remove Cu, V, and P because there's a lot of NaNs and
# below detection limit values
trace_elements = [
    element
    for element in data.loc[:, "7Li":"238U"].columns.tolist()
    if not ("V" in element or "Cu" in element or "31P" in element)
]

# list of major elements measured
major_elements = data.loc[:, "SiO2":"P2O5"].columns.tolist()

# convert strings in numeric columns to NaN so they can be removed
if data.index.name == "volcano":
    ignore_cols = [
        "Sample_shard",
        "timestamp",
        "experiment",
        "eruption",
        "AT_num",
        "Spot",
        "lat",
        "lon",
    ]
else:
    ignore_cols = [
        "Sample_shard",
        "volcano",
        "timestamp",
        "experiment",
        "eruption",
        "AT_num",
        "Spot",
        "lat",
        "lon",
    ]

data = kc.remove_strings(data, ignore_cols=ignore_cols)


# check for NaN values in trace element columns
nan_elements, nan_counts = kc.df_checkna(df=data, cols=trace_elements)
console.print("\nThese are the columns that have nans and their values\n",style = "result")
for element, count in zip(nan_elements, nan_counts):
    console.print(f"{element} | {count}",style = "result")

if data.index.name != "volcano":
    data = data.set_index("volcano")

volcanoes = data.index.unique().tolist()
console.print(f"\nYour volcanoes are: {volcanoes}\n",style = "result")

console.print(
    f"\nThe shape of your data with NaNs is {data.shape[0]} rows by {data.shape[1]} columns\n",style = "result")

# drop the rows where there are NaN values in trace element columns
data = kc.df_dropna(df=data, cols=nan_elements)
console.print(
    f"\nThe shape of your data is {data.shape[0]} rows by {data.shape[1]} columns after NaNs are removed\n",style = "result"
)


for col, major in zip([f"{element}_std" for element in major_elements], major_elements):
    # take care of the 0 uncertainties b/c can't take log of 0
    data[col] = np.where(
        data[col] == 0,
        (data[major] / 100) * np.exp(-0.64 * np.log(data[major]) + 1.57),
        data[col],
    )
    # take care of the NaNs
    data[col] = data[col].fillna(
        (data[major] / 100) * np.exp(-0.64 * np.log(data[major]) + 1.57)
    )


# do mean values
data = kc.df_oxide_to_ppm(data, oxides=major_elements)
major_elements_ppm = [column for column in data.columns if "ppm" in column]

# do uncertainty values
temp_df = data.loc[:, [f"{element}_std" for element in major_elements]]
temp_df.columns = major_elements
temp_df = kc.df_oxide_to_ppm(temp_df, oxides=major_elements)
temp_df.columns = [f"{col}_std" for col in temp_df.columns]

# add the uncertainty values from the temp dataframe back in
# to data
data = pd.concat(
    [
        data,
        temp_df.loc[:, [col for col in temp_df.columns if "ppm" in col]],
    ],
    axis="columns",
)

console.print(
    f"Congrats! You've added the following columns: \n {major_elements_ppm } and their uncertainties to the dataset",style = "result"
)


data = kc.filter_by_value(data, col_name="TiO2_ppm", value=0, operator="greater_than")
data = kc.filter_by_value(data, col_name="MnO_ppm", value=0, operator="greater_than")
data = kc.filter_by_value(data, col_name="P2O5_ppm", value=0, operator="greater_than")
console.print(f"The shape of your data is {data.shape[0]} rows by {data.shape[1]} columns",style = "result")


to_transform = data.loc[:, major_elements_ppm + trace_elements]
to_transform["O_ppm"] = 1e6 - to_transform.sum(axis="columns")

clr_df = to_transform.pyrocomp.CLR()
clr_df.columns = major_elements_ppm + trace_elements + ["O_ppm"]

back_transformed = clr_df.pyrocomp.inverse_CLR()
if np.allclose(back_transformed * 1e6, to_transform) is True:
    console.print("\nCongrats you transformed your data correctly!\n",style = "result")
else:
    console.print(
        "\nYour transformed data nad back transformed data are not the same. Try again\n",style = "result"
    )

rel_uncertainties = pd.DataFrame(
    np.array(
        data.loc[
            :,
            [f"{major}_std" for major in major_elements_ppm]
            + [f"{trace}_std" for trace in trace_elements],
        ]
    )
    / np.array(to_transform.loc[:, major_elements_ppm + trace_elements]),
    columns=[f"{element}_rel_std" for element in major_elements_ppm + trace_elements],
    index=clr_df.index,
)

rel_uncertainties["O_ppm_rel_std"] = 0.02

abs_uncertainties = pd.DataFrame(
    abs(np.array(clr_df) * np.array(rel_uncertainties)),
    columns=[f"{element}_std" for element in major_elements_ppm + trace_elements]
    + ["O_ppm_std"],
    index=clr_df.index,
)

clr_df = pd.concat([clr_df, rel_uncertainties, abs_uncertainties], axis="columns")

# numerators
nums = [
    "88Sr",
    "90Zr",
    "90Zr",
    "90Zr",
    "93Nb",
    "93Nb",
    "85Rb",
    "137Ba",
    "137Ba",
    "137Ba",
    "139La",
    "85Rb",
    "139La",
    "232Th",
    "232Th",
    "140Ce",
    "85Rb",
]
# denominators
dens = [
    "89Y",
    "89Y",
    "93Nb",
    "178Hf",
    "238U",
    "89Y",
    "147Sm",
    "232Th",
    "139La",
    "93Nb",
    "93Nb",
    "137Ba",
    "172Yb",
    "172Yb",
    "139La",
    "208Pb",
    "133Cs",
]
# corresponding uncertainty columns
num_stds = [f"{num}_std" for num in nums]
den_stds = [f"{den}_std" for den in dens]

# iterate through them all simultaneously calling the function
for num, den, num_std, den_std in zip(nums, dens, num_stds, den_stds):
    clr_df = kc.create_ratio_val(
        clr_df, num=num, den=den, num_std=num_std, den_std=den_std, uncertainty=True
    )
    data = kc.create_ratio_val(
        data, num=num, den=den, num_std=num_std, den_std=den_std, uncertainty=True
    )
ratios = [f"{num}/{den}" for num, den in zip(nums, dens)]
ratio_stds = [f"{num}/{den}_std" for num, den in zip(nums, dens)]

shape_final = clr_df.shape

percent_change = 100 * ((shape_original[0] - shape_final[0]) / shape_original[0])

console.print(
    f"\nData cleaning has resulted in a {np.round(percent_change,2)}% change in the number of observations\n",style = "result"
)


clr_df = clr_df.loc[
    :,
    major_elements_ppm
    + ["O_ppm"]
    + trace_elements
    + ratios
    + abs_uncertainties.columns.tolist()
    + ratio_stds
    + rel_uncertainties.columns.tolist(),
]
clr_df.columns = (
    [
        "".join(a.split("O")[0] for a in analyte if not a.isdigit())
        for analyte in major_elements_ppm
    ]
    + ["O_ppm"]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in trace_elements]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in ratios]
    + [
        "".join(a.split("O")[0] for a in analyte if not a.isdigit())
        for analyte in abs_uncertainties.columns.tolist()[:-1]
    ]
    + ["O_ppm_std"]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in ratio_stds]
    + [
        "".join(a.split("O")[0] for a in analyte if not a.isdigit())
        for analyte in rel_uncertainties.columns.tolist()[:-1]
    ]
    + ["O_ppm_rel_std"]
)

clr_df.insert(0, "latitude", data["lat"])
clr_df.insert(1, "longitude", data["lon"])
clr_df.insert(0, "eruption", data["eruption"])
clr_df.insert(3, "Sample_shard", data["Sample_shard"])

clr_df.to_excel("{}\B4_training_data_transformed_v2.xlsx".format(export_path))

clean_data = data.loc[
    :,
    major_elements_ppm
    + trace_elements
    + ratios
    + [f"{element}_std" for element in major_elements_ppm]
    + [f"{element}_std" for element in trace_elements]
    + ratio_stds,
]
clean_data.columns = (
    [
        "".join(a.split("O")[0] for a in analyte if not a.isdigit())
        for analyte in major_elements_ppm
    ]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in trace_elements]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in ratios]
    + [
        "".join(a.split("O")[0] for a in analyte if not a.isdigit())
        for analyte in abs_uncertainties.columns.tolist()[:-1]
    ]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in ratio_stds]
)
clean_data.insert(0, "latitude", data["lat"])
clean_data.insert(1, "longitude", data["lon"])
clean_data.insert(0, "eruption", data["eruption"])
clean_data.insert(3, "Sample_shard", data["Sample_shard"])
clean_data.to_excel("{}\B4_training_data_cleaned.xlsx".format(export_path))

####################################################################################
#######################IODP PROCESSING HERE#########################################
####################################################################################
console.print("\nWORKING ON IODP DATA\n",style = "main")
# import the data
iodp_data = pd.read_excel(spreadsheet_path, sheet_name="iodp_trace_data")

# remove the strings
iodp_data = kc.remove_strings(
    iodp_data,
    ignore_cols=[
        "timestamp",
        "experiment",
        "IODP_sample",
        "Spot",
    ],
)

# drop the rows with Nans
iodp_data = kc.df_dropna(df=iodp_data, cols=trace_elements)

# get the age of each layer
iodp_data["Age (yrs)"] = np.ceil(iodp_data["Age (yrs)"])

# iterate through them all simultaneously calling the function
for num, den, num_std, den_std in zip(nums, dens, num_stds, den_stds):
    iodp_data = kc.create_ratio_val(
        iodp_data, num=num, den=den, num_std=num_std, den_std=den_std, uncertainty=True
    )


# convert the oxide wt% to elemental concentrations in ppm
iodp_data = kc.df_oxide_to_ppm(iodp_data, oxides=major_elements)

# do uncertainty values
temp_df = iodp_data.loc[:, [f"{element}_std" for element in major_elements]]
temp_df.columns = major_elements
temp_df = kc.df_oxide_to_ppm(temp_df, oxides=major_elements)
temp_df.columns = [f"{col}_std" for col in temp_df.columns]

# add the uncertainty values from the temp dataframe back in
# to data
iodp_data = pd.concat(
    [
        iodp_data,
        temp_df.loc[:, [col for col in temp_df.columns if "ppm" in col]],
    ],
    axis="columns",
)


iodp_data.set_index("IODP_sample", inplace=True)

iodp_clean_df = iodp_data.loc[
    :,
    major_elements_ppm
    + trace_elements
    + ratios
    + [f"{element}_std" for element in major_elements_ppm]
    + [f"{element}_std" for element in trace_elements]
    + ratio_stds,
]


iodp_clean_df.columns = (
    [
        "".join(a.split("O")[0] for a in analyte if not a.isdigit())
        for analyte in major_elements_ppm
    ]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in trace_elements]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in ratios]
    + [
        "".join(a.split("O")[0] for a in analyte if not a.isdigit())
        for analyte in abs_uncertainties.columns.tolist()[:-1]
    ]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in ratio_stds]
)
iodp_clean_df.insert(0, "Age (yrs)", iodp_data["Age (yrs)"])

iodp_clean_df.to_excel("{}\IODP_test_data_cleaned.xlsx".format(export_path))

to_transform = iodp_data.loc[:, major_elements_ppm + trace_elements]
to_transform["O_ppm"] = 1e6 - to_transform.sum(axis="columns")

iodp_clr_df = to_transform.pyrocomp.CLR()
iodp_clr_df.columns = major_elements_ppm + trace_elements + ["O_ppm"]

back_transformed = iodp_clr_df.pyrocomp.inverse_CLR()
if np.allclose(back_transformed * 1e6, to_transform) is True:
    console.print("\nCongrats you transformed your data correctly!\n",style = "result")
else:
    console.print(
        "\nYour transformed data nad back transformed data are not the same. Try again\n",style = "result"
    )

rel_uncertainties = pd.DataFrame(
    np.array(
        iodp_data.loc[
            :,
            [f"{major}_std" for major in major_elements_ppm]
            + [f"{trace}_std" for trace in trace_elements],
        ]
    )
    / np.array(to_transform.loc[:, major_elements_ppm + trace_elements]),
    columns=[f"{element}_rel_std" for element in major_elements_ppm + trace_elements],
    index=iodp_clr_df.index,
)

rel_uncertainties["O_ppm_rel_std"] = 0.02

abs_uncertainties = pd.DataFrame(
    abs(np.array(iodp_clr_df) * np.array(rel_uncertainties)),
    columns=[f"{element}_std" for element in major_elements_ppm + trace_elements]
    + ["O_ppm_std"],
    index=iodp_clr_df.index,
)

iodp_clr_df = pd.concat(
    [iodp_clr_df, rel_uncertainties, abs_uncertainties], axis="columns"
)

for num, den, num_std, den_std in zip(nums, dens, num_stds, den_stds):
    iodp_clr_df = kc.create_ratio_val(
        iodp_clr_df,
        num=num,
        den=den,
        num_std=num_std,
        den_std=den_std,
        uncertainty=True,
    )

iodp_clr_df = iodp_clr_df.loc[
    :,
    major_elements_ppm
    + ["O_ppm"]
    + trace_elements
    + ratios
    + abs_uncertainties.columns.tolist()
    + ratio_stds
    + rel_uncertainties.columns.tolist(),
]
iodp_clr_df.columns = (
    [
        "".join(a.split("O")[0] for a in analyte if not a.isdigit())
        for analyte in major_elements_ppm
    ]
    + ["O_ppm"]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in trace_elements]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in ratios]
    + [
        "".join(a.split("O")[0] for a in analyte if not a.isdigit())
        for analyte in abs_uncertainties.columns.tolist()[:-1]
    ]
    + ["O_ppm_std"]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in ratio_stds]
    + [
        "".join(a.split("O")[0] for a in analyte if not a.isdigit())
        for analyte in rel_uncertainties.columns.tolist()[:-1]
    ]
    + ["O_ppm_rel_std"]
)
iodp_clr_df.insert(0, "Age (yrs)", iodp_data["Age (yrs)"])


iodp_clr_df.to_excel("{}\IODP_data_transformed.xlsx".format(export_path))

####################################################################################
#######################DERKACHEV PROCESSING HERE####################################
####################################################################################
console.print("\nWORKING ON DERKACHEV DATA\n",style = "main")

derk_data = pd.read_excel(spreadsheet_path, sheet_name="derkachev_data")
# remove the strings
derk_data = kc.remove_strings(
    derk_data,
    ignore_cols=[
        "Core ID",
        "Tephra interval (cm b.s.f.)",
        "Source volcano",
        "Eruption ID",
        "LAT / LONG",
        "Section location",
        "Date (yymmdd)",
    ],
)

derk_trace_elements = derk_data.loc[:, "7Li":"238U"].columns.tolist()
# drop the rows with Nans


# iterate through them all simultaneously calling the function
for num, den, num_std, den_std in zip(nums, dens, num_stds, den_stds):
    derk_data = kc.create_ratio_val(derk_data, num=num, den=den, uncertainty=False)


# convert the oxide wt% to elemental concentrations in ppm
derk_data = kc.df_oxide_to_ppm(derk_data, oxides=major_elements)


derk_data.set_index("Core ID", inplace=True)

derk_clean_df = derk_data.loc[:, major_elements_ppm + derk_trace_elements + ratios]


derk_clean_df.columns = (
    [
        "".join(a.split("O")[0] for a in analyte if not a.isdigit())
        for analyte in major_elements_ppm
    ]
    + [
        "".join(a for a in analyte if not a.isdigit())
        for analyte in derk_trace_elements
    ]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in ratios]
)
# iodp_clean_df.insert(0, "Age (yrs)", iodp_data["Age (yrs)"])

derk_clean_df.to_excel("{}\Derkachev_test_data_cleaned.xlsx".format(export_path))

to_transform = derk_data.loc[:, major_elements_ppm + derk_trace_elements]
to_transform["O_ppm"] = 1e6 - to_transform.sum(axis="columns")

derk_clr_df = to_transform.pyrocomp.CLR()
derk_clr_df.columns = major_elements_ppm + derk_trace_elements + ["O_ppm"]

back_transformed = derk_clr_df.pyrocomp.inverse_CLR()
if np.allclose(back_transformed * 1e6, to_transform) is True:
    console.print("\nCongrats you transformed your data correctly!\n",style = "result")
else:
    console.print(
        "\nYour transformed data nad back transformed data are not the same. Try again\n",style = "result"
    )

# )

for num, den, num_std, den_std in zip(nums, dens, num_stds, den_stds):
    derk_clr_df = kc.create_ratio_val(
        derk_clr_df,
        num=num,
        den=den,
        uncertainty=False,
    )


derk_clr_df.columns = (
    [
        "".join(a.split("O")[0] for a in analyte if not a.isdigit())
        for analyte in major_elements_ppm
    ]
    + ["O_ppm"]
    + [
        "".join(a for a in analyte if not a.isdigit())
        for analyte in derk_trace_elements
    ]
    + ["".join(a for a in analyte if not a.isdigit()) for analyte in ratios]
)

derk_clr_df = pd.concat(
    [
        derk_data.loc[
            :,
            [
                "Tephra interval (cm b.s.f.)",
                "Source volcano",
                "Eruption ID",
                "LAT / LONG",
                "Section location",
                "Date (yymmdd)",
            ],
        ],
        derk_clr_df,
    ],
    axis="columns",
)
derk_clr_df.to_excel("{}\Derkachev_test_data_transformed.xlsx".format(export_path))

t1 = time.time()
console.print(f"\nScript runtime is {np.round((t1 - t0)/60,2)} minutes\n",style = "main")
console.print(f"Spreadsheets output at: {export_path}",style = "path")
