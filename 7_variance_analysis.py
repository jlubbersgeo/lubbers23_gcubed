""" 
# Model Variance Analysis

_Jordan Lubbers<br>
U.S. Geological Survey Alaska Volcano Observatory_<br>

This notebook explores our soft voting classifier's variance (i.e., the relationship 
between its performance and dependence on the data used to train it). Having already 
established that our model performs well on our test and validation sets when training
data are built from portions of all eruptions (i.e., has low bias), we explore our model 
variance by asking the following question:

“How likely would we be able to predict the correct volcanic source for an eruption if 
the model was not trained on that eruption?” 

For example, could we predict that a tephra sample from the Katmai 1912 eruption was 
from Katmai if the model is only trained on the Lethe and 23ka eruptions? Or more broadly, 
could we predict the correct volcanic source for an unknown tephra if its composition is 
slightly different than that of the training data from the same source? As volcanoes rarely 
erupt compositionally identical magmas throughout their entire history, this, we argue gives 
us a better idea of 1 how reliant we are on certain samples within our training dataset and 
2) the actual probability of an accurate prediction when applying our model to tephras where 
the source is either unknown or ambiguous.
To explore this question, we conducted the following test: 

For each unique eruption:
1.	Generate a random composition dataset using the analytical uncertainty for each observation.
2.	Remove all observations for one eruption from the dataset and set it aside to be treated as an unknown.
3.	Train the voting classifier on all other eruptions in our dataset.
4.	Predict the source for each observation in the left-out eruption sample.
"""

import glob as glob
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track
from rich.prompt import Prompt
from rich.theme import Theme
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm import tqdm

# custom plotting defaults
import mpl_defaults


# from https://github.com/amueller/introduction_to_ml_with_python/blob/master/mglearn/tools.py
# and tweaked slightly to add **kwargs for pcolor
def delta_confusion_matrix(
    values,
    xlabel,
    ylabel,
    xticklabels,
    yticklabels,
    cmap=None,
    vmin=None,
    vmax=None,
    ax=None,
    fmt="%0.2f",
    xtick_rot=90,
    label_size=8,
    val_limit=0,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    # plot the mean cross-validation scores
    img = ax.pcolor(np.flipud(values), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + 0.5)
    ax.set_yticks(np.arange(len(yticklabels)) + 0.5)
    ax.set_xticklabels(xticklabels, rotation=xtick_rot)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(), img.get_facecolors(), img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.4:
            c = "k"
        else:
            c = "w"
        if value <= val_limit:
            ax.text(
                x,
                y,
                fmt % value,
                color=c,
                ha="center",
                va="center",
                fontsize=label_size,
                alpha=0,
            )

        else:
            ax.text(
                x,
                y,
                fmt % value,
                color=c,
                ha="center",
                va="center",
                fontsize=label_size,
            )

    return img


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

data = pd.read_excel(f"{data_path}\\\B4_training_data_transformed_v2.xlsx")

model_weights = pd.read_excel(f"{data_path}\\\deployment_votingclassifier_weights.xlsx")
major_elements = data.loc[:, "Si_ppm":"P_ppm"].columns.tolist()
trace_elements = data.loc[:, "Ca":"U"].columns.tolist()
ratios = data.loc[:, "Sr/Y":"Rb/Cs"].columns.tolist()
volcanoes = data["volcano"].unique().tolist()
data["eruption"] = data["eruption"].astype(str)
eruptions = data["eruption"].unique().tolist()
rfe_features = [
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

rs = 0

estimators = [
    ("lda", LinearDiscriminantAnalysis(shrinkage=0.3, solver="lsqr")),
    (
        "logreg",
        LogisticRegression(C=10, penalty="l2", solver="newton-cg", max_iter=250),
    ),
    ("knc", KNeighborsClassifier(algorithm="auto", n_neighbors=5)),
    ("svc", SVC(random_state=rs, C=10, gamma=1, probability=True)),
    (
        "rfc",
        RandomForestClassifier(
            criterion="gini", max_depth=10, n_estimators=500, random_state=rs
        ),
    ),
    (
        "gbc",
        GradientBoostingClassifier(
            learning_rate=0.1, max_depth=2, n_estimators=250, random_state=rs
        ),
    ),
]
# construct the voting classifier
vcs_tuned = VotingClassifier(
    estimators=estimators,
    voting="soft",
    n_jobs=-1,
    weights=model_weights["test_f1_weighted"],
)

df = data.set_index("eruption")

# where everything gets dumped
# ASK THE USER IF THEY WANT TO CHANGE THE OUTPUT DIRECTORY FOR THE SAVED MODEL
# IF N, THEN IT IS THE SAME AS THE SCRIPT LOCATION
dir_change = Prompt.ask(
    "[bold gold1]\n\n2. Do you want to change the output file directory. It is recommended that this output directory is its own folder because one csv will be generated for each eruption [y/n][bold gold1]"
)

if dir_change == "y":
    output_dir = Prompt.ask("[bold gold1]2a. Enter the output directory[bold gold1]")
else:
    output_dir = export_path
output_dir = output_dir.replace('"', "")

console.print(
    f"\nOVERALL VARIANCE RESULT SPREADSHEET DIRECTORY:\n{export_path}\n", style="path"
)

console.print(
    f"\nINDIVIDUAL VARIANCE RESULT SPREADSHEET AND FIGURE DIRECTORY:\n{output_dir}\n",
    style="path",
)

console.print(
    "STARTING VARIANCE ANALYSIS...THIS IS GOING TO TAKE A WHILE", style="main"
)


for option in ["major", "rfe"]:
    if option == "major":
        myfeatures = major_elements
        console.print("Working on major element variance analysis", style="main")

    elif option == "rfe":
        myfeatures = rfe_features
        console.print("Working on rfe element variance analysis", style="main")
    console.print(f"Your features are {myfeatures}", style="result")

    for e in tqdm(range(len(eruptions))):
        # progress.console.print(f"working on eruption: {e}")
        # generate a new random composition based off analytical uncertainty
        # for each iteration of the run
        random_comps = np.random.normal(
            data.loc[:, major_elements + trace_elements],
            data.loc[
                :, [f"{element}_rel_std" for element in major_elements + trace_elements]
            ],
        )
        df[major_elements + trace_elements] = np.array(random_comps)
        # this is where you choose your features
        X_train = df.loc[
            [eruption for eruption in eruptions if eruption != eruptions[e]], myfeatures
        ]
        X_test = df.loc[eruptions[e], myfeatures]

        y_train = df.loc[
            [eruption for eruption in eruptions if eruption != eruptions[e]], "volcano"
        ]
        y_test = df.loc[eruptions[e], "volcano"]

        cv = RepeatedStratifiedKFold(n_splits=6, n_repeats=5, random_state=rs)
        vcs_tuned_scores = cross_validate(vcs_tuned, X_train, y_train, cv=cv, n_jobs=-1)
        vcs_tuned_test_accuracies = vcs_tuned_scores["test_score"]
        vcs_tuned.fit(X_train, y_train)

        vcs_tuned_predictions = vcs_tuned.predict(X_test)

        proba_df = pd.DataFrame(
            vcs_tuned.predict_proba(X_test), columns=vcs_tuned.classes_
        )
        proba_df.insert(0, "Target", y_test.tolist())
        proba_df.insert(1, "Prediction", vcs_tuned.predict(X_test))
        proba_df.index = proba_df.shape[0] * [eruptions[e]]
        proba_df.index.name = "eruption"
        output_report = pd.concat([proba_df, X_test], axis="columns")
        output_report.to_excel(
            f"{output_dir}\{eruptions[e]}_prediction_probabilities_vcs_{option}_tuned.xlsx",
            index=True,
        )


infiles = glob.glob("{}/*.xlsx".format(output_dir))

rfe_files = [file for file in infiles if "rfe" in file]

major_files = [file for file in infiles if "major" in file]

for option in tqdm(["major", "rfe"], unit="feature option"):
    if option == "major":
        files = major_files
        color = "C1"
        console.print(
            "COMBINING MAJOR ELEMENT VARIANCE RESULTS AND MAKING FIGURES\n",
            style="main",
        )
    elif option == "rfe":
        files = rfe_files
        color = "C0"
        console.print(
            "COMBINING RFE FEATURE VARIANCE RESULTS AND MAKING FIGURES\n", style="main"
        )

    df_list = []

    for file in files:
        df = pd.read_excel(file)
        df_list.append(df)

    combined_df = pd.concat(df_list)
    combined_df["eruption"] = combined_df["eruption"].astype(str)
    combined_df.set_index("eruption", inplace=True)
    combined_df.to_excel(f"{export_path}\\vcs_{option}_variance_results.xlsx")

    for eruption in tqdm(eruptions, unit="figures"):
        fig, ax = plt.subplots()
        combined_df.loc[eruption, :].boxplot(
            column=sorted(volcanoes),
            ax=ax,
            boxprops=dict(color=color, facecolor="whitesmoke"),
            medianprops=dict(color=color, lw=1),
            capprops=dict(linewidth=0),
            whiskerprops=dict(color=color),
            flierprops=dict(markeredgecolor=color, alpha=0.5),
            showfliers=True,
            patch_artist=True,
        )
        xticklabels = ax.get_xticklabels()
        ax.set_xticklabels(xticklabels, rotation=90)
        ax.set_ylabel("Probability")
        ax.minorticks_off()
        ax.grid()
        ax.set_title(
            f"{combined_df.loc[eruption,'Target'].unique()[0]} {eruption} n = {combined_df.loc[eruption,:].shape[0]} ",
            loc="left",
        )
        plt.savefig(
            f"{output_dir}\{eruption}_{option}_predictions_boxplot.pdf",
            bbox_inches="tight",
        )
        plt.close()
