"""
RANDOM FOREST CLASSIFIER HYPERPARMETER TUNING
1. train a base estimator using RFE features determined in 
3_model_feature_selection.py

2. create a hyperparameter grid to search over and train 
a model using each possible combination in the grid.

3. produce output figure showing the performance results
for each combination

"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.prompt import Prompt
from rich.theme import Theme
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)

# custom plotting defaults
import mpl_defaults


# from https://github.com/amueller/introduction_to_ml_with_python/blob/master/mglearn/tools.py
# and tweaked slightly to add **kwargs for pcolor
def heatmap(
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
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
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
        if value < val_limit:
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
    "[bold gold1] Enter the path to where figures should be exported[bold gold1]"
)
export_path = export_path.replace('"', "")

data_path = Prompt.ask(
    "[bold gold1] Enter the folder path to where transformed data are stored[bold gold1]"
)
data_path = data_path.replace('"', "")

data = pd.read_excel(f"{data_path}\\\B4_training_data_transformed_v2.xlsx")


major_elements = data.loc[:, "Si_ppm":"P_ppm"].columns.tolist()
trace_elements = data.loc[:, "Ca":"U"].columns.tolist()
ratios = data.loc[:, "Sr/Y":"Rb/Cs"].columns.tolist()


#######################################################################
################## BASE RANDOM FOREST MODEL#######################
#######################################################################
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

X_train, X_test, y_train, y_test = train_test_split(
    data.loc[:, rfe_features],
    data.loc[:, "volcano"],
    stratify=data.loc[:, "volcano"],
    test_size=0.25,
    random_state=rs,
)


clf = RandomForestClassifier(random_state=rs)
cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=5, random_state=rs)
scores = cross_validate(clf, X_train, y_train, cv=cv, n_jobs=-1)
clf.fit(X_train, y_train)
test_accuracies = scores["test_score"]
console.print(
    f"Mean test accuracy: {np.round(np.mean(test_accuracies),3)} ± {np.round(np.std(test_accuracies),3)}",
    style="result",
)

#####################################################################
########################## HYPERARAMETER GRID ######################
###################################################################
param_grid = {
    "criterion": ["gini", "entropy"],
    "n_estimators": [100, 250, 500, 1000],
    "max_depth": [2, 3, 5, 7, 10, 20],
}
t0 = time.time()
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    error_score="raise",
)

grid_search.fit(X_train, y_train)
t1 = time.time()
t_total = (t1 - t0) / 60
console.print(
    f"\nRandom Forest Classifier GridSearchCV completed in:\n{np.round(t_total,2)} min",
    style="result",
)
report = pd.DataFrame.from_dict(grid_search.cv_results_).set_index("param_max_depth")


######################################################################################
############################### OUTPUT FIGURE ########################################
#####################################################################################
fig, ax = plt.subplots(3, 2, figsize=(6, 7), constrained_layout=True)
axes = ax.ravel()

for a, depth in zip(axes, param_grid["max_depth"]):
    report_scores = report.loc[depth, "mean_test_score"].to_numpy()
    report_scores = report_scores.reshape(
        len(param_grid["criterion"]), len(param_grid["n_estimators"])
    )

    h = heatmap(
        report_scores,
        xlabel="n estimators",
        xticklabels=param_grid["n_estimators"],
        ylabel="criterion",
        yticklabels=param_grid["criterion"],
        vmin=0.5,
        vmax=1,
        fmt="%0.3f",
        val_limit=0.5,
        xtick_rot=45,
        ax=a,
        cmap="PuBu",
    )
    if depth == 2 or depth == 3 or depth == 5 or depth == 7:
        a.set_xlabel("")
    if depth == 3 or depth == 7 or depth == 20:
        a.set_ylabel("")

    a.minorticks_off()
    a.set_title(f"Max Depth: {depth}", fontsize=16)
cbar = fig.colorbar(
    h, ax=ax, label="mean accuracy", shrink=0.6, orientation="horizontal"
)
plt.savefig(
    "{}\Randomforest_hyperparams_plot.pdf".format(export_path), bbox_inches="tight"
)
plt.show()
