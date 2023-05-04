"""
LOGISTIC REGRESSION CLASSIFIER HYPERPARMETER TUNING
1. train a base estimator using RFE features determined in 
3_model_feature_selection.py

2. create a hyperparameter grid to search over and train 
a model using each possible combination in the grid.

3. produce output figure showing the performance results
for each combination

"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.prompt import Prompt
from rich.theme import Theme
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)

# custom plotting defaults
import mpl_defaults

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
data.head()


#######################################################################
################## BASE LOGISTIC REGRESSION MODEL######################
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


clf = LogisticRegression()
cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=5, random_state=rs)
scores = cross_validate(clf, X_train, y_train, cv=cv, n_jobs=-1)
clf.fit(X_train, y_train)
test_accuracies = scores["test_score"]
console.print(
    f"Mean test accuracy: {np.round(np.mean(test_accuracies),3)} ± {np.round(np.std(test_accuracies),3)}",
    style="result",
)


param_grid = {
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
}

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    error_score="raise",
)

grid_search.fit(X_train, y_train)


report = pd.DataFrame.from_dict(grid_search.cv_results_).set_index("param_solver")


#####################################################################
########################## HYPERARAMETER GRID ######################
###################################################################


param_grid = {
    "solver": ["newton-cg"],
    "penalty": ["none", "l2"],
    "C": [0.01, 1, 10, 100, 1000, 10000],
}

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    error_score="raise",
)

grid_search.fit(X_train, y_train)

report = pd.DataFrame.from_dict(grid_search.cv_results_).set_index("param_penalty")


######################################################################################
############################### OUTPUT FIGURE ########################################
#####################################################################################
""" 
The higher the "C" parameter is the higher the variance of the model and lower the bias. 
That being said, if we can keep the "C" parameter low while still keeping test accuracy high, 
we should proably side with that. Note...the default is 1 wit ha lbfgs solver. In this instance
 I think we are justified in using either newton-cg : penalty combo with a C parameter of 1 or 10. 
 C values of 100 or 1000 are probably more likey to be overfit
"""
fig, ax = plt.subplots()
ax.plot(
    report.loc["l2", "param_C"],
    report.loc["l2", "mean_test_score"],
    lw=3,
    marker="o",
    label="newton-cg : l2",
)
ax.plot(
    report.loc["none", "param_C"],
    report.loc["none", "mean_test_score"],
    lw=3,
    marker="o",
    ls="--",
    label="newton-cg : none",
)

ax.set_xscale("log")

ax.axhline(np.mean(test_accuracies), c="k", ls=":", label="lbfgs : l2")
ax.legend(
    loc="lower right",
    title="solver : penalty",
    fontsize=14,
    title_fontsize=16,
    shadow=True,
)
ax.set_xlabel("C value")
ax.set_ylabel("mean test score")
mpl_defaults.left_bottom_axes(ax)

ax.annotate(
    f"max accuracy using\nC: {grid_search.best_params_['C']}",
    (grid_search.best_params_["C"], grid_search.best_score_),
    xytext=(10, 0.8),
    arrowprops={"arrowstyle": "-|>"},
)
plt.savefig(
    "{}\logisticregression_hyperparams_plot.pdf".format(export_path),
    bbox_inches="tight",
)
plt.show()
