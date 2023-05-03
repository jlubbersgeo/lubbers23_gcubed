import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    cross_validate,
    train_test_split,
)
from sklearn.model_selection import RepeatedStratifiedKFold
# custom plotting defaults
import mpl_defaults

# where all the figures get dumped
export_path = r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs"


data = pd.read_excel(
    r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs\B4_training_data_transformed_v2.xlsx"
)
major_elements = data.loc[:, "Si_ppm":"P_ppm"].columns.tolist()
trace_elements = data.loc[:, "Ca":"U"].columns.tolist()
ratios = data.loc[:, "Sr/Y":"Rb/Cs"].columns.tolist()

#######################################################################
################## BASE LDA MODEL#######################
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
    random_state = rs
)


clf = LinearDiscriminantAnalysis()
cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=5, random_state=rs)
scores = cross_validate(clf, X_train, y_train, cv=cv, n_jobs=-1)
clf.fit(X_train,y_train)
test_accuracies = scores['test_score']
print(f"Mean test accuracy: {np.round(np.mean(test_accuracies),3)}")

#####################################################################
########################## HYPERARAMETER GRID ######################
###################################################################

param_grid = {'solver': ['lsqr', 'eigen'],
              'shrinkage' : np.arange(0.01,1.01,0.01)
             }

grid_search = GridSearchCV(
    estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1,error_score = 'raise'
)

grid_search.fit(X_train,y_train)

report = pd.DataFrame.from_dict(grid_search.cv_results_).set_index('param_solver')



######################################################################################
############################### OUTPUT FIGURE ########################################
#####################################################################################

fig, ax = plt.subplots()
ax.plot(
    report.loc["lsqr", "param_shrinkage"],
    report.loc["lsqr", "mean_test_score"],
    lw=3,
    label="lsqr",
)
ax.plot(
    report.loc["eigen", "param_shrinkage"],
    report.loc["eigen", "mean_test_score"],
    ls="--",
    label="eigen",
)
ax.axhline(np.mean(test_accuracies), c="k", ls=":", label="svd (default)")
ax.legend(loc="lower left", title="solver", fontsize=14, title_fontsize=16, shadow=True)
ax.set_xlabel("shrinkage value")
ax.set_ylabel("mean test score")
mpl_defaults.left_bottom_axes(ax)

ax.annotate(
    f"max accuracy using\nshrinkage: {grid_search.best_params_['shrinkage']}",
    (grid_search.best_params_["shrinkage"], grid_search.best_score_),
    xytext=(0.2, 0.94),
    arrowprops={"arrowstyle": "-|>"},
)
plt.savefig('{}\LDA_hyperparams_plot.pdf'.format(export_path),bbox_inches = 'tight')
plt.show()