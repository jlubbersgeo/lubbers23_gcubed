""" 
SUPPORT VECTOR MACHINE HYPERPARAMETER TUNING
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import  GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import (

    cross_validate,
    train_test_split,
)
from sklearn.model_selection import RepeatedStratifiedKFold
# custom plotting defaults
import mpl_defaults

# where all the figures get dumped
export_path = r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs"

# from https://github.com/amueller/introduction_to_ml_with_python/blob/master/mglearn/tools.py
# and tweaked slightly to add **kwargs for pcolor
def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f",xtick_rot = 90,label_size = 8,val_limit = 0, **kwargs):
    if ax is None:
        ax = plt.gca()
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax,**kwargs)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels,rotation = xtick_rot)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(), img.get_facecolors(),
                               img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.4:
            c = 'k'
        else:
            c = 'w'
        if value < val_limit:
            ax.text(x, y, fmt % value, color=c, ha="center", va="center",fontsize = label_size,alpha = 0)

        else:
            ax.text(x, y, fmt % value, color=c, ha="center", va="center",fontsize = label_size)
            
    return img




data = pd.read_excel(
    r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs\B4_training_data_transformed_v2.xlsx"
)
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
    random_state = rs
)


clf = SVC()
cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=5, random_state=rs)
scores = cross_validate(clf, X_train, y_train, cv=cv, n_jobs=-1)
clf.fit(X_train,y_train)
test_accuracies = scores['test_score']
print(f"Mean test accuracy: {np.round(np.mean(test_accuracies),3)} ± {np.round(np.std(test_accuracies),3)}")
#####################################################################
########################## HYPERARAMETER GRID ######################
###################################################################
param_grid = {'kernel' : ['rbf'],
              'C' : 10.**np.arange(-5,10),
              'gamma': 10.**np.arange(-5,10)
             }

grid_search = GridSearchCV(
    estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1,error_score = 'raise'
)

grid_search.fit(X_train,y_train)

report = pd.DataFrame.from_dict(grid_search.cv_results_)


######################################################################################
############################### OUTPUT FIGURE ########################################
#####################################################################################


report_scores = report["mean_test_score"].to_numpy()
report_scores = report_scores.reshape(len(param_grid["C"]), len(param_grid["gamma"]))

fig, ax = plt.subplots(figsize=(8, 8))
h = heatmap(
    report_scores,
    xlabel="gamma",
    xticklabels=["{:.2E}".format(val) for val in param_grid["gamma"]],
    ylabel="C",
    yticklabels=["{:.2E}".format(val) for val in param_grid["C"]],
    fmt="%0.3f",
    val_limit = .5,
    xtick_rot = 45,
    cmap = 'PuBu'

)
cbar = fig.colorbar(h, ax = ax, label = "mean accuracy", shrink = .6)
ax.minorticks_off()

    

plt.savefig('{}\SVM_hyperparams_plot.pdf'.format(export_path),bbox_inches = 'tight')
plt.show()


