""" 
# Proximity calculations
_Jordan Lubbers<br>
U.S. Geological Survey Alaska Volcano Observatory_<br>

This notebook looks at random forest proximities between observations for 
both rfe feature and major element trained models. In this instance proximity 
is defined as the number of times two samples within the training dataset reach 
the same leaf within a decision tree, normalized by the number of trees in the forest. 

"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from rich.console import Console
from rich.prompt import Prompt
from rich.theme import Theme
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import MDS
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)

# custom plotting defaults
import mpl_defaults
from aleutian_colors import create_aleutian_colors
from kinumaax.source.kinumaax import learning as kl


def get_proximity_matrix(classifier,X_train):
    
    # Apply trees in the forest to X, return leaf indices.
    # a n_obs x n_trees matrix where each value is the index of the leaf
    # an observation ends up in
    terminals = classifier.apply(X_train)

    # number of trees in the forest
    n_trees = terminals.shape[1]
    # number of observations in training dataset
    n_obs = terminals.shape[0]


    prox_mat = 1 * np.equal.outer(n_obs, n_obs)

    for i in range(1, n_trees):

        # for each tree in the forest get the leaf indices
        # for each observation
        tree_obs = terminals[:, i]

        # take the outer product of each array. This is the same as
        # multiplying each element of a by each element of a
        # a1 x a1, a2 x a2, etc. which gives a (n_obs x n_obs) shape array.
        # Then check to see where they are equal. Fill those array values
        # with 1 instead of True or False. += is equivalent of summing all
        # trees in the forest to get total times each observation has the
        # same leaf index
        prox_mat += 1 * np.equal.outer(tree_obs, tree_obs)


    # normalize by total number of trees in the forest such that diags are 1.
    prox_mat = prox_mat / n_trees
    
    return prox_mat



custom_theme = Theme(
    {"main": "bold gold1", "path": "bold steel_blue1", "result": "magenta"}
)
console = Console(theme=custom_theme)

export_path = Prompt.ask("[bold gold1] Enter the path to where figures should be exported[bold gold1]")
export_path = export_path.replace('"',"")

data_path = Prompt.ask("[bold gold1] Enter the folder path to where transformed data are stored[bold gold1]")
data_path = data_path.replace('"',"") 

data = pd.read_excel(
    f"{data_path}\\\B4_training_data_transformed_v2.xlsx").set_index("volcano")
major_elements = data.loc[:, "Si_ppm":"P_ppm"].columns.tolist()
trace_elements = data.loc[:, "Ca":"U"].columns.tolist()
ratios = data.loc[:, "Sr/Y":"Rb/Cs"].columns.tolist()


# SORT VOLCANOES BY LOCATION
volcanoes = data.index.unique().tolist()
# by along arc distance (broadly east to west)
sorted_locations = [
    "Churchill",
    "Hayes",
    "Redoubt",
    "Augustine",
    "Kaguyak",
    "Katmai",
    "Ugashik",
    "Aniakchak",
    "Black Peak",
    "Veniaminof",
    "Emmons Lake",
    "Fisher",
    "Makushin",
    "Okmok",
    "Adagdak",
    "Gareloi",
    "Semisopochnoi",
    "Davidof",
]
sorted_abbreviations = [
    "CU",
    "HA",
    "RD",
    "AU",
    "KG",
    "KT",
    "UG",
    "AN",
    "BP",
    "VE",
    "EM",
    "FI",
    "MA",
    "OK",
    "AD",
    "GA",
    "SS",
    "DV"
]
data = data.loc[sorted_locations, :].reset_index()


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

# CALCULATE PROXIMITY MATRICES FOR EACH FEATURE SPACE
proximity_matrices = []
X_trains = []
y_trains = []
for features in [major_elements,rfe_features]:
    rs = 0

    X_train, X_test, y_train, y_test = train_test_split(
        data.loc[:, features],
        data.loc[:, "volcano"],
        stratify=data.loc[:, "volcano"],
        test_size=0.25,
        random_state=rs,
    )

    # optimally tuned hyperparameters
    clf = RandomForestClassifier(
        criterion="gini", max_depth=10, n_estimators=500, random_state=rs
    )
    X_train.index.name = "index"
    y_train.index.name = "index"
    X_train = X_train.sort_values(by="index")
    y_train = y_train.loc[X_train.index]

    cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=5, random_state=rs)
    scores = cross_validate(clf, X_train, y_train, cv=cv, n_jobs=-1)
    clf.fit(X_train, y_train)
    test_accuracies = scores["test_score"]
    console.print(f"feature space: {features}")
    console.print(
        f"Mean test accuracy: {np.round(np.mean(test_accuracies),3)} ± {np.round(np.std(test_accuracies),3)}",style = "result"
    )
    #proximity matrix
    prox_mat = get_proximity_matrix(clf, X_train)

    # dissimilarity matrix
    dis_mat = 1.0 - prox_mat



    prox_mat_original = prox_mat.copy()
    
    proximity_matrices.append(prox_mat)
    X_trains.append(X_train)
    y_trains.append(y_train)

####################################################################################
######################## MANUSCRIPT FIGURE 4 ########################################
#####################################################################################
fig, ax = plt.subplots(2,1, figsize = (6,12),constrained_layout = True)
axes = ax.ravel()

for matrix,a,model,labels in zip(proximity_matrices, axes, ['Major elements', 'RFE features'],y_trains):
    m = a.matshow(matrix, cmap = "turbo", vmin = 0, vmax = 1)
    a.set_title(model,fontsize = 18, loc = 'right', y = 0.93, c = 'w')
    a.xaxis.set_ticks_position("bottom")
    # a.set_xlabel("n$_{obs}$")
    a.set_ylabel("n$_{obs}$")
    
    d = pd.DataFrame(labels)
    d["count"] = np.arange(0,d.shape[0])
    d = d.set_index("volcano")
    
    for volcano in np.unique(y_train.values):
        start = d.loc[volcano,"count"].min()
        x = d.loc[volcano, "count"].max()
        y = d.loc[volcano, "count"].median()
        # ax.text(x, y, volcano, c="k", fontsize=8)
        a.add_patch(Rectangle(xy = (start,x), width = x - start,height = (start - x),color = 'none',ec = 'w',lw = .75),)

fig.supxlabel("n$_{obs}$",fontsize = 20, y = 0.05, x = .6)
cbar = fig.colorbar(m, ax = ax, label = "proximity", shrink = .4, orientation = 'horizontal',anchor = (0.1,-5) )
plt.savefig("{}\proximity_matrix_major_v_ratios.pdf".format(export_path),bbox_inches = 'tight')   
plt.show()