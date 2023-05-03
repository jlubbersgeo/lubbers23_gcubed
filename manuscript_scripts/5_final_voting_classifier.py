""" 
# The Voting Classifier
_Jordan Lubbers<br>
U.S. Geological Survey Alaska Volcano Observatory_<br>

Here we combine all the individual models with their tuned hyper-parameters into
a [soft (overal probability) voting](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier) scheme. 
We explore the performance difference between:
1. hard voting and soft voting for hyper parameter tuned models
2. hard voting and soft voting for base parameter models

We do this for both major element trained models as well as rfe feature trained models 
to ultimately compare them to one another. We then save the following models based on 
the training from only the training data:
- tuned voting classifier trained on rfe elements
- tuned voting classifier trained on major elements
- base voting classifier trained on rfe elements
- base voting classifier trained on major elements

We then save deployment models trained on our entire dataset:
- tuned voting classifier trained on rfe elements
- tuned voting classifier trained on major elements

We also save a spreadsheet for the tuned voting classifier trained on rfe elements that 
contains the weights for each individual algorithm used in the ensemble voting classifier. 
Finding these parameters is a relatively time intensive process so this may save some time 
in the future for notekeeping and documentation as this entire notebook, due to comparing 
a range of scenarios, takes about 25 minutes to run. 


"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_validate,
    train_test_split,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import classification_report, make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

import time
import sys
import pickle
from tqdm import tqdm

# custom plotting defaults
import mpl_defaults

# where all the figures get dumped
import time

export_path = r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs"
tstart = time.time()
dark_mode = False


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
    **kwargs
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



#########################################################################
######################### IMPORT DATA ###################################
#########################################################################
data = pd.read_excel(
    r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs\B4_training_data_transformed_v2.xlsx"
)
major_elements = data.loc[:, "Si_ppm":"P_ppm"].columns.tolist()
trace_elements = data.loc[:, "Ca":"U"].columns.tolist()
ratios = data.loc[:, "Sr/Y":"Rb/Cs"].columns.tolist()
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

#########################################################################
##############VOTING CLASSIFIER USING RFE FEATURES ######################
#########################################################################
X_train, X_test, y_train, y_test = train_test_split(
    data.loc[:, rfe_features],
    data.loc[:, "volcano"],
    stratify=data.loc[:, "volcano"],
    test_size=0.3,
    random_state=rs,
)

# ESTABLISH ALGORITHMS
estimators = [
    ("lda", LinearDiscriminantAnalysis(shrinkage=0.3, solver="lsqr")),
    ("logreg", LogisticRegression(C=10, penalty="l2", solver="newton-cg")),
    ("knc", KNeighborsClassifier(algorithm="auto", n_neighbors=5, )),
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


# FIND THE WEIGHTS OF EACH ALGORITHM TO BE USED IN THE SOFT VOTING SCHEME
print("\n FINDING RFE FEATURE WEIGHTS \n")
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
metrics = []
for estimator in tqdm(estimators):
    clf = estimator[1]
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1_score": make_scorer(f1_score),
    }

    df = pd.DataFrame.from_dict(
        cross_validate(
            estimator=clf, X=X_train, y=y_train, cv=skf, scoring=["f1_weighted", "accuracy"]
        )
    )
    d = df.loc[:,['test_f1_weighted','test_accuracy']].mean(axis="rows")
    metrics.append(d)


# SAVE THE WEIGHTS AS A DATAFRAME
metrics_df = pd.DataFrame(metrics)
metrics_df.index = [estimator[0] for estimator in estimators]


# CONSTRUCT THE ACTUAL VOTING CLASSIFIER
vcs_trace_tuned = VotingClassifier(
    estimators= estimators,
    voting="soft",
    n_jobs=-1,
    weights = metrics_df['test_f1_weighted']
)

t0 = time.time()
print("Working on the soft voting classifier...stay tuned!\n")
cv = RepeatedStratifiedKFold(n_splits=6, n_repeats=5, random_state=rs)
vcs_trace_tuned_scores = cross_validate(
    vcs_trace_tuned, X_train, y_train, cv=cv, n_jobs=-1
)
vcs_trace_tuned_test_accuracies = vcs_trace_tuned_scores["test_score"]
vcs_trace_tuned.fit(X_train, y_train)
t1 = time.time()
print(
    f"Survey says...\nSoft voting mean test accuracy: {np.round(np.mean(vcs_trace_tuned_test_accuracies),3)} ± {np.round(np.std(vcs_trace_tuned_test_accuracies),3)}\nfit time : {np.round((t1-t0)/60,2)} min\n"
)


# MAKE PREDICTIONS ON THE TEST DATASET AND BUILD CONFUSION MATRIX
vcs_trace_tuned_predictions = vcs_trace_tuned.predict(X_test)
vcs_trace_tuned_confusion_matrix = confusion_matrix(
    y_test, vcs_trace_tuned_predictions, labels=list(np.unique(y_test)), normalize=None
)

# SAVE THE MODEL 
pickle.dump(
    vcs_trace_tuned,
    open("{}\Aleutian_tuned_vcs_classifier_trace.sav".format(export_path), "wb"),
)


#########################################################################
##############VOTING CLASSIFIER USING MAJOR ELEMENTS ####################
#########################################################################

X_train_major, X_test_major, y_train_major, y_test_major = train_test_split(
    data.loc[:, major_elements],
    data.loc[:, "volcano"],
    stratify=data.loc[:, "volcano"],
    test_size=0.3,
    random_state=rs,
)
estimators = [
    ("lda", LinearDiscriminantAnalysis(shrinkage=0.3, solver="lsqr")),
    ("logreg", LogisticRegression(C=10, penalty="l2", solver="newton-cg")),
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

skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
metrics = []
print("\n FINDING MAJOR ELEMENT WEIGHTS \n")
for estimator in tqdm(estimators):

    clf = estimator[1]


    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1_score": make_scorer(f1_score),
    }

    df = pd.DataFrame.from_dict(
        cross_validate(
            estimator=clf, X=X_train, y=y_train, cv=skf, scoring=["f1_weighted", "accuracy"]
        )
    )
    d = df.loc[:,['test_f1_weighted','test_accuracy']].mean(axis="rows")
    metrics.append(d)
metrics_df = pd.DataFrame(metrics)
metrics_df.index = [estimator[0] for estimator in estimators]
# construct the voting classifier
vcs_major_tuned = VotingClassifier(
    estimators=estimators,
    voting="soft",
    n_jobs=-1,
    weights = metrics_df['test_f1_weighted']
)



t0 = time.time()
print("Working on the soft voting classifier...stay tuned!\n")
cv = RepeatedStratifiedKFold(n_splits=6, n_repeats=5, random_state=rs)
vcs_major_tuned_scores = cross_validate(
    vcs_major_tuned, X_train_major, y_train_major, cv=cv, n_jobs=-1
)
vcs_major_tuned_test_accuracies = vcs_major_tuned_scores["test_score"]
vcs_major_tuned.fit(X_train_major, y_train_major)
t1 = time.time()
print(
    f"Survey says...\nSoft voting mean test accuracy: {np.round(np.mean(vcs_major_tuned_test_accuracies),3)} ± {np.round(np.std(vcs_major_tuned_test_accuracies),3)}\nfit time : {np.round((t1-t0)/60,2)} min\n"
)



vcs_major_tuned_predictions = vcs_major_tuned.predict(X_test_major)
vcs_major_tuned_confusion_matrix = confusion_matrix(
    y_test_major,
    vcs_major_tuned_predictions,
    labels=list(np.unique(y_test_major)),
    normalize=None,
)



pickle.dump(
    vcs_major_tuned,
    open("{}\Aleutian_tuned_vcs_classifier_major_tuned.sav".format(export_path), "wb"),
)

#########################################################################
#########################MANUSCRIPT FIGURE S8 ##########################
#########################################################################
fig,ax = plt.subplot_mosaic(
    [
        ["major_matrix","."],
        ["major_matrix","major_bw"],
        ["major_matrix","."],
        ["rfe_matrix", "."],
        ["rfe_matrix","rfe_bw"],
        ["rfe_matrix","."],
    ],
    layout = 'constrained',
    figsize = (12,16),
    width_ratios = [1.5,1]
)
labels = sorted(y_test_major.unique())
lw = 0
# ===================================================
# ===================================================

h = delta_confusion_matrix(
    vcs_major_tuned_confusion_matrix,
    xlabel="Predicted Volcano",
    xticklabels=labels,
    ylabel="True Volcano",
    yticklabels=labels[::-1],
    ax=ax["major_matrix"],
    fmt="%0.0f",
    val_limit=0,
    xtick_rot=90,
    cmap="Oranges",
    linewidth=lw,
    ec="k",
)

ax["major_matrix"].minorticks_off()
ax["major_matrix"].minorticks_off()

# ===================================================
# ===================================================

h = delta_confusion_matrix(
    vcs_trace_tuned_confusion_matrix,
    xlabel="Predicted Volcano",
    xticklabels=labels,
    ylabel="True Volcano",
    yticklabels=labels[::-1],
    ax=ax["rfe_matrix"],
    fmt="%0.0f",
    val_limit=0,
    xtick_rot=90,
    cmap="PuBu",
    linewidth=lw,
    ec="k",
)

ax["rfe_matrix"].minorticks_off()
ax["rfe_matrix"].minorticks_off()

# ===================================================
# ===================================================

df = data.set_index("volcano")
volcanoes = df.index.unique().tolist()

lats = df["latitude"].unique()
lons = df["longitude"].unique()
loc_list = []
for source, lat, lon in zip(volcanoes, lats, lons):
    loc_list.append([lat, lon, source])

sorted_locations = np.array(sorted(loc_list, key=lambda x: int(x[1])))
df = df.loc[list(sorted_locations[:, 2]), :].reset_index()

proba_df = pd.DataFrame(
    vcs_trace_tuned.predict_proba(X_test), columns=vcs_trace_tuned.classes_
)
proba_df.insert(0, "Target", y_test.tolist())
proba_df.insert(1, "Prediction", vcs_trace_tuned.predict(X_test))

dff_tuned = pd.DataFrame()
pred_classes = (
    proba_df[proba_df["Target"] == proba_df["Prediction"]]["Target"].unique().tolist()
)

for source in sorted_locations[:, 2]:
    d = (
        proba_df[proba_df["Target"] == proba_df["Prediction"]]
        .set_index("Target")
        .loc[source][source]
    )
    dff_tuned = pd.concat([dff_tuned, d])

dff_tuned.reset_index(inplace=True)
dff_tuned.columns = ["Source", "Probability"]
dff_tuned.set_index("Source", inplace=True)


ax["rfe_bw"].boxplot(
    [dff_tuned.loc[source, "Probability"] for source in volcanoes],
    boxprops=dict(color="navy", facecolor="cornflowerblue"),
    medianprops=dict(color="navy", lw=1),
    capprops=dict(linewidth=0),
    whiskerprops=dict(color="navy"),
    flierprops=dict(markeredgecolor="navy", alpha=0.5),
    showfliers=False,
    patch_artist=True,
)
ax["rfe_bw"].set_xticklabels(volcanoes, rotation=75)
for i, source in zip(range(len(pred_classes)), volcanoes):
    y = dff_tuned.loc[source].to_numpy()
    q1, q3 = np.percentile(y, [25, 75])

    iqr = q3 - q1

    med = np.median(y)

    val = 1.01 * np.sort(y[y <= q3 + 1.5 * iqr])[-1]

    ax["rfe_bw"].text(i + 0.85, val, f"{dff_tuned.loc[source].shape[0]}", fontsize=8)

ax["rfe_bw"].minorticks_off()

ax["rfe_bw"].set_ylabel("Probability")

ax["rfe_bw"].set_ylim(0.3, 1.05)
mpl_defaults.left_bottom_axes(ax["rfe_bw"])
# ===================================================
# ===================================================


proba_df_major = pd.DataFrame(
    vcs_major_tuned.predict_proba(X_test_major), columns=vcs_major_tuned.classes_
)
proba_df_major.insert(0, "Target", y_test_major.tolist())
proba_df_major.insert(1, "Prediction", vcs_major_tuned.predict(X_test_major))

dff_major = pd.DataFrame()
pred_classes_major = (
    proba_df_major[proba_df_major["Target"] == proba_df_major["Prediction"]]["Target"]
    .unique()
    .tolist()
)

for source in sorted_locations[:, 2]:
    d = (
        proba_df_major[proba_df_major["Target"] == proba_df_major["Prediction"]]
        .set_index("Target")
        .loc[source][source]
    )
    dff_major = pd.concat([dff_major, d])

dff_major.reset_index(inplace=True)
dff_major.columns = ["Source", "Probability"]
dff_major.set_index("Source", inplace=True)


ax["major_bw"].boxplot(
    [dff_major.loc[source, "Probability"] for source in volcanoes],
    boxprops=dict(color="C1", facecolor="bisque"),
    medianprops=dict(color="C1", lw=1),
    capprops=dict(linewidth=0),
    whiskerprops=dict(color="C1"),
    flierprops=dict(markeredgecolor="C1", alpha=0.5),
    showfliers=False,
    patch_artist=True,
)
ax["major_bw"].set_xticklabels(volcanoes, rotation=75)
for i, source in zip(range(len(pred_classes_major)), volcanoes):
    y = dff_major.loc[source].to_numpy()
    q1, q3 = np.percentile(y, [25, 75])

    iqr = q3 - q1

    med = np.median(y)

    val = 1.01 * np.sort(y[y <= q3 + 1.5 * iqr])[-1]

    ax["major_bw"].text(i + 0.85, val, f"{dff_major.loc[source].shape[0]}", fontsize=8)

ax["major_bw"].minorticks_off()

ax["major_bw"].set_ylabel("Probability")

ax["major_bw"].set_ylim(0.3, 1.1)
mpl_defaults.left_bottom_axes(ax["major_bw"])
ax["major_bw"].set_title("Major element voting classifier", fontsize = 20,y = 1.1)
ax["rfe_bw"].set_title("RFE element voting classifier", fontsize = 20, y = 1.1)
plt.savefig(f"{export_path}/major_v_trace_voting_classifier_matrix_boxwhisker.pdf",bbox_inches = 'tight')
plt.show(block = False)

#########################################################################
################################ MANUSCRIPT FIGURE 5 ####################
#########################################################################
# gets the same indices for correct predictions
# this will give us an apples to apples comparison
trace_idx = proba_df[proba_df["Target"] == proba_df["Prediction"]].index
major_idx = proba_df_major[
    proba_df_major["Target"] == proba_df_major["Prediction"]
].index

both_idx = np.intersect1d(trace_idx, major_idx)
neither_idx = np.intersect1d(proba_df[proba_df["Target"] != proba_df["Prediction"]].index, proba_df_major[proba_df["Target"] != proba_df_major["Prediction"]].index)
major_only_idx = np.intersect1d(proba_df_major[
    proba_df_major["Target"] == proba_df_major["Prediction"]
].index, proba_df[proba_df["Target"] != proba_df["Prediction"]].index)

trace_only_idx = np.intersect1d(proba_df[
    proba_df["Target"] == proba_df["Prediction"]
].index, proba_df_major[proba_df_major["Target"] != proba_df_major["Prediction"]].index)



fig, ax = plt.subplots(5, 4, figsize=(8, 10),layout = 'constrained')
axes = ax.ravel()
for a, source, i in zip(axes, sorted(volcanoes), range(len(axes))):

    a.plot(
        proba_df.iloc[both_idx, :].set_index("Target").loc[source, source],
        proba_df_major.iloc[both_idx, :].set_index("Target").loc[source, source],
        marker="o",
        ls="",
        mec = "darkgreen",
        mfc="mediumseagreen",
        ms=5,
    )
    if source in proba_df.iloc[neither_idx, :].set_index("Target").index:

        a.plot(
        proba_df.iloc[neither_idx, :].set_index("Target").loc[source, source],
        proba_df_major.iloc[neither_idx, :].set_index("Target").loc[source, source],
        marker="^",
        ls="",
        mec="maroon",
        mfc = 'lightcoral',
        ms=5,
    )
    if source in proba_df.iloc[major_only_idx, :].set_index("Target").index:

        a.plot(
        proba_df.iloc[major_only_idx, :].set_index("Target").loc[source, source],
        proba_df_major.iloc[major_only_idx, :].set_index("Target").loc[source, source],
        marker="s",
        ls="",
        mec="C1",
        mfc = 'bisque',
        ms=5,
    )

    if source in proba_df.iloc[trace_only_idx, :].set_index("Target").index:

        a.plot(
        proba_df.iloc[trace_only_idx, :].set_index("Target").loc[source, source],
        proba_df_major.iloc[trace_only_idx, :].set_index("Target").loc[source, source],
        marker="d",
        ls="",
        mfc="lightsteelblue",
        mec = 'C0',
        ms=5,
    )

    a.set_xticks(np.array([0,.5,1.]))

    if i % 4 == 0:
        a.set_ylabel("P$_{ME}$", fontsize=16)
    if i > 13:
        a.set_xlabel("P$_{RFE}$", fontsize=16)
    a.set_aspect(1)
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_yticks(a.get_xticks())
    a.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), "k--",lw = 1)
    a.set_title(f"{source}", loc="left", x=0.05, y=0.82)
    # a.text(0.05, 0.78, f"{n}/{n_original}", transform=a.transAxes, fontsize=8)
    a.text(0.3, 0.37, "P$_{ME}$ = P$_{RFE}$", rotation = 45, fontsize=8)

for a in axes[-2:]:
    a.axis("off")

legend_elements = [
    Line2D([0], [0],marker = 'o',markerfacecolor="mediumseagreen", markeredgecolor="darkgreen", ls = '', label="both correct"),
    Line2D([0], [0],marker = 's',markerfacecolor="bisque", markeredgecolor="C1", ls = '', label="ME only"),
    Line2D([0], [0],marker = 'd',markerfacecolor="lightsteelblue", markeredgecolor="C0", ls = '', label="RFE only"),
    Line2D([0], [0],marker = '^',markerfacecolor="lightcoral", markeredgecolor="maroon", ls = '', label="Neither"),
]
# add the legend
ax[4,2].legend(
    handles=legend_elements,
    title="Prediction",
    title_fontsize=20,
    fontsize = 12,
    frameon=True,
    shadow=True,
    facecolor="w",
)
plt.savefig(
    "{}\major_vs_trace_tuned_prediction_probabilities.pdf".format(export_path),
    bbox_inches="tight",
)

plt.show(block = False)
""" 
## Generate deployment model
For the purposes of building, testing, and validating our model we split the data into 
a train and test set. This sets aside 30% of the data to be used for testing, ultimately 
making it not available for further training. While imperative for building and testing 
our features/hyper-parameters, for deployment we want to capitalize on all the data we've 
collected. Below we train on our entire dataset and use this model as our deployment model 
to use with unknown data (IODP, Derkachev, etc.). 

"""

# Deployment model that is trained on everything rather than 70%
X_train, X_test, y_train, y_test = train_test_split(
    data.loc[:, rfe_features],
    data.loc[:, "volcano"],
    stratify=data.loc[:, "volcano"],
    test_size=0.02,
    random_state=rs,
)
estimators = [
    ("lda", LinearDiscriminantAnalysis(shrinkage=0.3, solver="lsqr")),
    ("logreg", LogisticRegression(C=10, penalty="l2", solver="newton-cg")),
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

skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
metrics = []
print("\n WORKING ON DEPLOYMENT MODEL \n")

print("\n FINDING WEIGHTS \n")

for estimator in tqdm(estimators):

    clf = estimator[1]

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1_score": make_scorer(f1_score),
    }

    df = pd.DataFrame.from_dict(
        cross_validate(
            estimator=clf,
            X=X_train,
            y=y_train,
            cv=skf,
            scoring=["f1_weighted", "accuracy"],
        )
    )
    d = df.loc[:, ["test_f1_weighted", "test_accuracy"]].mean(axis="rows")
    metrics.append(d)

metrics_df = pd.DataFrame(metrics)
metrics_df.index = [estimator[0] for estimator in estimators]
metrics_df.index.name = "model"
# construct the voting classifier
vcs_trace_tuned_deployment = VotingClassifier(
    estimators=estimators,
    voting="soft",
    n_jobs=-1,
    weights=metrics_df["test_f1_weighted"],
)


vcs_trace_tuned_deployment.fit(X_train, y_train)

pickle.dump(
    vcs_trace_tuned_deployment,
    open(
        "{}\Aleutian_tuned_vcs_classifier_trace_deployment.sav".format(export_path),
        "wb",
    ),
)
metrics_df = metrics_df.reset_index()
metrics_df.to_excel(
    f"{export_path}\test_deployment_votingclassifier_weights.xlsx",
    index=False,
)

tfinal = time.time()
print(f"Script runtime is {np.round((tfinal - tstart)/60,2)} minutes")
