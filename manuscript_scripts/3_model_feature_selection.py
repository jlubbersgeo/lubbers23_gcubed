"""
# Feature engineering

_Jordan Lubbers<br>
U.S. Geological Survey Alaska Volcano Observatory_<br>

This takes our training dataset and explores the feature engineering process 
by looking at a few different scenarios for features. We'll use a basic ```RandomForestClassifier``` 
so as to keep hyper-parameter tuning separate and allow for metrics like feature importances 
to be calculated. The scenarios are as follows:

1. Major elements only
2. Trace elements only
3. Ratios only
4. All components
5. Recursive feature elimination (RFE) chosen features
6. A subset of RFE features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

sys.path.append(
    r"C:\Users\jlubbers\OneDrive - DOI\Research\Coding\lubbers23_gcubed\kinumaax\source"
)

from kinumaax import learning as kl
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

import time
import mpl_defaults

# where all the figures get dumped
export_path = r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs"
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
    ticklabel_size = 8,
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
    ax.set_xticklabels(xticklabels, rotation=xtick_rot,size = ticklabel_size)
    ax.set_yticklabels(yticklabels,size = ticklabel_size)
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



t0 = time.time()

data = pd.read_excel(
    r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs\B4_training_data_transformed_v2.xlsx"
)
major_elements = data.loc[:, "Si_ppm":"P_ppm"].columns.tolist()
trace_elements = data.loc[:, "Ca":"U"].columns.tolist()
ratios = data.loc[:, "Sr/Y":"Rb/Cs"].columns.tolist()
data.head()

################################################################################
############################# MAJOR ELEMENTS AS FEATURES ######################
################################################################################
print("\nWORKING ON MAJOR ELEMENTS AS FEATURES\n")
rs = 0
# instantiate model with algorithm type, # of estimators,
# max depth of the trees, and fix the random state
majors_model = kl.tephraML(name="major_elements")
majors_model.instantiate(
    model_type="random_forest", n_estimators=500, max_depth=6, random_state=rs
)
majors_model.get_data(data)
majors_model.split_data(test_size=0.3, random_state=rs)
majors_model.get_train_test_data(feature_cols=major_elements, target_col="volcano")
majors_model.train_model()
majors_model.predict()
majors_model.get_feature_importance(kind="permutation", n_iterations=10, n_jobs=-1)
majors_model.make_confusion_matrix(normalize=None)
majors_model.get_prediction_probability()
majors_model.get_cross_val_score(
    stratified=True, n_splits=5, shuffle=True, random_state=rs
)

################################################################################
############################# TRACE ELEMENTS AS FEATURES ######################
################################################################################
print("\nWORKING ON TRACE ELEMENTS AS FEATURES\n")
traces_model = kl.tephraML(name="trace_elements")
traces_model.instantiate(
    model_type="random_forest", n_estimators=500, max_depth=6, random_state=rs
)
traces_model.get_data(data)
traces_model.split_data(test_size=0.3, random_state=rs)
traces_model.get_train_test_data(feature_cols=trace_elements, target_col="volcano")
traces_model.train_model()
traces_model.predict()
traces_model.get_feature_importance(kind="permutation", n_iterations=10, n_jobs=-1)
traces_model.make_confusion_matrix(normalize=None)
traces_model.get_prediction_probability()
traces_model.get_cross_val_score(
    stratified=True, n_splits=5, shuffle=True, random_state=rs
)

################################################################################
############################# TRACE RATIOS AS FEATURES ######################
################################################################################
print("\nWORKING ON TRACE ELEMENT RATIOS AS FEATURES\n")
ratios_model = kl.tephraML(name="trace_ratios")
ratios_model.instantiate(
    model_type="random_forest", n_estimators=500, max_depth=6, random_state=rs
)
ratios_model.get_data(data)
ratios_model.split_data(test_size=0.3, random_state=rs)
ratios_model.get_train_test_data(feature_cols=ratios, target_col="volcano")
ratios_model.train_model()
ratios_model.predict()
ratios_model.get_feature_importance(kind="permutation", n_iterations=10, n_jobs=-1)
ratios_model.make_confusion_matrix(normalize=None)
ratios_model.get_prediction_probability()
ratios_model.get_cross_val_score(
    stratified=True, n_splits=5, shuffle=True, random_state=rs
)

################################################################################
############################# ALL 3 ABOVE SCENARIOS COMBINED AS FEATURES #######
################################################################################
print("\nWORKING ON MAJOR ELEMENTS + TRACE ELEMENTS + TRACE ELEMENT RATIOS AS FEATURES\n")
all_model = kl.tephraML(name="all_elements")
all_model.instantiate(
    model_type="random_forest", n_estimators=500, max_depth=6, random_state=rs
)
all_model.get_data(data)
all_model.split_data(test_size=0.3, random_state=rs)
all_model.get_train_test_data(
    feature_cols=major_elements + trace_elements + ratios, target_col="volcano"
)
all_model.train_model()
all_model.predict()
all_model.get_feature_importance(kind="permutation", n_iterations=10, n_jobs=-1)
all_model.make_confusion_matrix(normalize=None)
all_model.get_prediction_probability()
all_model.get_cross_val_score(
    stratified=True, n_splits=5, shuffle=True, random_state=rs
)

################################################################################
############################# COMPUTE RECURRING FEATURE ELIMINATION ############
################################################################################
print('\n')
all_model.compute_rfe(cross_validate=True, step=1, cv=7)

################################################################################
############################# PLOT RFE RESULTS ######################
################################################################################
fig, ax = plt.subplots(figsize=(10, 4))

all_model.plot_feature_importance(
    show_error=False, sorted=True, ax=ax, fc='C0', ec='navy', width=0.8
)
ax.set_xticks(all_model.rfe_results["feature"])
ax.set_xticklabels(all_model.rfe_results["feature"], rotation=90, fontsize=10)
for patch, rank in zip(ax.patches, all_model.rfe_results["rank"]):
    ax.annotate(
        str(rank),
        (patch.get_x() + patch.get_width() / 2, patch.get_height()),
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax.minorticks_off()
ax.set_ylim(bottom=0)
mpl_defaults.left_bottom_axes(ax)
ax.text(
    0.1, 0.95, "*Bars labeled with a 1 are those chosen by RFECV", transform=ax.transAxes
)
labels = [
    "".join([i for i in l.get_text() if not i.isdigit()]) for l in ax.get_xticklabels()
]
labels = [label if "ppm" not in label else label.split("_")[0] for label in labels]
ax.set_xticklabels(labels)
ax.set_ylabel("Mean decrease in accuracy")
plt.savefig('{}\RFE_annotated_barplot.pdf'.format(export_path),bbox_inches = "tight")
plt.show()


################################################################################
############################# TOP RANKED RFE ELEMENTS AS FEATURES ##############
################################################################################
print("\nWORKING ON TOP RANKED RFE FEATURES AS FEATURES\n")
useful_features = all_model.rfe_results.set_index("rank").loc[1, :]["feature"].tolist()
# useful_features.remove("133Cs")
bad_features = ['Mn','Mn_ppm', "P_ppm", "Ti_ppm"]
useful_features = [feature for feature in useful_features if feature not in bad_features]
rfe_model = kl.tephraML(name="rfe_elements")
rfe_model.instantiate(
    model_type="random_forest", n_estimators=500, max_depth=6, random_state=rs
)
rfe_model.get_data(data)
rfe_model.split_data(test_size=0.3, random_state=rs)
rfe_model.get_train_test_data(feature_cols=useful_features, target_col="volcano")
rfe_model.train_model()
rfe_model.predict()
rfe_model.get_feature_importance(kind="permutation", n_iterations=10, n_jobs=-1)
rfe_model.make_confusion_matrix(normalize=None)
rfe_model.get_prediction_probability()
rfe_model.get_cross_val_score(
    stratified=True, n_splits=5, shuffle=True, random_state=rs
)


################################################################################
############################# PLOT RFE PRECISION RECALL ######################
################################################################################
dark_mode = False
colorblind_colors = mpl_defaults.create_colorblind_palette(n=9)
rfe_stats_list = []
rfe_features = all_model.rfe_results.sort_values(by="importance", ascending=False)[
    "feature"
].tolist()
clf = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=500, max_depth=6, random_state=rs)
)

fig, ax = plt.subplots(
    figsize=(4, 4),
)
# n_classes = np.arange(2,50,2)
n_classes = np.array([1, 2, 6, 12, 50])
linestyles = ["-", "-", "--", "--", ":"]
for i, color, style in zip(n_classes, colorblind_colors, linestyles):

    # split into training and test data but binarize the labels
    Y = label_binarize(
        data.loc[:, "volcano"], classes=data["volcano"].unique().tolist()
    )
    X_train, X_test, y_train, y_test = train_test_split(
        data.loc[:, rfe_features[:i]],
        Y,
        stratify=data.loc[:, "volcano"],
        test_size=0.25,
        random_state=rs,
    )
    # fit the one v rest classifier
    clf.fit(X_train, y_train)

    y_score = clf.predict_proba(X_test)

    # precision recall curve values as well as avg
    precision = dict()
    recall = dict()
    average_precision = dict()

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_test, y_score, average="micro"
    )

    # display the average
    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name=f"{i} feature avg", color=color, ls=style)


ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
handles, labels = display.ax_.get_legend_handles_labels()
ax.legend(
    handles=handles,
    labels=[label.split("(")[0] for label in labels],
    loc="upper right",
    ncol=1,
    bbox_to_anchor=(1.5, 1),
    fontsize=8,
)
plt.savefig(
    "{}\RFE_precision_recall_summary.pdf".format(export_path), bbox_inches="tight"
)
if dark_mode is True:

    mpl_defaults.make_dark_bkgd_compatible(ax=ax, bkgd_color="none")
    ax.set_xlim(0,1.05)
    ax.set_ylim(0,1.05)
    plt.savefig(
        "{}\RFE_precision_recall_summary_darkmode.pdf".format(export_path),
        bbox_inches="tight",
    )
    plt.savefig(
        "{}\RFE_precision_recall_summary_darkmode.png".format(export_path),
        bbox_inches="tight",
    )
plt.show()

################################################################################
############################# FINAL FEATURES AS FEATURES ######################
################################################################################
print("\nWORKING ON TOP RANKED PETROLOGICALLY SIFNICANG ELEMENTS AS FEATURES\n")
rfe_small_model = kl.tephraML(name="rfe_subset_elements")
rfe_small_model.instantiate(
    model_type="random_forest", n_estimators=500, max_depth=6, random_state=rs
)
rfe_small_model.get_data(data)
rfe_small_model.split_data(test_size=0.3, random_state=rs)
rfe_small_model.get_train_test_data(
    feature_cols=[
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
    ],
    target_col="volcano",
)
rfe_small_model.train_model()
rfe_small_model.predict()
rfe_small_model.get_feature_importance(kind="permutation", n_iterations=10, n_jobs=-1)
rfe_small_model.make_confusion_matrix(normalize=None)
rfe_small_model.get_prediction_probability()
rfe_small_model.get_cross_val_score(
    stratified=True, n_splits=5, shuffle=True, random_state=rs
)

stats_list = []
for model in [majors_model, traces_model, ratios_model, all_model, rfe_model,rfe_small_model]:

    df = pd.DataFrame.from_dict(
        classification_report(
            model.test_target_data,
            model.predicted_class,
            output_dict=True,
            zero_division=0,
        )
    ).T.reset_index()
    df.columns = ["volcano", "precision", "recall", "f1-score", "n"]
    df.index = [model.name] * df.shape[0]
    df.index.name = "model_type"
    stats_list.append(df)

stats_df = pd.concat(stats_list)
stats_df = stats_df.reset_index().set_index(["model_type", "volcano"])
################################################################################
############################# SAVING STATS FOR EACH MODEL SCENARIO #############
################################################################################
print("\nSAVING ALL SCENARIOS STATISTICS\n")
stats_df.loc[
    (
        [
            "major_elements",
            "trace_elements",
            "trace_ratios",
            "all_elements",
            "rfe_elements",
            "rfe_subset_elements"
        ],
        ["weighted avg"],
    ),
    :,
].T.to_excel("{}\Feature_engineering_summary_table.xlsx".format(export_path))


################################################################################
############################# PRECISION RECALL CURVE FOR ALL SCENARIOS #########
################################################################################

clf = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=500, max_depth=6, random_state=rs)
)
fig, ax = plt.subplots(figsize=(4, 4),)
ax_inset = ax.inset_axes([0.2,0.2,0.5,0.5])
ax_inset.set_xlim(.9,1)
ax_inset.set_ylim(.9,1)
# n_classes = np.arange(2,50,2)
features_dict = {
    "major elements": major_elements,
    "trace elements": trace_elements,
    "trace ratios": ratios,
    "all elements": major_elements + trace_elements + ratios,
    "rfe elements": useful_features,
    "rfe subset elements": [
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
    ],
}
linestyles = ['-', '-', '--', '--',':',':']
for key,value, color, style in zip(features_dict.keys(),features_dict.values(), colorblind_colors,linestyles):

    # split into training and test data but binarize the labels
    Y = label_binarize(
        data.loc[:, "volcano"], classes=data["volcano"].unique().tolist()
    )
    X_train, X_test, y_train, y_test = train_test_split(
        data.loc[:, value],
        Y,
        stratify=data.loc[:, "volcano"],
        test_size=0.25,
        random_state=rs,
    )
    # fit the one v rest classifier
    clf.fit(X_train, y_train)

    y_score = clf.predict_proba(X_test)

    # precision recall curve values as well as avg
    precision = dict()
    recall = dict()
    average_precision = dict()

   
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_test, y_score, average="micro"
    )

    

    # display the average
    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(
        ax=ax, name=key, lw = 1, color = color, ls = style
    )

    display.plot(ax = ax_inset, name = key, color = color, ls = style) 


ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
# ax_inset.set_xticklabels([])
ax_inset.set_yticks(ax_inset.get_xticks())

ax_inset.set_xlabel("")
ax_inset.set_ylabel("")


handles, labels = display.ax_.get_legend_handles_labels()
ax.legend(
    handles=handles,
    labels=[label.split("(")[0] for label in labels],
    loc="upper right",
    ncol=1,
    bbox_to_anchor=(1.5, 1),
    fontsize=8,
)

ax_inset.get_legend().remove()

ax.indicate_inset_zoom(ax_inset, edgecolor = "black")

plt.savefig(
    "{}\Feature_selection_precision_recall_summary.pdf".format(export_path), bbox_inches="tight"
)


if dark_mode is True:

    mpl_defaults.make_dark_bkgd_compatible(ax=ax, bkgd_color="none")
    ax.set_xlim(0,1.05)
    ax.set_ylim(0,1.05)
    plt.savefig(
        "{}\Feature_selection_precision_recall_summary_darkmode.pdf".format(export_path), bbox_inches="tight"
    )
    plt.savefig(
        "{}\Feature_selection_precision_recall_summary_darkmode.png".format(export_path), bbox_inches="tight"
    )
plt.show()

##################################################################################################
############################ MANUSCRIPT FIGURE 3 ################################################
#################################################################################################
labels = sorted(majors_model.test_target_data.unique())
dark_mode = True
fig, ax = plt.subplot_mosaic(
    [
    ["major_bar","rfe_bar"],
    ["major_matrix","rfe_matrix"]
    ],
    gridspec_kw = {"height_ratios": [0.2,0.8], "hspace": 0.01},
    figsize = (18,12),
    layout = 'constrained'
    )

majors_model.plot_feature_importance(
    ax=ax["major_bar"],
    show_error=False,
    sorted=True,
    ec="C1",
    fc="bisque",
    ecolor="k",
    error_kw={"linewidth": 0.5},
)

ax["major_bar"].minorticks_off()
bar_labels = [label.get_text() for label in ax["major_bar"].get_xticklabels()]
ax["major_bar"].set_xticklabels([label.split("_")[0] for label in bar_labels])
ax["major_bar"].set_ylabel("Accuracy\ndecrease", fontsize=14)
ax["major_bar"].set_xlabel("Feature")
mpl_defaults.left_bottom_axes(ax["major_bar"])
for patch, label in zip(ax["major_bar"].patches, ax["major_bar"].get_xticklabels()):
    ax["major_bar"].annotate(
        label.get_text(),
        (patch.get_x() + patch.get_width() / 2, patch.get_height() + .003),
        ha="center",
        va="bottom",
        fontsize=14,
        rotation = 60
    )
ax["major_bar"].set_xticks('')
#-----------------------------------------------------
rfe_small_model.plot_feature_importance(
    ax=ax["rfe_bar"],
    show_error=False,
    sorted=True,
    ec="navy",
    fc="C0",
    ecolor="k",
    error_kw={"linewidth": 0.5},
)

# ax["rfe_bar"].minorticks_off()
bar_labels = [label.get_text() for label in ax["rfe_bar"].get_xticklabels()]
ax["rfe_bar"].set_xticklabels([label.split("_")[0] for label in bar_labels])
# ax["rfe_bar"].set_ylabel("Accuracy decrease", fontsize=14)
ax["rfe_bar"].set_xlabel("Feature")
mpl_defaults.left_bottom_axes(ax["rfe_bar"])
for patch, label in zip(ax["rfe_bar"].patches, ax["rfe_bar"].get_xticklabels()):
    ax["rfe_bar"].annotate(
        label.get_text(),
        (patch.get_x() + patch.get_width() / 2, patch.get_height() + .003),
        ha="center",
        va="bottom",
        fontsize=14,
        rotation = 60
    )
ax["rfe_bar"].set_xticks('')
#----------------------------------------------------
h = delta_confusion_matrix(
    majors_model.confusion_matrix,
    xlabel="Predicted Volcano",
    xticklabels=labels,
    ylabel="True Volcano",
    yticklabels=labels[::-1],
    ax=ax["major_matrix"],
    fmt="%0.0f",
    val_limit=0,
    xtick_rot=90,
    cmap="Oranges",
    linewidth=0,
    ec="k",
    label_size = 16,
    ticklabel_size= 18
)
#----------------------------------------------------
h2 = delta_confusion_matrix(
    rfe_small_model.confusion_matrix,
    xlabel="Predicted Volcano",
    xticklabels=labels,
    ylabel="",
    yticklabels=[" " for i in range(len(labels))],
    ax=ax["rfe_matrix"],
    fmt="%0.0f",
    val_limit=0,
    xtick_rot=90,
    cmap="PuBu",
    linewidth=0,
    ec="k",
    label_size = 16,
    ticklabel_size= 18
)

axes = []
for a in ax.items():
    a[1].minorticks_off()
    axes.append(a[1])
    
mpl_defaults.label_subplots(np.array(axes)[:2], location = 'upper right',fontsize = 24)

plt.savefig('{}\RFE_subset_vs_major_summary_confusion_matrix.pdf'.format(export_path),bbox_inches = "tight")

if dark_mode is True:
    for a in axes:
        mpl_defaults.make_dark_bkgd_compatible(a, bkgd_color="none")

    plt.savefig('{}\RFE_subset_vs_major_summary_confusion_matrix_darkmode.pdf'.format(export_path),bbox_inches = "tight")

    plt.savefig('{}\RFE_subset_vs_major_summary_confusion_matrix_darkmode.png'.format(export_path),bbox_inches = "tight")



plt.show()