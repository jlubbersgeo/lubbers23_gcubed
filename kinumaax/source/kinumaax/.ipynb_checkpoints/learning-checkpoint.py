# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 13:32:44 2021

@author: jlubbers
"""

#%% machine learning model class wrapped around scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold,cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.inspection import permutation_importance

from sklearn.feature_selection import RFE, RFECV
import time

from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS


class tephraML(BaseEstimator, ClassifierMixin):
    """
    build a machine learning pipeline for volcano classification using tephra
    geochemistry data. Inherits BaseEstimator and ClassifierMixin base classes
    from scikit-learn.
    
    This will store the data, performance metrics, and predictions in an object
    to be accessed as attributes and functions can be called as methods. An 
    example pipeline is as follows:
        
        rs = 0
        # instantiate the model object, algorithm type and algorithm kwargs
        model = kgl.tephraML(model_name="extra trees",
                            n_estimators=1000,
                            max_depth=9,
                            random_state=rs
                            )
        
        # assign model.data attribute
        model.get_data(model_data)
        
        # assign model.training_data and model.test_data attributes
        model.split_data(test_size=0.3, random_state=rs)
        
        # list of columns in data to be used as features
        myfeatures = major_elements + trace_elements
        
        # assign model.training_feature_data, model.training_target_data,
        # model.test_feature_data, model.test_target_data attributes
        model.get_train_test_data(feature_cols=myfeatures,
                                  target_col="Volcano"
                                  )
        
        # scale data using StandardScalar
        model.scale_data()
        
        # train the model using model.training_feature_data
        model.train_model()
        
        # predict classifications using model.test_feature_data
        model.predict()
        
        # assign model.feature_importance_ attribute describing relative 
        # importance of features on classification
        model.get_feature_importance()
        
        # assign model.confusion_matrix attribute that can be visualized using
        # model.plot_confusion_matrix()
        model.make_confusion_matrix(normalize="true")
        
        # assign model.prediction_probability attribute which is a DataFrame 
        # of prediction probabilities for each class for a given observation.
        # the highest probability is the predicted class
        model.get_prediction_probability()
        
        # assign model.cross_val_score attribute which is stores the result of 
        # cross-validation in a numpy array
        model.get_cross_val_score(stratified = True,
                                  n_splits=10,
                                  shuffle=True,
                                  random_state=rs
                                  )

    """

    def __init__(self, name):
        self.name = name
        
    def instantiate(self,model_type,**kwargs):
        """
        instantiates the learning model with the desired algorithm

        Parameters
        ----------
        model_type : string
            "decision_tree", "random_forest", "extra_trees", "gradient_boosted_trees"
        **kwargs : keyword arguments
            keyword arguments inherited from the respective model:
                eg "decision_tree" will take kwargs from DecisionTreeClassifier
                in scikit-learn

        Returns
        -------

        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.data = []
        if self.model_type == "decision_tree":
            self.model = DecisionTreeClassifier(**kwargs)
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(**kwargs)
        elif self.model_type == "extra_trees":
            self.model = ExtraTreesClassifier(**kwargs)
        elif self.model_type == "gradient_boosted_trees":
            self.model = GradientBoostingClassifier(**kwargs)
    def get_data(self, df):
        """
        

        Parameters
        ----------
        df : pandas DataFrame
            The data to be used in the machine learning model. Only target
            and feature columns should be in here e.g.:
                
                target_col|feature1_col|feature2_col|...|featuren_col
                ----------|------------|------------|---|------------

        Returns
        -------
        self.data
        
        

        """

        self.data = df
        
    
    def split_data(self, **kwargs):
        """
        

        Parameters
        ----------
        **kwargs :
            keyword arguments from sklearn.model_selection.train_test_split

        Raises
        ------
        Exception
            If self.data is not assigned, prompts the user to assign it

        Returns
        -------
        self.training_data
        self.test_data

        """

        if len(self.data) > 0:
            self.training_data, self.test_data = train_test_split(self.data, **kwargs)
                

        else:
            raise Exception(
                "You have not established data for your model yet. Please call the model.get_data(your_data_here) function"
            )
    
    def scale_data(self):
        """
        

        Returns
        -------
        rescaled self.training_feature_data and self.test_feature_data
        

        """

        # scale the data with standard scalar
        scaler = StandardScaler().fit(self.training_feature_data)
        self.scaler = scaler
        self.training_feature_data = scaler.transform(self.training_feature_data)

        # scale test dataset
        self.test_feature_data = scaler.transform(self.test_feature_data)
        
        # generate scaler to be used with unknown data

    def get_train_test_data(self, feature_cols, target_col):
        """
        

        Parameters
        ----------
        feature_cols : list
            list of column headers to be used as features in the model
        target_col : string
            column header to be used as target in the model

        Returns
        -------
        self.training_feature_data
        self.training_target_data
        self.test_feature_data
        self.test_target_data

        """

        self.training_feature_data = self.training_data.loc[:, feature_cols]
        self.training_target_data = self.training_data.loc[:, target_col]

        self.test_feature_data = self.test_data.loc[:, feature_cols]
        self.test_target_data = self.test_data.loc[:, target_col]

        self.features = self.training_feature_data.columns.tolist()
        self.target = self.training_target_data.name

    
    def train_model(self):
        """
        

        Returns
        -------
        self.model that has attributes inherited from sklearn e.g.
        self.model.fit, self.model.score

        """

        self.model.fit(self.training_feature_data, self.training_target_data)

        print(
            "Accuracy on training set: {:.3f}".format(
                self.model.score(self.training_feature_data, self.training_target_data)
            )
        )
        print(
            "Accuracy on test set: {:.3f}".format(
                self.model.score(self.test_feature_data, self.test_target_data)
            )
        )

    def predict(self):
        """
        
        applies self.model.predict(self.test_feature_data) to generate predictions
        for each observation in self.test_feature_data

        Returns
        -------
        self.predicted_class 

        """

        self.predicted_class = self.model.predict(self.test_feature_data)
        print(
            "Accuracy of prediction: {:.3f}".format(
                metrics.accuracy_score(self.test_target_data, self.predicted_class)
            )
        )

    def make_confusion_matrix(self, **kwargs):
        """
        generate a confusion matrix for the predicted class and target data

        Parameters
        ----------
        **kwargs : 
            keyword arguments for sklearn.metrics.confustion_matrix

        Returns
        -------
        self.confusion_matrix

        """

        self.confusion_matrix = confusion_matrix(
            self.test_target_data,
            self.predicted_class,
            labels=list(np.unique(self.test_target_data)),
            **kwargs,
        )

    def plot_confusion_matrix(self,ax = None, **kwargs ):
        """
        plot a confusion matrix using sklearn.metrics.ConfusionMatrixDisplay

        Parameters
        ----------
        **kwargs : 
            keyword arguments for sklearn.metrix.ConfusionMatrixDisplay

        Returns
        -------
        None.

        """
        if ax is None:
            
            ax = plt.gca()

        ConfusionMatrixDisplay(
            self.confusion_matrix, display_labels=list(np.unique(self.test_target_data))
        ).plot(ax=ax, **kwargs)
        
        
        
    def plot_feature_importance(self,show_error = False,sorted = False, ax = None,label_rotation = 90,**kwargs):
        """
        plots a bar chart of feature importances for random forest and random
        trees ensemble classifiers
        

        Parameters
        ----------
        show_error : Boolean
        Whether or not to show errorbars on the bar chart. Error bar kwargs 
        can be customized using aictionary of kwargs inherited from the 
        plt.errorbar method. Values of ecolor or capsize defined here take 
        precedence over the independent kwargs.
        
        sorted : Boolean
        Whether or not to sort the feature importance plot by decreasing importance.
        Default is False
        
        ax : matplotlib axes object (optional)
        The object to map the plot to
        

        **kwargs : matplotlib.pyplot.bar kwargs
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html

        Returns
        -------
        None

        """
        if ax is None:

            ax = plt.gca()

        df = pd.DataFrame(
            {
                "feature": self.features,
                "importance": self.feature_importance_,
                "uncertainty": self.feature_importance_std_,
            }
        )

        if sorted is True:

            df = df.sort_values(by="importance", ascending=False)

        if show_error is True:

            ax.bar(df["feature"], df["importance"], yerr=df["uncertainty"], **kwargs)
            ax.set_xticks(df["feature"])
            ax.set_xticklabels(df["feature"], rotation=label_rotation)
            
        else:
            ax.bar(df["feature"], df["importance"], **kwargs)
            ax.set_xticks(df["feature"])
            ax.set_xticklabels(df["feature"], rotation=label_rotation)
            

    def get_feature_importance(self, method="impurity", **kwargs):

        """
        determines the relative importance of each feature in a forest of trees
        model
        

        Parameters
        ----------
        method : string
        The scoring method for how feature importance is determined. Options are
        "impurity" or "permutation". Default is "impurity". For more information
        see: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html


        Returns
        -------
        self.feature_importances_ which is a pandas Series of relative feature
        importances for each feature

        """
        if method == "impurity":

            self.feature_importance_ = pd.Series(
                self.model.feature_importances_, index=self.features
            )

            std = np.std(
                [tree.feature_importances_ for tree in self.model.estimators_], axis=0
            )
            self.feature_importance_std_ = std

        elif method == "permutation":

            result = permutation_importance(
                self.model, self.test_feature_data, self.test_target_data, **kwargs
            )

            self.feature_importance_ = pd.Series(
                result.importances_mean, index=self.features
            )
            self.feature_importance_std_ = pd.Series(
                result.importances_std, index=self.features
            )

    def get_prediction_probability(self):
        """
        

        Returns
        -------
        self.prediction_probabilities which is a pandas DataFrame of prediction
        probabilities for each class for a given observation. The class with 
        the highest probability is the predicted class

        """

        self.prediction_probability = self.model.predict_proba(self.test_feature_data)
        self.prediction_probability = pd.DataFrame(
            self.prediction_probability, columns=self.model.classes_
        )

    def get_cross_val_score(self,stratified = True, **kwargs):
        """
        

        Parameters
        ----------
        stratified : boolean, optional
            whether to compute stratified kfold cross validation. The default
            is True.
        **kwargs : 
            keyword arguments for sklearn.model_selection import KFold, StratifiedKFold

        Returns
        -------
        self.cross_val_score which is a numpy array of training dataset 
        accuracies

        """
        
        if stratified:
            
            kfold = StratifiedKFold(**kwargs)
        
        else:
            
            kfold = KFold(**kwargs)
            
            
        self.cross_val_score = cross_val_score(
            self.model, self.training_feature_data, self.training_target_data, cv=kfold
        )
        print(
            "Cross-validation mean: {} | std: {}".format(
                np.round(self.cross_val_score.mean(), 2),
                np.round(self.cross_val_score.std(), 2),
            )
        )
        
        
    def reduce_dimensions(self, kind, **kwargs):
        """
        Apply principal component analysis on data to reduce dimensionality of
        dataset. 
        
        This should be done in between scaling the dataset and training the 
        ML model:
            rs = 0
            model = tf.tephraML(model_name="extra trees",
                                n_estimators=1000,
                                max_depth=9,
                                random_state=rs
                                )
            model.get_data(model_data)
            model.split_data(test_size=0.3, random_state=rs)
            myfeatures = major_elements + trace_elements
            model.get_train_test_data(feature_cols=myfeatures, target_col="Volcano")
            model.scale_data()
            -----
            #PCA
            model.reduce_dimensions(kind = 'linearPCA',n_components = None)
            #get data from just the first 10 components
            n = 10
            model.training_feature_data = model.training_feature_data[:,:n]
            model.test_feature_data = model.test_feature_data[:,:n]
            model.features = np.arange(model.training_feature_data.shape[1])
            -----
            model.train_model()
            model.predict()

        Parameters
        ----------
        kind : string
            the kind of dimension reduction to compute. Options are
            "linearPCA", "kernelPCA". 
        **kwargs :
            keyword arguments for either sklearn.decomposition.PCA or
            sklearn.decomposition.KernelPCA
            
            
        Raises
        ------
        Exception
           if neither "linearPCA" or "kernelPCA" is chosen for kind, throw
           an error prompting the user to do so

        Returns
        -------
        Changes the training and test feature data to now reflect principal
        component units rather than original units
        self.training_feature_data
        self.test_feature_data
        
        The dimension reduction model
        self.pca_
        
        renamed features now as principal components (e.g. 1, 2, 3...n)
        self.features

        """
        
        if kind == 'linearPCA':
            
            pca = PCA(**kwargs)
            self.training_feature_data = pca.fit_transform(self.training_feature_data)
            self.test_feature_data = pca.transform(self.test_feature_data)
            self.pca_ = pca
            self.features = np.arange(self.training_feature_data.shape[1])

            
            
        elif kind == 'kernelPCA':
            
            pca = KernelPCA(**kwargs)
            self.training_feature_data = pca.fit_transform(self.training_feature_data)
            self.test_feature_data = pca.transform(self.test_feature_data)
            self.pca_ = pca
            self.features = np.arange(self.training_feature_data.shape[1])


        
        else:
            raise Exception(
                "Please choose either 'linear' or 'kernel' to complete dimensionality reduction via PCA"
            )
            
            
    def compute_rfe(self,cross_validate = True, **kwargs):
        """
        complete recurring feature elimination for the model:
            https://scikit-learn.org/stable/modules/feature_selection.html#rfe

        Parameters
        ----------
        cross_validate : boolean
            whether or not to do cross validation with RFE if False, uses 
            sklearn.feature_selection.RFE. If True, uses sklearn.feature_selection.RFECV.
            The default is True.
        **kwargs : key word arguments
            key word arguments for either RFE or RFECV in sklearn.feature_selection

        Returns
        -------
        self.rfe_results, a dataframe with columns for features, rank, importance
        where rank = 1 are features chosen by the rfe computation. 

        """
        t0 = time.time()

        if cross_validate is True:
            print('completing RFE with cross validation, be patient!')
            rfe = RFECV(estimator = self.model, **kwargs)
            
        else:
            print('completing RFE, be patient!')
            rfe = RFE(estimator = self.model,**kwargs)
            
        rfe = rfe.fit(self.training_feature_data, self.training_target_data)
        
        feature_data = []
        for feature, importance, ranking in zip(
            self.features, self.feature_importance_, rfe.ranking_
        ):
            feature_data.append([ranking, importance, feature])
        
        feature_data = np.array(feature_data)
        feature_data = pd.DataFrame(feature_data, columns=["rank", "importance", "feature"])
        feature_data["rank"] = pd.to_numeric(feature_data["rank"])
        feature_data["importance"] = pd.to_numeric(feature_data["importance"])
        feature_data = feature_data.sort_values(by="importance", ascending=False)
        
        self.rfe_results = feature_data
        t1 = time.time()

        print("RFE completed in {}s".format(t1 - t0))
        
        
    def get_proximities(self, rs = None):
        """
        generate a proximity matrix for all observations in training dataset
        and their respective [x,y] coordinates to be plotted in 2D space
        Example from: 
            https://github.com/glouppe/phd-thesis/blob/master/scripts/ch4_proximity.py

        Parameters
        ----------
        rs : int, optional
            option to fix the random state. The default is None.

        Returns
        -------
        self.proximity_matrix
            an n x n matrix that where n is the number of observations in the 
            training dataset. values represent pairwise euclidian distances from 
            other observations. Diagonal values will be 0. 
            
        self.proximity_coordinates
            pandas Dataframe with columns for target class, x coordinate, and
            y coordinate of the proximity matrix converted to 2D space after 
            multi dimensional scaling. The further coordinates are from one 
            another, the more dissimilar they are in the random forest model

        """
        
        
        # making a pairwise distance matrix for proximities between all observations
        # in training set
        prox = pdist(self.model.apply(self.training_feature_data), lambda u, v: (u == v).sum()) / self.model.n_estimators
        prox = squareform(prox)
        
        
        # convert to 2D 
        mds = MDS(dissimilarity="precomputed", n_jobs=-1,random_state = rs)
        
        # [x,y] coordinates for proximities
        coords = mds.fit_transform(1. - prox)
        
        #make it a dataframe for easy plotting later
        df = pd.DataFrame(coords, columns = ['x','y'])
        df.insert(0,'target_class',self.training_target_data.to_numpy())
        
        # create proximity_matrix attribute
        self.proximity_matrix = prox
        
        # create proximity_coordinates attribute for plotting
        self.proximity_coordinates = df
        
        
    def plot_proximities(self, cmap, ax = None, **plt_kwargs):
        """
        Plot the proximity coordinates. Example from:
            https://github.com/glouppe/phd-thesis/blob/master/scripts/ch4_proximity.py

        Parameters
        ----------
        cmap : string
            any valid matplotlib colormap. This will be mapped to discrete
            colors based on the number of target classes
        ax : matplotlib axis object, optional
            The axis to map the plot to. The default is None.
        **plt_kwargs : matplotlib keyword arguments
            any valid matplotlib.pyplot.plot() keyword argument

        Returns
        -------
        Example usage:
            
            fig, ax = plt.subplots(figsize = (8,8))
            model.plot_proximities(cmap = 'tab20',
                                   ax = ax,
                                   marker = 'o',
                                   mec = 'k',
                                   mew = .5
                                   )
            legend = fig.legend(loc = 'upper right', ncol = 1)
            legend.set_title('Target Class', prop={'size': 12})
            ax.set_xticks([-1,0,1])
            ax.set_yticks([-1,0,1])

        """
        
        if ax is None:
            ax = plt.gca()
        
        n_classes = self.model.n_classes_
        
        cm = plt.get_cmap(cmap)
        
        colors = (cm(1. * i / n_classes) for i in range(n_classes))
        
        df = self.proximity_coordinates.set_index('target_class')
        
        for k, c in zip(df.index.unique(), colors):
                        
            ax.plot(df.loc[k,'x'],df.loc[k,'y'],ls = '',color = c, label = k,**plt_kwargs)
            
        
        

        
        
        
        
        
        
