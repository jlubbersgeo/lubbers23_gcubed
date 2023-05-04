# Probabilistic source classification of large tephra producing eruptions using supervised machine learning: An example from the Alaska-Aleutian arc

Jordan Lubbers¹, Matthew Loewen¹, Kristi Wallace¹, Michelle Coombs¹, Jason Addison² <br>
¹U.S. Geological Survey Alaska Volcano Observatory<br>
²U.S. Geological Survey Minerals Energy and Geophysics Science Center<br>

Contact: jlubbers@usgs.gov

This repository is home to the jupyter notebooks and python scripts that recreate the figures of the above manuscript. Below is a brief explanation of each notebook.

## Explanation of code notebooks

1. `build_training_data.py`: this takes the `major_trace_proximal_train_data` sheet in the supplementary data and builds the training dataset for subsequent use in machine learning applications. To do this, we will largely be using functions from the `kinumaax.crunching` module to clean the data and the `pyrolite` package to transform the data. It also cleans and transforms data in the `iodp_trace_data`, and `derkachev_data` sheets such that the eventual machine learning model can be applied to them.
2. `bivariate_plots.py`: This notebook is devoted to creating Figure 2.

3. `model_feature_selcetion.py`: This notebook explores the feature engineering process whereby a basic random forest model is trained using a variety of features such that the results from each feature space can be compared, ultimately producing Figures 3, A3, A4.

4. `gradient_boosting_tuning.py`: Using a grid search to find the optimal parameters for the gradient boosting classifier used in the final ensemble voting classifier. Produces Figure A7B

5. `KNN_tuning.py`: Using a grid search to find the optimal parameters for the K-nearest neighbors classifier used in the final ensemble voting classifier. Produces Figure A6A.

6. `LDA_tuning.py`: Using a grid search to find the optimal parameters for the Linear Discriminant Analysis classifier used in the final ensemble voting classifier. Produces Figure A5A.

7. `logistic_regression_tuning.py`: Using a grid search to find the optimal parameters for the Logistic Regression classifier used in the final ensemble voting classifier. Produces Figure A5B.

8. `RF_tuning.py`: Using a grid search to find the optimal parameters for the Random Forest classifier used in the final ensemble voting classifier. Produces Figure A7A.

9. `SVM_tuning.py`: Using a grid search to find the optimal parameters for the Support Vector Machine classifier used in the final ensemble voting classifier. Produces Figure A6B.

10. `final_voting_classifier.py`: Builds thet final ensemble voting classifier using the optimal hyperparameters for each individual algorithm. The influence of each individual aglorithm is determined by its F₁ score. Produces Figures 5 and A8.

11. `rf_proximities.py`: Uses an out of the box random forest classifier and the optimal features found in `model_feature_selection.py` to compare class similarities as viewed by a random forest algorithm. Produces Figure 4.

12. `variance_analysis.py`: completes the variance test as described the section "Assessing model variance" where we test how likely we would be able to correctly predict thte volcanic source for an eruption if the model was not trained on that eruption.

13. `variance_analysis_plots.py`: Visualizing the results of the code run in `variance_analysis_plots.py`. Produces Figures 6, 7, 8.

14. `IODP_novelty_detection.py`: Makes plots in multivariate space to compare where our training data plots with respect to IODP data to see if they overlap. Produces Figures 9, A9, A10.

15. `IODP_predictions.py`: Uses the final model developed in `final_voting_classifier.py` to predict the most likely volcanic source for each unknown sample listed in Table A4. Produces Figures 10, 11, 12.

16. `derkachev_predictions.py`: Uses the final model developed in `final_voting_classifier.py` to predict the most likely volcanic source for samples Br2 and SR2 from Derkachev et al., (2018). Produces Figures 13, 14.

17. `EPMA_stds_check.py`: Quantifies the accuracy and precision of EPMA secondary standards analyzed in the study. Produces values in Table A2 and Figure A1.

18. `lasercalc_standards_only.py`: Determines concentrations of LA-ICP-MS secondary standards analyzed in this study. Input data are secondary standards data normalized to ²⁹Si using LaserTRAM-DB. Produces values in Table A3 and Figure A2.

19. `aleutian_colors.py`: creates a dictionary of `volcano_name : {**kwargs}` pairs for each volcano in the dataset such that it can be consistently plotted in all figures.

20. `mpl_defaults.py`: establishes default `matplotlib` settings and helper functions to produce figures for the manuscript.

21. `kinumaax`: A small collection of python scripts to help working with tephra data

    1. `crunching.py`: has functions used for data cleaning, filtering, unit conversions, etc. Largely used by `build_training_data.py`

    2. `learning.py`: builds a machine learning pipeline for volcano classification using tephra geochemistry data. Inherits `BaseEstimator` and `ClassifierMixin` base classes from `scikit-learn`. This will store the data, performance metrics, and predictions in an object to be accessed as attributes and functions can be called as methods. Used largely in `model_feature_selection.py` to efficiently create numerous Random Forest classifier models.
    3. `visualization.py`: various functions for producing plots molre specific to volcanology/petrology. Not used in this manuscript.
