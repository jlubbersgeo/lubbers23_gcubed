# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 13:34:38 2021

@author: jlubbers
"""
import numpy as np
import pandas as pd
import warnings
from mendeleev import element as e

from scipy import stats

import re

#%% Data cleaning functions

def remove_strings(df,ignore_cols = None):
    """
    removes all strings from a dataframe

    Parameters
    ----------
    df : pandas DataFrame
        the input dataframe. Careful, this will remove ALL strings from 
        a dataframe, including columns that are solely comprised of strings.
        All means All.
        
    ignore_cols : list
    list of column headers to ignore when filtering for strings. If strings 
    are in these columns, they will remain.

    Returns
    -------
    df : pandas DataFrame
        dataframe same shape as input dataframe but with strings replaced by 
        NaN values. 

    """
    
    dff = df.copy()
    columns_to_check = dff.columns.tolist()
    if ignore_cols is not None:
        
        for col in ignore_cols:
            columns_to_check.remove(col)
            
            
    strings_to_remove = []
    for c in columns_to_check:
        for r in dff[c]:
            if type(r) == str:
                strings_to_remove.append(r)
                                


    # get a list of unique values...no repeats!
    strings_to_remove = list(np.unique(np.array(strings_to_remove)))

    # create a dictionary with {string : np.nan}
    # to utilize the df.replace() function
    dict_to_remove = {i: np.nan for i in strings_to_remove}

    # get rid of them strings!
    for c in dff.loc[:,columns_to_check]:
        dff.replace({c: dict_to_remove}, inplace=True)
        
    return dff



def filter_by_quantile(df,col_name,quantile,operator,nan = 'drop'):
    """
    

    Parameters
    ----------
    df : pandas dataframe
        the original data to be filtered
    col_name : string
        the name of the column in your dataframe that will be used to filter 
        the rest of the data with
    quantile : float
        quantile of the data between 0 and 1. E.g. 0.5 = 50% quantile
    operator : string
    "less_than", "greater_than", "equal". Whether or not to filter by less than,
    greater than, or equal to the specified quantile. 
    

    Returns
    -------
    filtered_df : pandas dataframe
        dataframe with the same number of columns as the input dataframe, however
        now it has been filtered by the specified column and the only rows
        that remain are rows that meet the criteria in the 

    """
    if nan == "drop":
        
        df = df.dropna()
    
    if nan != "drop":
        
        df = df.fillna(value = nan)
    
    if operator == 'less_than':
        
        filtered_df = df.loc[df[col_name] <= df[col_name].quantile(quantile)]
        
    elif operator == 'greater_than':
        
        filtered_df = df.loc[df[col_name] >= df[col_name].quantile(quantile)]
        
    elif operator == 'equals':
        
        filtered_df = df.loc[df[col_name] == df[col_name].quantile(quantile)]
        
    else:
        
        raise ValueError("please choose an operator that is either 'less_than', 'greater_than', or 'equal'.")
        


        
    
    
    return filtered_df


def filter_by_value(df,col_name,value,operator,nan = np.NaN):
    """
    

    Parameters
    ----------
    df : pandas dataframe
        the original data to be filtered
    col_name : string
        the name of the column in your dataframe that will be used to filter 
        the rest of the data with
    value : float
        the discriminating value to be used in filtering
    operator : string
    "less_than", "less_than_equal", "greater_than", "greater_than_equal", "equal". Whether or not to filter by less than,
    greater than, or equal to the specified quantile. 
    

    Returns
    -------
    filtered_df : pandas dataframe
        dataframe with the same number of columns as the input dataframe, however
        now it has been filtered by the specified column and the only rows
        that remain are rows that meet the criteria in the 

    """
    if nan == "drop":
        
        df = df.dropna()
    
    if nan != "drop":
        
        df = df.fillna(value = nan)
    
    if operator == 'less_than':
        
        filtered_df = df.loc[df[col_name] < value]
        
    elif operator == 'greater_than':
        
        filtered_df = df.loc[df[col_name] > value]
        
    elif operator == 'less_than_equal':
        
        filtered_df = df.loc[df[col_name] <= value]
        
    elif operator == 'greater_than_equal':
        
        filtered_df = df.loc[df[col_name] >= value]
        
    elif operator == 'equals':
        
        filtered_df = df.loc[df[col_name] == value]
        
        
        
    else:
        
        raise ValueError("please choose an operator that is either 'less_than', 'greater_than', or 'equal'.")
        
    
    return filtered_df

def oxide_to_ppm(wt_percent,oxide):
    """
    convert concentration oxide in weight percent to concentration ppm for 
    a 1D series of data

    Parameters
    ----------
    wt_percent : array-like
        the oxide values to be converted to ppm
    oxide : string
        the oxide that is being converted (e.g., 'SiO2','Al2O3','TiO2')

    Returns
    -------
    ppm : array-like
        concentrations in ppm the same shape as the wt_percent input

    """
    
    
    
    s = oxide.split('O')
    el = [i for i in s[0] if not i.isdigit()]

    if len(el) == 2:

        element = el[0] + el[1]

    else:
        element = el[0]


    cat_subscript = s[0]
    an_subscript = s[1]

    cat_subscript = [i for i in cat_subscript if i.isdigit()]
    if cat_subscript:
        cat_subscript = int(cat_subscript[0])
    else:
        cat_subscript = 1

    an_subscript = [i for i in an_subscript if i.isdigit()]
    if an_subscript:
        an_subscript = int(an_subscript[0])
    else:
        an_subscript = 1



    ppm = 1e4 * ((wt_percent * e(element).atomic_weight * cat_subscript) / ( e(element).atomic_weight + e('O').atomic_weight*an_subscript))

    
    return ppm

def df_oxide_to_ppm(data, oxides):
    """
    convert concentration oxide in weight percent to concentration ppm for a 
    2D pandas dataframe

    Parameters
    ----------
    data : pandas DataFrame
        the dataframe containing wt % oxide values to be converted
    oxides : list
        list of oxides to be converted into ppm

    Returns
    -------
    pandas DataFrame
        dataframe with columns added for specified oxides converted to ppm.
        added columns will be named as follows "element_ppm", e.g., 
        SiO2 --> Si_ppm
        

    """
    df = data.copy()
    ppm_data = np.zeros(df.loc[:, oxides].shape)

    for element in enumerate(oxides):
        ppm_data[:, element[0]] = oxide_to_ppm(df[element[1]], element[1])

    ppm_cols = [element.split("O")[0] for element in oxides]
    elements = []
    
    for col in ppm_cols:
        el = [i for i in col if not i.isdigit()]
        if len(el) == 1:
            elements.append(el[0])
        else:
            elements.append(el[0] + el[1])

    ppm_data = pd.DataFrame(
        ppm_data,
        columns=[oxide + "_ppm" for oxide in oxides],
        index=data.index,
    )
    
    for column in ppm_data.columns.tolist():
        df[column] = ppm_data[column].copy()
        
    return df


def normalize_anhydrous(data,major_elements):
    """
    adds columns for major elements normalized to 100% without the following 
    volatiles: ['H2O', 'P2O5', 'Cl', 'F']
    

    Parameters
    ----------
    data : pandas pandas DataFrame
        dataframe where the major element data to be normalized lives
        
    major_elements : list
        list of all columns with major element data 
        e.g. ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 
              'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']

    Returns
    -------
    pandas DataFrame
        dataframe with columns added for major elements normalized to 100% 
        without volatiles. added columns will have the suffix "_norm_dry", 
        e.g., SiO2 --> SiO2_norm_dry

    """
    
    major_data = data.loc[:,major_elements]
    volatiles = ['H2O', 'P2O5', 'Cl', 'F']

    for col in major_data.columns:

        if col in volatiles:
            major_data.drop(col,axis = 'columns',inplace = True)

    major_data['Total'] = major_data.sum(axis = 'columns')
    major_data_norm = major_data.divide(major_data['Total'],axis = 'rows')*100
    major_data_norm.columns = ['{}_norm_dry'.format(col) for col in major_data_norm.columns]
    
    return pd.concat([data,major_data_norm],axis = 'columns')


def normalize_hydrous(data,major_elements):
    """
    adds columns for major elements normalized to 100% 
    

    Parameters
    ----------
    data : pandas pandas DataFrame
        dataframe where the major element data to be normalized lives
        
    major_elements : list
        list of all columns with major element data 
        e.g. ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 
              'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']

    Returns
    -------
    pandas DataFrame
        dataframe with columns added for major elements normalized to 100%.
        added columns will have the suffix "_norm_wet", e.g., 
        SiO2 --> SiO2_norm_wet

    """
    
    major_data = data.loc[:,major_elements]
    major_data['Total'] = major_data.sum(axis = 'columns')
    major_data_norm = major_data.divide(major_data['Total'],axis = 'rows')*100
    major_data_norm.columns = ['{}_norm_wet'.format(col) for col in major_data_norm.columns]
    
    return pd.concat([data,major_data_norm],axis = 'columns')
        
def df_checkna(df, cols):
    """
    checks if certain columns in a pandas dataframe contain NaN values. If they
    do, they get added to a list.

    Parameters
    ----------
    df : pandas DataFrame
        the dataframe that will be checked for NaN values
    cols : list or string
        list of column names to check for NaNs. Also accepts single string.

    Returns
    -------
    nan_cols : list
        list of columns in dataframe that have NaN values.

    """
    nan_cols = []
    nan_cols_count = []
    # accept single column input by converting 
    # to a list
    if type(cols) == str:
        cols = [cols]
        
        
    for col in cols:
        if df[col].isnull().sum() > 0:
            nan_cols.append(col)
            nan_cols_count.append(df[col].isnull().sum())
            # print('column: {} - # of nan rows: {}'.format(col,df[col].isnull().sum()))
    if len(nan_cols) < 1:
        warnings.warn("none of the specified columns have NaNs. Please choose different columns ")
    
    return nan_cols,np.array(nan_cols_count)



def df_dropna(df, cols):
    """
    drop all rows in a pandas dataframe that has NaN values in a user specified
    list of columns

    Parameters
    ----------
    df : pandas DataFrame
        the dataframe that will have the rows removed
    cols : list or string
        list of column names to filter from. Also accepts single string

    Returns
    -------
    df : pandas DataFrame
        cleaned dataframe that has rows with NaN values in specified columns 
        removed

    """
    
    # accept single column input by converting
    # to list
    if type(cols) == str:
        cols = [cols]
        
    for col in cols:
        df = df[df[col].notna()]
        
    return df

def df_fillna(df, cols, val):
    """
    drop all rows in a pandas dataframe that has NaN values in a user specified
    list of columns

    Parameters
    ----------
    df : pandas DataFrame
        the dataframe that will have the rows removed
    cols : list or string
        list of column names to filter from. Also accepts single string

    Returns
    -------
    df : pandas DataFrame
        cleaned dataframe that has rows with NaN values in specified columns 
        removed

    """
    
    # accept single column input by converting
    # to list
    if type(cols) == str:
        cols = [cols]
        
    for col in cols:
        df[col] = df[col].fillna(value = val)
        
    return df

def create_ratio_val(data,num,den,num_std = None, den_std = None,uncertainty = False):
    """
    adds a column to an existing dataframe that represents the ratio of 
    two columns. Will also add a column for the uncertainty of the ratio
    if specified

    Parameters
    ----------
    data : pandas dataframe
        the dataframe from which the columns are taken
    num : string
        name of column that represents the numerator
    den : string
        name of column that represents the denominator
    num_std :string, optional
        name of the column that represents the uncertainty of the numerator. 
        The default is None.
    den_std : string, optional
        name of the column that represents the uncertainty of the denominator.
        The default is None.
    uncertainty : boolean, optional
        Whether or not to create a column that represents the uncertainty of 
        the ratio. If True, columns must be specified for "num_std" and "den_std".
        The default is False.

    Returns
    -------
    df : pandas dataframe
        a copy of the input dataframe with columns for the ratio and, if specified, 
        its uncertainty.
        

    """
    df = data.copy()
    df[f"{num}/{den}"] = df[num] / df[den]

    if uncertainty is True:
        df[f"{num}/{den}_std"] = df[f"{num}/{den}"] * np.sqrt((df[f"{num_std}"] / df[f"{num}"]) ** 2 + (df[f"{den_std}"] / df[f"{den}"]) ** 2)
    return df

#%% descriptive stats on data

def normal_test(x, kind):
    """
    compute normality test from a range of tests

    Parameters
    ----------
    x : array-like
        The data to test normality for  
    kind : string
        which test to use. options are:
            'anderson': https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html
            'shapiro': https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
            'dagostino': https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
            

    Returns
    -------
    results : results of the test in the form of [statistic, p value]. If kind is 'anderson'
              in the form [statistic, critical value, significance level]
       

    """
    
    if kind  == 'anderson':
        #if stat > crit val reject normality at corresponding
        #significance level. Significance values are in %
        
        results = stats.anderson(x,dist = 'norm')
                
    elif kind == 'shapiro':
        #null is normal. p < alpha reject normal
        
        results = stats.shapiro(x)
    
    elif kind == 'dagostino':
        #null is normal. p < alpha reject normal
        
        results = stats.normaltest(x)
        
        
    return results

#%% Mixing model related functions
def mixing(c1, c2, f):
    """
    mixing creates a mixing model between two endmembers

    Inputs:
        c1 = concentration of endmember 1
        c2 = concentration of endmember 2
        f = fraction of endmember 1 in the model

    Returns:
        cm = concnetration of the mixture
    """
    cm = c1 * f + c2 * (1 - f)
    return cm


# isotopic mixing model.
# fix this to add in a conditional to choose either one or two ratios
def isomix(rationum, c1, r1, c2, r2, *data):
    """
    isomix uses equations 18.24-18.26 from Faure 1998 to calculate isotopic mixing
    compositions for a given isotopic pair

    Inputs:
    rationum: use the input 'oneratio' or 'tworatios' to define how many isotopic 
            systems you are interested in
    c1 = concentration of element for endmember 1
    c2 = concentration of element for endmember 2
    r1 = isotopic ratio for endmember 1
    r2 = isotopic ratio for endmember 2
    *data = repeat the first 4 inputs for the second isotopic system of interest

    Returns:

    cm = concentrations of mixture for various values of 'f' where
    f is the fraction of endmember 1 in the mixture

    rm = isotopic ratios of mixture for various values of 'f'
    """

    # array of fractions of component 1
    f = np.linspace(0, 1, 11)
    # concentration of the mixture
    # eq. 18.19
    if rationum == "oneratio":
        cm = c1 * f + c2 * (1 - f)

        # eq. 18.25
        a = (c1 * c2 * (r2 - r1)) / (c1 - c2)
        # eq. 18.26
        b = (c1 * r1 - c2 * r2) / (c1 - c2)
        # eq. 18.24
        rm = a / cm + b

        return cm, rm

    elif rationum == "tworatios":
        cm = c1 * f + c2 * (1 - f)

        # eq. 18.25
        a = (c1 * c2 * (r2 - r1)) / (c1 - c2)
        # eq. 18.26
        b = (c1 * r1 - c2 * r2) / (c1 - c2)
        # eq. 18.24
        rm = a / cm + b

        cm2 = data[0] * f + data[2] * (1 - f)

        # eq 18.25
        c = (data[0] * data[2] * (data[3] - data[1])) / (data[0] - data[2])
        # eq 18.26
        d = (data[0] * data[1] - data[2] * data[3]) / (data[0] - data[2])
        rm2 = c / cm2 + d

        return cm, rm, cm2, rm2
    else:
        print(
            "Check your input. Ensure to specify rattionum and the correct amount of concentrations or ratios"
        )

def ratio_mixing(df, n_components, resolution=0.1):

    """
    Mixing of ratios as described by Albarede 1995
    Introduction to Geochemical Modeling equation 1.3.1
    
    Inputs:
    
    df | pandas DataFrame
    
    DataFrame of inputs. should be formatted as follows:
    
    For 2 component mixing:
    
    Index|Element1_c|Element1_r|Element2_c|Element2_r
    -------------------------------------------------
      A  |          |          |          |
    -------------------------------------------------  
      B  |          |          |          |
    
      
    
    For 3 component mixing:
    
    Index|Element1_c|Element1_r|Element2_c|Element2_r
    -------------------------------------------------
      A  |          |          |          |
    -------------------------------------------------  
      B  |          |          |          |
    -------------------------------------------------
      C  |          |          |          |
      
      
    Where the name of each component is the index of the dataframe and the 
    concentration and ratio columns for each elemental species contain "_c" and "_r" 
    somewhere in the column header, respectively. 
    
    n_components | int
    
    Number of end-member components (either 2 or 3)
    
    resolution | float
    
    The resolution you want to run your mixing model at. This is a number between 0.01 
    and 0.5. This is how far apart to space points in the eventual mixing mesh
    (e.g. .1 will return a mixing mesh spaced by 1O% increments for each component)
    
    Default is 0.1
    
    
    
    
    Returns:
    
    results | pandas DataFrame
    
    The results of the mixing model that is n x 7 in shape:
    
    f_A|f_B|f_C|Element1_c_mix|Element2_c_mix|Element1_r_mix|Element2_r_mix
    -----------------------------------------------------------------------
    
    Where f columns are fraction of each component in the mixture and other columns
    Are for the concentrations and ratios of the mixture for each respective combination
    of f values
    
    
    """

    if n_components == 2:

        if resolution < 0.01:
            print(
                "Please pick a lower resolution (e.g., bigger number).\nYou don't need it and it your computer may explode"
            )

        if resolution > 0.5:
            print("Please pick a higher resolution (e.g., number < 0.5). \n")

        else:

            # generate an array for fraction of each component
            f = np.arange(0, 1 + resolution, resolution)

            # all possible combinations for three f arrays
            a = np.array(np.meshgrid(f, f)).T.reshape(-1, 2)

            # where the combinations sum to 1
            f_vals = a[a.sum(axis=1) == 1]

            # get names of components
            components = df.index.tolist()

            # get names of columns where concentrations and ratios are held
            # IMPORTANT TO HAVE DATAFRAME IN THIS FORMAT
            elements = [col for col in df.columns if "_c" in col]
            ratios = [col for col in df.columns if "_r" in col]

            # Concentration of mixture

            if len(elements) == 1:

                el1_mix_concentrations = (
                    df.loc[components[0], elements[0]] * f_vals[:, 0]
                    + df.loc[components[1], elements[0]] * f_vals[:, 1]
                )

                # ratio values of the mixture using Albarede 1995 eq. 1.3.1
                el1_mix_ratios = df.loc[components[0], ratios[0]] * (
                    (f_vals[:, 0] * df.loc[components[0], elements[0]])
                    / el1_mix_concentrations
                ) + df.loc[components[1], ratios[0]] * (
                    (f_vals[:, 1] * df.loc[components[1], elements[0]])
                    / el1_mix_concentrations
                )

                results = pd.DataFrame(
                    {
                        "f_{}".format(components[0]): f_vals[:, 0],
                        "f_{}".format(components[1]): f_vals[:, 1],
                        "{}_mix".format(elements[0]): el1_mix_concentrations,
                        "{}_mix".format(ratios[0]): el1_mix_ratios,
                    }
                )
            else:

                el1_mix_concentrations = (
                    df.loc[components[0], elements[0]] * f_vals[:, 0]
                    + df.loc[components[1], elements[0]] * f_vals[:, 1]
                )
                el2_mix_concentrations = (
                    df.loc[components[0], elements[1]] * f_vals[:, 0]
                    + df.loc[components[1], elements[1]] * f_vals[:, 1]
                )

                # ratio values of the mixture using Albarede 1995 eq. 1.3.1
                el1_mix_ratios = df.loc[components[0], ratios[0]] * (
                    (f_vals[:, 0] * df.loc[components[0], elements[0]])
                    / el1_mix_concentrations
                ) + df.loc[components[1], ratios[0]] * (
                    (f_vals[:, 1] * df.loc[components[1], elements[0]])
                    / el1_mix_concentrations
                )

                el2_mix_ratios = df.loc[components[0], ratios[1]] * (
                    (f_vals[:, 0] * df.loc[components[0], elements[1]])
                    / el2_mix_concentrations
                ) + df.loc[components[1], ratios[1]] * (
                    (f_vals[:, 1] * df.loc[components[1], elements[1]])
                    / el2_mix_concentrations
                )

                results = pd.DataFrame(
                    {
                        "f_{}".format(components[0]): f_vals[:, 0],
                        "f_{}".format(components[1]): f_vals[:, 1],
                        "{}_mix".format(elements[0]): el1_mix_concentrations,
                        "{}_mix".format(elements[1]): el2_mix_concentrations,
                        "{}_mix".format(ratios[0]): el1_mix_ratios,
                        "{}_mix".format(ratios[1]): el2_mix_ratios,
                    }
                )

    if n_components == 3:

        if resolution < 0.01:
            print(
                "Please pick a lower resolution (e.g., bigger number).\nYou don't need it and it your computer may explode"
            )

        if resolution > 0.5:
            print("Please pick a higher resolution (e.g., number < 0.5). \n")

        else:

            # generate an array for fraction of each component
            f = np.arange(0, 1 + resolution, resolution)

            # all possible combinations for three f arrays
            a = np.array(np.meshgrid(f, f, f)).T.reshape(-1, 3)

            # where the combinations sum to 1
            f_vals = a[a.sum(axis=1) == 1]

            # get names of components
            components = df.index.tolist()

            # get names of columns where concentrations and ratios are held
            # IMPORTANT TO HAVE DATAFRAME IN THIS FORMAT
            elements = [col for col in df.columns if "_c" in col]
            ratios = [col for col in df.columns if "_r" in col]

            if len(elements) == 1:
                # Concentration of mixture using basic 3 component mixing
                # of concentrations
                el1_mix_concentrations = (
                    df.loc[components[0], elements[0]] * f_vals[:, 0]
                    + df.loc[components[1], elements[0]] * f_vals[:, 1]
                    + df.loc[components[2], elements[0]] * f_vals[:, 2]
                )

                # ratio values of the mixture using Albarede 1995 eq. 1.3.1
                el1_mix_ratios = (
                    df.loc[components[0], ratios[0]]
                    * (
                        (f_vals[:, 0] * df.loc[components[0], elements[0]])
                        / el1_mix_concentrations
                    )
                    + df.loc[components[1], ratios[0]]
                    * (
                        (f_vals[:, 1] * df.loc[components[1], elements[0]])
                        / el1_mix_concentrations
                    )
                    + df.loc[components[2], ratios[0]]
                    * (
                        (f_vals[:, 2] * df.loc[components[2], elements[0]])
                        / el1_mix_concentrations
                    )
                )

                results = pd.DataFrame(
                    {
                        "f_{}".format(components[0]): f_vals[:, 0],
                        "f_{}".format(components[1]): f_vals[:, 1],
                        "f_{}".format(components[2]): f_vals[:, 2],
                        "{}_mix".format(elements[0]): el1_mix_concentrations,
                        "{}_mix".format(ratios[0]): el1_mix_ratios,
                    }
                )

            else:

                # Concentration of mixture using basic 3 component mixing
                # of concentrations
                el1_mix_concentrations = (
                    df.loc[components[0], elements[0]] * f_vals[:, 0]
                    + df.loc[components[1], elements[0]] * f_vals[:, 1]
                    + df.loc[components[2], elements[0]] * f_vals[:, 2]
                )
                el2_mix_concentrations = (
                    df.loc[components[0], elements[1]] * f_vals[:, 0]
                    + df.loc[components[1], elements[1]] * f_vals[:, 1]
                    + df.loc[components[2], elements[1]] * f_vals[:, 2]
                )

                # ratio values of the mixture using Albarede 1995 eq. 1.3.1
                el1_mix_ratios = (
                    df.loc[components[0], ratios[0]]
                    * (
                        (f_vals[:, 0] * df.loc[components[0], elements[0]])
                        / el1_mix_concentrations
                    )
                    + df.loc[components[1], ratios[0]]
                    * (
                        (f_vals[:, 1] * df.loc[components[1], elements[0]])
                        / el1_mix_concentrations
                    )
                    + df.loc[components[2], ratios[0]]
                    * (
                        (f_vals[:, 2] * df.loc[components[2], elements[0]])
                        / el1_mix_concentrations
                    )
                )

                el2_mix_ratios = (
                    df.loc[components[0], ratios[1]]
                    * (
                        (f_vals[:, 0] * df.loc[components[0], elements[1]])
                        / el2_mix_concentrations
                    )
                    + df.loc[components[1], ratios[1]]
                    * (
                        (f_vals[:, 1] * df.loc[components[1], elements[1]])
                        / el2_mix_concentrations
                    )
                    + df.loc[components[2], ratios[1]]
                    * (
                        (f_vals[:, 2] * df.loc[components[2], elements[1]])
                        / el2_mix_concentrations
                    )
                )

                results = pd.DataFrame(
                    {
                        "f_{}".format(components[0]): f_vals[:, 0],
                        "f_{}".format(components[1]): f_vals[:, 1],
                        "f_{}".format(components[2]): f_vals[:, 2],
                        "{}_mix".format(elements[0]): el1_mix_concentrations,
                        "{}_mix".format(elements[1]): el2_mix_concentrations,
                        "{}_mix".format(ratios[0]): el1_mix_ratios,
                        "{}_mix".format(ratios[1]): el2_mix_ratios,
                    }
                )

    return results

#%% zircon saturation
# function to calculate zr saturation temperature
def t_zr_sat(M, zrmelt, model):
    """
    t_zr_sat calculates the zircon saturation temperature using
    the relationships found in both Watson and Harrison 1983 as
    well as Boehnke et al., 2013

        Inputs:
        M = (Na + K + 2Ca)/(Al*Si) in normalized cation fraction
        
        zrmelt = concentration of Zr in the melt
        
        model = 'watson' or 'boehnke'. This will govern the equation used
        to calculate zircon saturation temperature based on the equations 
        from watson and harrison 1983 or boehnke et al., 2013, respectively

        Returns:
        t = zircon saturation temperature for the chosen model

    BOTH TEMPERATURES ARE IN DEGREES CELSIUS
    
    """

    if model == "watson":

        t = 12900 / (2.95 + 0.85 * M + np.log(496000 / zrmelt)) - 273.15

    elif model == "boehnke":

        t = 10108 / (0.32 + 1.16 * M + np.log(496000 / zrmelt)) - 273.15

    return t

#%% basic melting and crystallization equations


def kd(Cs, Cl):
    """
    kd calculates a partition coefficient for a given set of measurements. For 
    igneous petrology, this is commonly the concentration of a trace element in
    the mineral divided by the concentration of the same trace element in the
    melt (e.g. Rollinson 1993 Eq. 4.3)
    
        Inputs:
        Cs = concnetration in the mineral
        Cl = concentration in the melt
        
        Returns:
        kd = partition coefficient for the given input parameters
        
    """
    kd = Cs / Cl
    return kd


# Distribution coefficient
def bulk_kd(kds, f_s):
    """
    bulk_kd generates a distribution coefficient that is the weighted sum of 
    partition coefficients for an element in a given mineral assemblage.
    Based off Rollinson 1993 Eq. 4.5

    Parameters
    ----------
    kds : array-like
        the individual partition coefficients of the mineral assemblage
    f_s : array-like
        the individual fractions of each mineral in the overall assemblage
        between 0 and 1

    Returns
    -------
    bulk_kd : the bulk partition coefficient for a given trace element for
    the mineral assemblage

    """
    D = np.sum(kds * f_s)
    return D


# melting equations
def non_modal_batch_melt(Co, Do, F, P):
    """
    non_modal_batch calculates the concentration of a given trace element in a melt produced from non modal
    batch melting of a source rock as described by Shaw (1970) equation 15.
        Inputs:
        Co = Concentration of trace element in the original solid
        Do = Bulk distribution coefficient for element when F = 0
        F = Fraction of original solid melted (fraction of melt)
        P = Bulk distribution coefficient of the melting mineral assemblage

        Returns:
        Cl = concentration in the newly formed liquid

    Note: if Do and P are the same, then you effectively have modal batch melting
    """

    Cl = Co * (1 / (F * (1 - P) + Do))
    return Cl


def non_modal_frac_melt(Co, Do, F, P):
    """
    non_modal_frac_melt calculates the composition of a trace element in a melt produced from non modal
    fractional melting of a source rock as described by Rollinson 1993 Eq. 4.13 and 4.14.
        Inputs:
        Co = Concentration of trace element in the original solid
        Do = Bulk distribution coefficient for element when F = 0
        F = Fraction of original solid melted (fraction of melt)
        P = Bulk distribution coefficient of melting mineral assemblage

        Returns:
        Cl = concentration in the extracted liquid. This is different from the 
        concentration of the instantaneous liquid.
        Cs = concentration in the residual solid

    Note: if Do and P are the same, then you effectively have modal fractional melting
    """

    Cl = (Co / F) * (1 - (1 - F * (P / Do)) ** (1 / P))

    return Cl


# dynamic melting
def non_modal_dynamic_melt(Co, Do, F, P, phi):
    """
    non_modal_dynamic_melt calculates the concentration of a liquid extracted via 
    dynamic melting as described in McKenzie (1985) and Zou (2007) Eq. 3.18. This is 
    applicable for a sitiuation in which melt is in equilibrium when the fraction 
    is below a critical value and then fractional when it is above that value.

    Parameters
    ----------
    Co : array-like
        Concentration of trace element in original solid
    Do : array-like
        Bulk distribution coefficient for element when F = 0
    F : array-like
        fraction of original solid melted (fraction of melt)
    P : array-like
        Bulk distribution coefficient of melting mineral assemblage
    phi : array-like
        critical mass porosity of residue

    Returns
    -------
    Cl : array-like
        Concentration of trace element in the liquid

    """

    X = (F - phi) / (1 - phi)

    Cl = (Co / X) * (
        1
        - (1 - ((X * (P + phi * (1 - P))) / (Do + phi * (1 - P))))
        ** (1 / (phi + (1 - phi) * P))
    )
    return Cl


# crystallization equations
def eq_xtl(
    Cl, D, F,
):
    """
    eq_xtl calculates the composition of a trace element in the remaining liquid after a certain amount of
    crystallization has occured from a source melt when the crystal remeains in equilibrium with the melt
    as described by White (2013) Chapter 7 eq. 7.81. It then calculates the concentration of that trace element
    in a specific solid phase based on partition coefficient input.
        Inputs:
        Cl = concentration of trace element in original liquid
        D = bulk distribution coefficient for trace element of crystallizing assemblage
        F = fraction of melt remaining

        Returns:
        Cl_new = concentration of trace element in the remaining liquid

    """
    Cl_new = Cl / (D + F * (1 - D))
    return Cl_new


# fractional crystallization
def frac_xtl(
    Cl, D, F,
):
    """
    frac_xtl calculates the composition of a trace element in the remaining liquid after a certain amount of
    crystallization has occured from a source melt when the crystal is removed from being in equilibrium with
    the melt as described by White (2013) Chapter 7 eq. 7.82.  It also calculates the 
    concentration of the trace element in the mean cumulate assemblage as described by Rollinson 1993 Eq. 4.20
        Inputs:
        Cl = concentration of trace element in original liquid
        D = bulk distribution coefficient for trace element of crystallizing assemblage
        F = fraction of melt remaining

        Returns:
        Cl_new = concentration of trace element in the remaining liquid
        Cr = concentration in the cumulate
    """
    Cl_new = Cl * (F) ** (D - 1)
    Cr = Cl * ((1 - F ** D) / (1 - F))
    return Cl_new, Cr


# in situ crystallization


def insitu_xtl(Cl, D, F, f, fa):
    """
    insitu_xtl calculates the concentration of the remaining melt as described
    in Langmuir (1989) and Rollinson 1993 Eq. 4.21 whereby crystallization 
    predominantly takes place at the sidewalls of a magma reservoir. Rather than 
    crystals being extracted from the liquid, liquid is extracted from a sidewall
    'mush' in situ. The solidification zone progressively moves through the magma
    chamber until crystallization is complete. In general this amounts in less
    enrichment of incompatible elements and less depletion of compatible elements
    than fractional crystallization

    Parameters
    ----------
    Cl : array-like
        concentration of trace element in original liquid
    D : array-like
        bulk partition coefficient of crystallizing of crystallizing assemblage
    F : array-like
        fraction of melt remaining (between >0 and 1). If 0 is in this array,
        error message will be thrown because python does not do division by 0
    f : array-like
        the fraction of interstitial liquid remaining after crystallization
        within the solidification zone. It is assumed that some of this is 
        trapped in the cumulate (ft) and some is returned to the magma (fa).
        therefore f = ft + fa
    fa : fraction of interstitial liquid that returns to the magma.f = fa would
        be an example where there is no interstital liquid in the crystallization
        front

    Returns
    -------
    Cl_new : array like
        concentration of extracted liquid from crystallization front

    """
    E = 1.0 / (D * (1.0 - f) + f)
    Cl_new = Cl * (F ** ((fa * (E - 1)) / (fa - 1)))
    return Cl_new


def fraclin_xtl(Cl, a, b, F):
    """
    fraclin_xtl calculates the composition of the liquid remaining after it 
    has experienced fractional crystallization where the distribution coefficient
    varies linearly with melt fraction. This was originally described by 
    Greenland 1970.

    Parameters
    ----------
    Cl : array-like
        concentration of the trace element in the original liquid
    a : array-like
        intercept of the relationship describing the linear change in D with melt
        fraction
    b : array-like
        slope of the relationship describing the linear change in D with 
        melt fraction
    F : array-like
        fraction of melt remaining (between 0 and 1). 

    Returns
    -------
    Cl_new : TYPE
        DESCRIPTION.

    """
    Cl_new = Cl * np.exp((a - 1) * np.log(F) + b * (F - 1))
    return Cl_new



