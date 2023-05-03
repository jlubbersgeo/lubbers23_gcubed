# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 13:34:13 2021

@author: jlubbers
"""

from io import BytesIO

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

# import panel as pn
# import holoviews as hv
# import hvplot.pandas


#%%
def harker_plot(
    data,
    x_element,
    y_elements,
    axes,
    custom_ylabels=None,
    categorical=None,
    legend_loc="best",
    **sct_kwargs
):
    """


    Parameters
    ----------
    data : pandas DataFrame
        the host dataframe that holds data to be plotted
    x_element : string
        x-axis data e.g. "SiO2"
    y_elements : list
        list of strings containing column names for y variables
        to be utilized
    axes : ndarray
        1D array of matplotlib axis objects to be used in plotting e.g.:
            fig, ax = plt.subplots(3,2, figsize)
            axes = ax.ravel()
    custom_ylabels : list, optional
        list of custom labels to be used for labeling the plots. Accepts
        latex formatting. Default is [x_element + y_elements].
    categorical : string, optional
        column to be utilized as a categorical variable, default = None
        If not none, will utilize seaborn scatterplot function to
        categorically color data in accordance to unique column values
    **sct_kwargs :
        **sct_kwargs = inherited keyword arguments from
        matplotlib.pyplot.scatter
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    axes : ndarray
        1D array of matplotlib axis objects now filled with data. Same shape
        as input axes argument.

    """

    if len(y_elements) % 2 != 0:
        raise Exception("Please choose an even number of elements to plot.")
    else:

        if custom_ylabels is None:
            my_labels = [x_element] + y_elements
        else:
            my_labels = custom_ylabels

        # basic plot no categorical variable chosen (default)
        if categorical is None:

            for element, label, a, i in zip(
                y_elements, my_labels[1:], axes, range(len(y_elements))
            ):

                if i >= len(y_elements) - 2:
                    a.scatter(data[x_element], data[element], **sct_kwargs)
                    a.set_xlabel(my_labels[0], fontsize=20)
                else:
                    a.scatter(data[x_element], data[element], **sct_kwargs)
                    a.set_xticklabels("")

                a.set_ylabel(label, fontsize=20)

                a.minorticks_on()
                for axis in ["top", "bottom", "left", "right"]:
                    a.spines[axis].set_linewidth(2)

                a.tick_params(axis="both", which="major", width=2)

        # if a categorical column is chosen:
        # inherits from seaborn.scatterplot()
        else:
            for element, label, a, i in zip(
                y_elements, my_labels[1:], axes, range(len(y_elements))
            ):

                if i == 0:

                    sns.scatterplot(
                        data=data,
                        x=x_element,
                        y=element,
                        ax=a,
                        hue=categorical,
                        **sct_kwargs,
                    )
                    a.legend(loc=legend_loc)
                else:
                    sns.scatterplot(
                        data=data,
                        x=x_element,
                        y=element,
                        ax=a,
                        hue=categorical,
                        **sct_kwargs,
                        legend=False,
                    )

                a.set_ylabel(label, fontsize=20)

                if i >= len(y_elements) - 2:
                    a.set_xlabel(my_labels[0], fontsize=20)
                else:
                    a.set_xticklabels("")
                    a.set_xlabel("")

                a.minorticks_on()
                for axis in ["top", "bottom", "left", "right"]:
                    a.spines[axis].set_linewidth(2)

                a.tick_params(axis="both", which="major", width=2)

    return axes


# Plot LeMaitre lines
def add_LeMaitre_fields(
    plot_axes, x_lim=0, fontsize=8, color=(0.6, 0.6, 0.6), **kwargs
):
    """Add fields for geochemical classifications from LeMaitre et al (2002)
    to pre-existing axes.  If necessary, the axes object can be retrieved via
    plt.gca() command. e.g.

    ax1 = plt.gca()
    add_LeMaitre_fields(ax1)
    ax1.plot(silica, total_alkalis, 'o')

    Fontsize and color options can be used to change from the defaults.

    It may be necessary to follow the command with plt.draw() to update
    the plot.

    COPYRIGHT:  (C) 2015 John A Stevenson / @volcan01010
                       Joaquin Cortés
                WEBSITE: http://all-geo.org/volcan01010

    Le Maitre RW (2002) Igneous rocks : IUGS classification and glossary of
        terms : recommendations of the International Union of Geological
        Sciences Subcommission on the Systematics of igneous rocks, 2nd ed.
        Cambridge University Press, Cambridge

    """

    # Prepare the field information
    from collections import namedtuple

    FieldLine = namedtuple("FieldLine", "x1 y1 x2 y2")
    lines = (
        FieldLine(x1=41, y1=0, x2=41, y2=7),
        FieldLine(x1=41, y1=7, x2=52.5, y2=14),
        FieldLine(x1=45, y1=0, x2=45, y2=5),
        FieldLine(x1=41, y1=3, x2=45, y2=3),
        FieldLine(x1=45, y1=5, x2=61, y2=13.5),
        FieldLine(x1=45, y1=5, x2=52, y2=5),
        FieldLine(x1=52, y1=5, x2=69, y2=8),
        FieldLine(x1=49.4, y1=7.3, x2=52, y2=5),
        FieldLine(x1=52, y1=5, x2=52, y2=0),
        FieldLine(x1=48.4, y1=11.5, x2=53, y2=9.3),
        FieldLine(x1=53, y1=9.3, x2=57, y2=5.9),
        FieldLine(x1=57, y1=5.9, x2=57, y2=0),
        FieldLine(x1=52.5, y1=14, x2=57.6, y2=11.7),
        FieldLine(x1=57.6, y1=11.7, x2=63, y2=7),
        FieldLine(x1=63, y1=7, x2=63, y2=0),
        FieldLine(x1=69, y1=12, x2=69, y2=8),
        FieldLine(x1=45, y1=9.4, x2=49.4, y2=7.3),
        FieldLine(x1=69, y1=8, x2=77, y2=0),
    )

    FieldName = namedtuple("FieldName", "name x y rotation")
    names = (
        FieldName("Picro\nbasalt", 43, 2, 0),
        FieldName("Basalt", 48.5, 2, 0),
        FieldName("Basaltic\nandesite", 54.5, 2, 0),
        FieldName("Andesite", 60, 2, 0),
        FieldName("Dacite", 68.5, 2, 0),
        FieldName("Rhyolite", 76, 9, 0),
        FieldName("Trachyte", 64.5, 11.5, 0),
        FieldName("Basaltic\ntrachyandesite", 53, 8, -20),
        FieldName("Trachy-\nbasalt", 49, 6.2, 0),
        FieldName("Trachyandesite", 57.2, 9, 0),
        FieldName("Phonotephrite", 49, 9.6, 0),
        FieldName("Tephriphonolite", 53.0, 11.8, 0),
        FieldName("Phonolite", 57.5, 13.5, 0),
        FieldName("Tephrite\n(Ol < 10%)", 45, 8, 0),
        FieldName("Foidite", 44, 11.5, 0),
        FieldName("Basanite\n(Ol > 10%)", 43.5, 6.5, 0),
    )

    # Plot the lines and fields
    for line in lines:
        plot_axes.plot(
            [line.x1, line.x2], [line.y1, line.y2], "-", color=color, zorder=0, **kwargs
        )
    for name in names:
        if name.x > x_lim:

            plot_axes.text(
                name.x,
                name.y,
                name.name,
                color=color,
                size=fontsize,
                horizontalalignment="center",
                verticalalignment="top",
                rotation=name.rotation,
                zorder=0,
            )


def bytes_to_df(filename, bytes_object, sheet_name="Sheet1"):
    """
    function for converting the input of the fileinput widget to a dataframe
    if it is either an xlsx or csv file

    Parameters
    ----------
    filename : string
        file_input.filename
    bytes_object : bytesIO object
        file_input.value
    sheet_name : string, optional
        name of sheet if xlsx file. Default is "Sheet1"

    Raises
    ------
    SyntaxError
        DESCRIPTION.

    Returns
    -------
    df : pandas DataFrame
        pandas dataframe of the input file

    """

    if "xlsx" in filename:

        df = pd.read_excel(BytesIO(bytes_object), sheet_name=sheet_name)

    elif "csv" in filename:

        df = pd.read_csv(BytesIO(bytes_object))

    else:
        raise SyntaxError(
            "Please choose either a csv or xlsx file to convert to a dataframe"
        )

    return df


# def create_data_explorer(df):
#     """
#     This builds a simple bivariate data explorer that allows the user to
#     interactively and rapidly inspect their data by plotting any two columns
#     from their input data frame against one another

#     Parameters
#     ----------
#     df : pandas DataFrame
#         the input data to be plotted (e.g., your geochemistry)

#     Returns
#     -------
#     a Panel dashboard for plotting data

#     """
#     # Building the widgets
#     cols = list(df.columns)
#     # x y data widgets
#     x = pn.widgets.Select(value=None, options=cols, name="x-values")
#     y = pn.widgets.Select(value=None, options=cols, name="y-values")

#     # widgets for categorical and value coloring
#     cols.insert(0, "None")
#     groupby = pn.widgets.Select(value="None", options=cols, name="Group by categorical")
#     colorby = pn.widgets.Select(value="None", options=cols, name="Color by value")

#     # widgets for colormaps
#     cmaps = hv.plotting.list_cmaps(category="Uniform Sequential")
#     colormaps = pn.widgets.Select(value="viridis", options=cmaps, name="Colormap")


#     # widgets for face and edge color
#     face_color = pn.widgets.ColorPicker(name="face color", value="#1e81b0")
#     edge_color = pn.widgets.ColorPicker(name="edge color", value="#030303")

#     # widgets for transparency, and sizing
#     alpha = pn.widgets.FloatSlider(name="Transparency", start=0, end=1, step=0.1, value=1)
#     size = pn.widgets.FloatSlider(name="Marker Size", start=0, end=100, step=1, value=50)
#     linewidth = pn.widgets.FloatSlider(name="Linewidth", start=0, end=5, step=.25, value=.5)

#     # widgets for logx and logy scales
#     logx = pn.widgets.Checkbox(name='log x-scale')
#     logy = pn.widgets.Checkbox(name='log y-scale')


#     #Building a plotting function

#     def plot_data(
#         xval,
#         yval,
#         facecolor="gray",
#         edgecolor="black",
#         groupby="None",
#         colorby="None",
#         cmap="viridis",
#         alpha=1,
#         size=5,
#         linewidth = 1,
#         logx = False,
#         logy = False
#     ):
#         height = 400
#         width = 600

#         if groupby != "None":

#             return df.hvplot.scatter(
#                 xval,
#                 yval,
#                 by=groupby,
#                 line_color=edgecolor,
#                 alpha=alpha,
#                 s=size,
#                 height = height,
#                 width = width,
#                 line_width = linewidth,
#                 logx = logx,
#                 logy = logy

#             ).opts(tools=["hover", "lasso_select", "box_select"])

#         elif colorby != "None":

#             return df.hvplot.scatter(
#                 xval,
#                 yval,
#                 c=colorby,
#                 line_color=edgecolor,
#                 line_width = linewidth,
#                 clabel=colorby,
#                 cmap=cmap,
#                 alpha=alpha,
#                 s=size,
#                 height = height,
#                 width = width,
#                 logx = logx,
#                 logy = logy,

#             ).opts(tools=["hover", "lasso_select", "box_select"])

#         else:

#             return df.hvplot.scatter(
#                 xval,
#                 yval,
#                 c=facecolor,
#                 line_color=edgecolor,
#                 alpha=alpha,
#                 s=size,
#                 height = height,
#                 width = width,
#                 line_width = linewidth,
#                 logx = logx,
#                 logy = logy


#             ).opts(tools=["hover", "lasso_select", "box_select"])

#     # The actual layout of our app.
#     layout = pn.Card(
#         pn.Row(
#             pn.Column(
#                 pn.Row(
#                     pn.Column(x, y, face_color, edge_color,linewidth, width=200),
#                     pn.Column(groupby, colorby, colormaps, alpha, size,logx,logy, width=200),
#                 ),
#             ),
#             pn.Column(
#                 plot_data(
#                     x.value,
#                     y.value,
#                     face_color.value,
#                     edge_color.value,
#                     groupby.value,
#                     colorby.value,
#                     colormaps.value,
#                     alpha.value,
#                     size.value,
#                     linewidth.value,
#                     logx.value,
#                     logy.value
#                 ),
#             ),
#         ),
#         header_background="gray",
#         header_color="black",
#         title="Data Explorer: More thinking, less coding",
#         background = 'whitesmoke',
#         width = 1100
#     )

#     # This calls the plot_data function each time a value of a widget changes
#     # ultimately updating our plot!

#     def update(event):
#         layout[0][1][0].object = plot_data(
#             x.value,
#             y.value,
#             face_color.value,
#             edge_color.value,
#             groupby.value,
#             colorby.value,
#             colormaps.value,
#             alpha.value,
#             size.value,
#             linewidth.value,
#             logx.value,
#             logy.value
#         )


#     x.param.watch(update, "value")
#     y.param.watch(update, "value")
#     face_color.param.watch(update, "value")
#     edge_color.param.watch(update, "value")
#     groupby.param.watch(update, "value")
#     colorby.param.watch(update, "value")
#     colormaps.param.watch(update, "value")
#     alpha.param.watch(update, "value")
#     size.param.watch(update, "value")
#     linewidth.param.watch(update, "value")
#     logx.param.watch(update,"value")
#     logy.param.watch(update,"value")

#     return layout


#%% prediction probability plot
def plot_probabilities_map(
    probabilities,
    n,
    lat,
    lon,
    extent,
    center_lon,
    cbar=True,
    ax=None,
    sct_kwargs={},
    cbar_kwargs={},
):
    """
    Plot the probability of each class for a given prediction from a
    kigusiq.learning.tephraML model that incorporates its geospatial
    information. Uses cartopy basemap features:
    https://scitools.org.uk/cartopy/docs/latest/gallery/lines_and_polygons/features.html#sphx-glr-gallery-lines-and-polygons-features-py

    Parameters
    ----------
    probabilities : pandas DataFrame
        dataframe with model prediction probabilities. This is the output
        from tephraML.model.prediction_probability. Shape is mxn where m is
        number of predictions and n is number of classes.
    n : int
        Prediction number. This corresponds to the row number in
        probabilities dataframe.
    lat : array-like
        array of latitudes for each class in probabilities dataframe
    lon : array-like
        array of longitudes for each class in probabilities dataframe
    extent : array-like
        bounding box for the map [lon_left,lon_right,lat_bottom,lat_top]
    center_lon : scalar
        centering longitude for bounding box
    cbar : Boolean, optional
        Whether or not to show a colorbar. The default is True.
    ax : matplotlib.pyplot.axes, optional
        axis to map the plot to. The default is None.
    sct_kwargs : dictionary, optional
        dictionary of scatterplot keyword arguments. The default is {}.
        see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    cbar_kwargs : dictionary, optional
        dictionary of colorbar keyword arguments. The default is {}.
        see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html

    Returns
    -------
    ax : matplotlib.pyplot.axes
        axis filled with basemap and probability scatterplot data

    """
    # get probabilities for a single prediction
    probability = probabilities.iloc[n, :].to_numpy(dtype="float")

    fig = plt.gcf()
    if ax is None:

        ax = plt.axes(projection=ccrs.Mercator(central_longitude=center_lon))

    else:
        ax = ax

    # # add geoaxes and set extent with lat and lon ticks
    # ax = fig.add_subplot(plot_loc[0],plot_loc[1],plot_loc[2],projection=ccrs.Mercator(central_longitude=center_lon)
    # )
    ax.set_extent(extent)
    ax.coastlines(linewidth=1)
    ax.set_xticks(np.arange(extent[0], extent[1], 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(extent[2], extent[3], 5), crs=ccrs.PlateCarree())

    lon_formatter = LongitudeFormatter(
        number_format=".1f", degree_symbol="", dateline_direction_label=True
    )
    lat_formatter = LatitudeFormatter(number_format=".1f", degree_symbol="")
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    # add land and ocean features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)

    # add thick border around edge
    ax.spines["geo"].set_linewidth(2)

    # add scatter at each location sized and colored by the probability
    scatter = ax.scatter(
        x=lon,
        y=lat,
        c=probability,
        transform=ccrs.PlateCarree(),
        **sct_kwargs,
    )

    # add a colorbar with kwargs. Default is True
    if cbar is True:

        cbar = fig.colorbar(scatter, ax=ax, **cbar_kwargs)
        cbar.set_label("Source Probability", fontsize=12)

    return ax
