# `kinumaax`: modules for working with tephra data

_under significant development, changes happening often!_

## Motivation

Data gathered from volcanic tephras can be extremely valuable for helping unravel many questions volcanologists and petrologists are interested in, namely:

- How large was a given eruption?
- What is the source volcano for a given tephra layer?
- How often does a given volcano have eruptions that produce significant amounts of tephra?

While all of these questions are extremely important to help better understand and mitigate the hazards associated with tephra producing eruptions, addressing them involves no small amount of work once tephra samples and their geochemical data have been collected! This is where we introduce `kinumaax`, a small but powerful package to help manipulate, visualize, and learn about your tephra data.

**_Why kinumaax̂?_**

- "kinumaax̂" is the [Unungax̂](https://www.uaf.edu/anlc/languages/aleut.php) word for "volcano ashes". Much of this project was developed working on volcanoes that are located within traditional Unungax̂ lands across the Aleutian Islands and is our way of acknowledging that we are but guests in our scientific pursuits.

## Modules

`kinumaax` is currently comprised of three modules:

1. `kinumaax.learning`
   - Using machine learning to predict source volcanoes based off of input geochemistry
2. `kinumaax.visualization`
   - plotting various visualizations related to working with tephra data (e.g., Harker, TAS, REE diagrams) and the results from `kinumaax.learning` (e.g., Confusion Matrix and feature importance plots).
3. `kinumaax.crunching`
   - This has functions for data cleaning, filtering, pre-processing, geochemical modeling, and more number crunching activities
