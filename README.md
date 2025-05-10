# Sedimentary Velocity Model for the San Francisco Bay Area

## Description

This repository contains the regression scripts, model implementation, and data for the Sedimentary Velocity Model for the San Francisco Bay Area described in: 
[Lavrentiadis G, Seylabi E, Xia F, Tehrani H, Asimaki D, McCallen D. Data-driven characterization of near-surface velocity in the San Francisco Bay Area: A stationary and spatially varying approach. Earthquake Spectra. 2025 doi:10.1177/87552930251320666](https://journals.sagepub.com/doi/abs/10.1177/87552930251320666)

An open-access version of the manuscript can be found in [arXiv](https://arxiv.org/abs/2409.18856).

This paper supersedes the USGS report: [Tehrani, H., Lavrentiadis, G., Seylabi, E., McCallen, D., & Asimaki, D. (2023). Final Technical Report (2021-2022) Towards a Three-Dimensional Geotechnical Layer Model for Northern California Collaborative Research with the University of Nevada Reno and California Institute of Technology.](https://earthquake.usgs.gov/cfusion/external_grants/reports/G21AP10448.pdf)

## Repository Structure

The main folder, ``Analyses``, contains all the preprocessing, regression, prediction, validation, and library scripts. 

Within the `Analyses` folder:

- The `preprocessing` contains scripts for homogenizing the velocity data for regression.
- The `regression` holds scripts for running both the stationary and spatially varying regression models, as well as for performing semivariogram analysis.
- The `implementation` includes scripts for evaluating the velocity models.
- The `site_response` contains files for generating velocity profiles and conducting the site response analysis.

The main `Data` folder mirrors the structure of the `Analyses` folder and includes all the corresponding input and output files.

The ``Raw_files`` includes project files in raw format. 

    .
    |--Analyses
    |     |--preprocessing
    |     |--regression
    |     |--implementation
    |     |--prof_termination
    |     |--scaling_functions
    |     |--site_reponse
    |     |--gis
    |     |--miscellaneous
    |     |--python_lib
    |     |--stan_lib
    |
    |--Data
    |     |--vel_profiles_dataset
    |     |--regression
    |     |--scaling_functions
    |     |--site_reponse
    |     |--misc
    |     |--gis
    |     
    |--Raw_files


## Collaborators
 - Grigorios Lavrentiadis -- Postdoctoral Associate, California Institute of Technology
 - Elnaz Esmaeilzadeh Seylabi -- Assistant Professor, University of Nevada Reno
 - Feiruo Xia -- Ph.D. Candidate, California Institute of Technology
 - Hesam Tehrani -- Ph.D. Candidate, University of Nevada Reno
 - Domniki Asimaki -- Professor, California Institute of Technology
 - David McCallen -- Earth Senior Scientist, Lawrence Berkeley National Laboratory

## Acknowledgments 
This material is based upon work supported by the U.S. Geological Survey under Grant No. G21AP10518 and G21AP10448.
The views and conclusions contained in this paper are those of the authors and should not be interpreted as representing the opinions or policies of the U.S. Geological Survey. Mention of trade names or commercial products does not constitute their endorsement by the U.S. Geological Survey.
