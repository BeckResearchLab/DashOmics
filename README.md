# DashOmics
DashOmics is a visualization tool to explore *omics data using clustering analysis. It is created by [Dash Plot.ly](https://plot.ly/products/dash/), a Python framework for building interactive analytical tools. 

Users can play with existing example data, or upload their own data in SQLite database. K-Means clustering method would be applied on RNA-seq data, including two model evaluation methods — elbow method and silhouette analysis, to help find the optimal k value (the number of clusters). Users can explore cluster profiles of grouped genes based on specific k value and generate insights into gene functions and networks.

**Table of contents:**

* Requirements
* Installation
* Getting Started
* Examples



## Requirements

* [pandas](http://pandas.pydata.org/)
* [sklearn](https://github.com/scikit-learn/scikit-learn)
* [numpy](http://www.numpy.org/)
* [plot.ly dash](https://plot.ly/products/dash/)
  * dash
  * dash-html-component
  * dash-core-components



## Installation

Open the terminal:

```Bash
1. git clone "https://github.com/BeckResearchLab/DashOmics.git"
2. cd DashOmics/dashomics
3. python run.py
...Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)
```



## Getting Started

![DashOmics Workflow](https://github.com/BeckResearchLab/DashOmics/blob/master/images/DashOmics-Workflow.jpg)

## Examples

**Example-1:**

Data from [**Cu_transition_time_course-**](https://github.com/gilmana/Cu_transition_time_course-) by [Alexey Gilman](https://digital.lib.washington.edu/researchworks/handle/1773/39973)

This RNA-seq dataset consists of over 40 samples from 12 different bioreactor experiments. It is normalized for gene length and total counts using the transcripts per million method (TPM) and Log2 transformation to reduce the variance between samples. 

The normalized read counts are the starting point for the computational data analysis. 

 

