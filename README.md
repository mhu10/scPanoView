# Single-cell Panoramic View Clustering (PanoView) #

**PanoView** is an iterative PCA-based method integrated with a novel density-based clustering, ordering local maximum by convex hull (OLMC) algorithm, to identify cell subpopulations for single-cell RNA-sequencing.




![PanoView](https://github.com/mhu10/scPanoView/blob/master/PanoView.jpg)
<p align="center">
  :heavy_plus_sign:

<p align="center">
  <img width="350" height="350" src="https://github.com/mhu10/scPanoView/blob/master/OLMC_animation.gif">
</p>


## Installation ##
**PanoView** is a python module that uses other common python libraries such as *numpy*, *scipy*, *pandas*, *scikit-learn*, etc., to realize the proposed algorithm. Prior to installing **PanoView** from Github repository, please make sure that Git is properly installed or go to https://git-scm.com/  for the installation of Git.
To install **PanoView** at your local computer, open your command prompt and type the following

```
pip install git+https://github.com/mhu10/scPanoView.git#egg=scPanoView
```

It will install all the required python libraries for executing **PanoView**. To test the installation of **PanoView**, open the python interpreter or your preferred IDE (*Spyder*, *PyCharm*, *Jupyter*, etc. ) and type the following

```
from PanoramicView import scPanoView
```
There should not be any error message popping out.

Note: PanoView was implement and tested by python3.6.  python2.7 may also work expcept the visualization of identified clusters. The output figures may not be identical to the ones in the manual.



## Tutorial ##

Plese refer to the manuaul ( *"PanoViewManual.pdf"* ) for details of executing **PanoView** algorithm in python.

For running tutorial in the manual, please download the example dataset (*"ExamplePollen.zip"* ) and upzip it into your python working directory.
