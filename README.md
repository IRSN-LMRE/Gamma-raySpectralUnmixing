#  Spectral unmixing for activity estimation in Gamma-ray spectrometry
The toolbox contains the Python implementation of spectral unmixing algorithms to unmix a measured gamma-ray spectrum, includes the following modules:
* Spectral unmixing based on the Chambolle-Pock algorithm, which tackles the activity estimation in a supervised framework (spectral signatures and the background spectrum are known).
* Sparse spectral unmixing based on Poisson-based Greedy algorithm, named as P-OMP. This algorithm identifies spectral signatures (active radionuclides).
* Spectrum analysis pipeline for an HPGe measurement

1. [Spectral unmxing](#unmix )
2. [Sparse spectral unmixing](#sparseunmix)
3. [Metrological use of spectral unmixing algorithm](#data)
4. [Test examples](#test)

## Spectral unmxing <a name="unmix"></a>
A measured gamma-ray spectrum is the sum of individual spectra of radionuclides and the background spectrum. The Poisson process of radioactive decay leads to model the problem as: 
<p align="center">
<img src=figures/model.png width=600>
</p>
The activity estimation problem is formulated as an inverse problem, which aims to find the mixing weights by minimizing the sum of data fidelity term and regularization terms:
<p align="center">
<img src=figures/fig1.png width=250>
</p>
Along with the non-negativity penalization, the Poisson statistics-based estimator can be formulated as follows:
<p align="center">
<img src=figures/fig2.png width=550>
</p>

## Sparse spectral unmixing <a name="sparseunmix"></a>
The sparse spectral unmixing jointly estimates the set of active radionuclides and their activities with a sparsity constraint:
<p align="center">
<img src=figures/fig3.png width=400>
</p>
It selects forward the radionuclides in a dictionary by sequentially adding the radionuclide that maximizes the Poisson likelihood. For more details refer to:
</p>

* [Xu, J., Bobin, J., de Vismes Ott, A., and Bobin, C. (2020). Sparse spectral unmixing for activity estimation in γ-ray spectrometry ap- plied to environmental measurements. Applied Radiation and Isotopes, 156:108903.](https://www.sciencedirect.com/science/article/abs/pii/S0969804319303422)

## Metrological use of spectral unmixing algorithm <a name="data"></a>
The metrological use of spectrum analysis tool requires:

* Characterisitc limits
* Instrument calibrations

## Test examples<a name="test"></a>
The test examples can be found in the following Jupyter Notebook:

* [example_unmix.ipynb](https://github.com/IRSN-LMRE/Gamma-raySpectralUnmixing/blob/master/example_unmix.ipynb), spectral unmixing.
* [example_sparse.ipynb](https://github.com/IRSN-LMRE/Gamma-raySpectralUnmixing/blob/master/example_sparse.ipynb), sparse spectral unmixing.
* [example_realdata.ipynb](https://github.com/IRSN-LMRE/Gamma-raySpectralUnmixing/blob/master/example_realdata.ipynb), Real data analysis.

#### This work was done in the thesis of Jiaxin Xu, under the supervision of Jérôme Bobin, CEA and Anne de Vismes Ott, IRSN.

