# Free energy landscapes of KcsA inactivation
**Authors: Sergio Pérez-Conesa and Lucie Delemotte**

------------

<div align="center"><p>
<a href="">
  <img src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python">
</a>
<a href="">
  <img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter" alt="Jupyter">
</a>
<a href="">
  <img src="https://img.shields.io/badge/VIM-%2311AB00.svg?style=for-the-badge&logo=vim&logoColor=white" alt="VIM">
</a>
<a href="https://www.linkedin.com/in/sperezconesa/">
  <img src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
</a>
</p>
</div>

<div align="center"><p>
<a href="https://github.com/psf/black">
  <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
</a>
<a href="https://lbesson.mit-license.org/">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT license">
</a>
<a href="">
  <img src="http://img.shields.io/badge/DOI-535698-B31B1B.svg" alt="DOI:535698">
</a>
<a href="https://twitter.com/intent/follow?screen_name=sperezconesa">
  <img src="https://img.shields.io/twitter/follow/sperezconesa?style=social&logo=twitter" alt="follow on Twitter">
</a>
<a href="https://twitter.com/intent/follow?screen_name=delemottelab">
  <img src="https://img.shields.io/twitter/follow/delemottelab?style=social&logo=twitter" alt="follow on Twitter">
</a>
<a href="https://github.com/sperezconesa/KcsA_string_method_FES">
    <img title="Star on GitHub" src="https://img.shields.io/github/stars/sperezconesa/KcsA_string_method_FES.svg?style=social&label=Star">
</a>
</p>
</div>



------------
![](./reports/final_figures/plots/FES_LB-CHARMM.png)
 The bacterial ion channel KcsA has become a useful model of complex K+-ion channels thanks to its single pore domain structure whose sequence shares many similarities with eukaryotic channels. Like many physiologically-relevant ion channels, KcsA inacti
vates after prolonged exposure to stimuli (in this case, a lowered pH). The inactivation mechanism has been heavily investigated, using structural, functional and simulations methods, but the molecular basis underlying the energetics of the process remain
s actively debated. In this work, we use the ``string method with swarms of trajectories'' enhanced sampling technique to characterize the free energy landscape lining the KcsA inactivation process. After channel opening following a pH drop, KcsA presents
 metastable open states leading to an inactivated state. The final inactivation step consists of a constriction of the selectivty filter and entry of three water molecules into binding sites behind each selectivity filter subunit. Based our simulations, w
e propose a key role for residue L81 in opening a gateway for water molecules to enter their buried sites, rather than for Y82 which has previously been suggested to act as a lid. In addition, since we found the energetically favored inactivation mechanis
m to be dependent on the force field, our results also address the importance of parameter choice for this type of mechanism. In particular, inactivation involves passing through the fully-open state only when using the AMBER force field. In contrast, usi
ng CHARMM, selectivity filter constriction proceeds directly from the partially open state. Finally, our simulations suggest that removing the co-purifying lipids stabilizes the partially open states, rationalizing their importance for the proper inactiva
tion of the channel.

This code was developed by [Sergio Pérez-Conesa](https://www.linkedin.com/in/sperezconesa/). I am a member of the [Delemottelab](https://github.com/delemottelab) led by [prof. Lucie Delemotte](https://www.biophysics.se/index.php/members/lucie-delemotte/). All the explanations can be found in the article and the rest of code and data [here](https://osf.io/snwbc/?view_only=1338fd9e92f941deb7452525c1e9fdfa)

I am happy to connect and discuss this and other projects through [github](https://github.com/sperezconesa), [linkedin](https://www.linkedin.com/in/sperezconesa), [twitter](https://twitter.com/sperezconesa), [email](sperezconesa@gmail.com) etc.
Feel free to suggest ways we could have improved this code.

You can find more updates on the Delemottelab on [twitter](https://twitter.com/delemottelab) and the rest of our [publications](https://scholar.google.es/citations?user=OaHNSvEAAAAJ&hl=en&oi=ao).

If you want to cite this code, please use CITE.bib, thank you!

[Published Preprint](https://www.biorxiv.org/content/10.1101/2023.04.05.535698v1)

Published Article: Coming soon :wink: []()

## Running the code


### Recreate conda environment

To recreate the conda environment used:

```bash
conda env create -f environment.yml
conda activate string_sims
ipython kernel install --user --name=string_sims
pip install -e .
```

Use `environment_exact.yml` for the exact environment.

### Getting additional data files

All the data, including the inference models, simulations etc. can be found in [Open Software Foundation](https://osf.io/snwbc/?view_only=1338fd9e92f941deb7452525c1e9fdfa).

## Project Organization

```text
├── LICENSE
│
├── Makefile           <- Makefile with commands like `make update_data` or `make format`
│
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── raw            <- Raw data generated or colected for this work
│   ├── interim        <- Intermediate data that has been transformed.
│   └── processed      <- The final, canonical data sets for modeling.
│
├── models             <- MD and experimental models and input files
│
├── notebooks          <- Jupyter notebooks.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── tex            <- Latex input files
│   ├── figures        <- Figures of the data.
│   └── final_figures  <- Generated graphics and figures to be used in final report.
│
├── environment.yml    <- The necessary packages to install in conda environment.
│
├── environment_exact.yml   <- The exact package versions used.
│
├── setup.py           <- File to install python code
│
├── src                <- Source code for use in this project.
│   ├── analysis       <- Python code for analysis of data
│   ├── data           <- Python code for handling the data
│   ├── __init__.py    <- Makes src a Python module
│   └── data           <- Scripts to download or generate data
│
├── visualization                <- File to help visualize data and trajectories
│
└──
```

------------

Project based on the [cookiecutter for Molecular Dynamics](https://github.com/sperezconesa/cookiecutter-md). Which is itself based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/) \#cookiecutterdatascience

------------

