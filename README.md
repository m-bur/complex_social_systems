# complex_social_systems

Welcome to the GitHub of Team BJSS-Model

Please read all of the readme before diving into our code, it will save you some time.

We used [numpy doc style](https://numpydoc.readthedocs.io/en/latest/format.html) for our code. This also allows for automatic documentation using sphinx.

The documentation of our project can be found in the [docs](docs) folder. More instructions can be found there.
It can be opened with your favorite browser and looks something like this:

![picture of documentation](documentation_example.png)
## CI Status:

Flake8 check:
![tests](https://github.com/m-bur/complex_social_systems/actions/workflows/style-check.yml/badge.svg)

## Some important information for you

The most important file is [main.py](main.py). This is where the main loop of our programm is located. In this file, many functions and classes are used that are defined in the [utils](utils) folder. In particular, we have:

- nodes.py which defines the classes for voter and media nodes
- network.py which generates the network, and defines processes on the network
- measure.py defines functions to calculate important quantities and also to save them
- visualization.py offers tools visualize these quantities

Furthermore, there is folder called [networks](networks), which contains precalculated networks, which avoids the necessesity to calculate the initial network each time from scratch.

[calibration.py](calibration.py) was used to perform the calibration, the results can be found in the [calibration](calibrations) folder.

The files named 'experiment' are scripts that allow for the parallel execution of the main file for different input parameters, this leads to a faster simulation.

### Multidimensional data

The code for the multidimensional opinion dynamics can be found in another branch called `opinion_multi-dimensionality`.

### Data availability:
The results of the experiments that are analyzed in the [analysis.ipynb](analysis.ipynb) are available in this
polybox [link](https://polybox.ethz.ch/index.php/s/SydFlQwt6FcSetc). 


### Presentation and Report

You can also find the report of the project in [complex_social_systems_report.pdf](complex_social_systems_report.pdf).

The presentation is available is available via this polybox [link](https://polybox.ethz.ch/index.php/s/CcdPgqdmIzNWI4w).

#### Abstract:

This work aims to investigate the impact of media on the opinion dynamics of a simulated election via agent-based
modeling. The complex relationship between voters and media is simplified by using a network of voters with hierarchical
mutual connections on a two dimensional grid and a separate set of media nodes with random connections to voters.
The results reveal a strong media influence on voter opinions, with significant implications for media-driven manipulation.
When voters are allowed to choose the media outlets they are listening to, media manipulation becomes ineffective over
almost the entire range of media opinions. Only if neutral media nodes are manipulated, a substantial lead of one opinion
over the other emerges, suggesting that susceptibility to media manipulation concentrates among undecided voters in
the presence of echo chambers.

