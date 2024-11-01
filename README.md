# KinSynpy

> Short for Kinematics-Synergies-py

Collection of functions written for analyzing data related to my lab, which studies locomotion. This includes functions relating to kinematic analysis of markerless tracking done with [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) under `latstability` and `dlctools`. It also includes some functions for electromyography (EMG) analysis recorded through CED's Spike2 software.

## Installation

Not recommended at this point but if you're curious you can grab the kinsynpy folder from the repo and install it to a local conda environment.

```
conda activate <name of env>
git clone https://github.com/Kenmac589/kinsynpy.git
cd ./kinsynpy/
pip intall .
```

## Credits

Credit for `load_data` and `smooth_trajectory` in `dlctools` are exact copies from `dlc2kinematics` and of course the whole suite of software around it. Simply wanted to minimize the dependencies for the scripts.

```
@software{dlc2kinematics,
  author       = {Mathis, Mackenzie and
                  Lauer, Jessy and
                  Nath, Tanmay and
                  Sandbrink, Kai and
                  Beauzile, Michael and
                  Hausmann, Sébastien and
                  Schneider, Steffen and
                  Mathis, Alexander},
  title        = {{DLC2Kinematics: a post-deeplabcut module for kinematic analysis}},
  month        = feb,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.0.4},
  doi          = {10.5281/zenodo.6669074},
  url          = {https://doi.org/10.5281/zenodo.6669074}
}
```
