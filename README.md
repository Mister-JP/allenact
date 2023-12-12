# Instructions to run the code locally:

To install a conda environment with name allenact using this file you can simply run the following :
```bash
export MY_ENV_NAME=allenact
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
export PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc"
conda env create --file ./conda/environment-base.yml --name $MY_ENV_NAME
```

Activate the environment
```bash
conda activate allenact
```
Navigate to the main directory by running: `cd allenact`

Download the datasets:
```bash
bash datasets/download_navigation_datasets.sh robothor-pointnav
```







[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Documentation Status](https://img.shields.io/badge/docs-up%20to%20date-Green.svg)](https://allenact.org)
[![Latest Release](https://img.shields.io/github/v/release/allenai/allenact)](https://github.com/allenai/allenact/releases/latest)
[![Python 3.7](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![LGTM Grade: Python](https://img.shields.io/lgtm/grade/python/g/allenai/allenact.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/allenai/allenact/context:python)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**AllenAct** is a modular and flexible learning framework designed with a focus on the unique requirements of Embodied-AI research. It provides first-class support for a growing collection of embodied environments, tasks and algorithms, provides reproductions of state-of-the-art models and includes extensive documentation, tutorials, start-up code, and pre-trained models.

AllenAct is built and backed by the [Allen Institute for AI (AI2)](https://allenai.org/). AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.

## Quick Links

- [Website & Docs](https://www.allenact.org/)
- [Github](https://github.com/allenai/allenact)
- [Install](https://www.allenact.org/installation/installation-allenact/)
- [Tutorials](https://www.allenact.org/tutorials/)
- [AllenAct Paper](https://arxiv.org/abs/2008.12760)
- [Citation](#citation)

## Acknowledgments
This work builds upon the [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) library of Ilya Kostrikov and uses some data structures from FAIR's [habitat-lab](https://github.com/facebookresearch/habitat-lab). We would like to thank Dustin Schwenk for his help for the public release of the framework.

## License
AllenAct is MIT licensed, as found in the [LICENSE](LICENSE) file.

## Team
AllenAct is an open-source project built by members of the PRIOR research group at the Allen Institute for Artificial Intelligence (AI2). 

<div align="left">
    <a href="//prior.allenai.org/" target="_blank">
        <img src="docs/img/ai2-prior.svg" width="400">
    </a>
    <br>
</div>

## Citation
If you use this work, please cite our [paper](https://arxiv.org/abs/2008.12760):

```bibtex
@article{AllenAct,
  author = {Luca Weihs and Jordi Salvador and Klemen Kotar and Unnat Jain and Kuo-Hao Zeng and Roozbeh Mottaghi and Aniruddha Kembhavi},
  title = {AllenAct: A Framework for Embodied AI Research},
  year = {2020},
  journal = {arXiv preprint arXiv:2008.12760},
}
```


