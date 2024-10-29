# GHQ
Official simplified implementation for "GHQ: Grouped Hybrid Q Learning for Heterogeneous Cooperative Multi-agent Reinforcement Learning"

# Instructions

This repository is based on an older version of the [PyMARL](https://github.com/oxwhirl/pymarl/tree/master) and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

The results in SMAC (https://arxiv.org/abs/1902.04043) use SC2.4.10.

Installation instructions and other detailed instructions could be found in the [PyMARL](https://github.com/oxwhirl/pymarl/tree/master) repo.

A running command line example is:

```shell
python3 src/main.py main.py --config=hetero_qmix_latent --env-config=sc2 with env_args.map_name=MMM2 n_medivacs=1 t_max=5050000
```

For further support, plese email the authors in the following citation.

## Citation

If you use GHQ in your research, please cite the [GHQ paper](https://arxiv.org/abs/2303.01070).

*Yu X, Lin Y, Wang X, et al. GHQ: grouped hybrid Q-learning for cooperative heterogeneous multi-agent reinforcement learning[J]. Complex & Intelligent Systems, 2024: 1-20.*

In BibTeX format:

```tex
@article{yu2024ghq,
  title={GHQ: grouped hybrid Q-learning for cooperative heterogeneous multi-agent reinforcement learning},
  author={Yu, Xiaoyang and Lin, Youfang and Wang, Xiangsen and Han, Sheng and Lv, Kai},
  journal={Complex \& Intelligent Systems},
  pages={1--20},
  year={2024},
  publisher={Springer}
}
```

## License

All the source code that has been taken from the PyMARL repository was licensed (and remains so) under the Apache License v2.0 (included in LICENSE-Apache License file). Any new code is also licensed under the Apache License v2.0

The original codes in this repository follows the MIT License.
