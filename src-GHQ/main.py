try:
    # until python 3.10
    from collections import Mapping
except:
    # from python 3.10
    from collections.abc import Mapping
import numpy as np
import os
from os.path import dirname, abspath
from copy import deepcopy
from datetime import datetime
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run
from hetero_run import hetero_run

# SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
SETTINGS['CAPTURE_MODE'] = "no"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    assert config['name'] is not None, "config['name'] is empty!"
    if 'hetero' in config['name']:
        print('-----hetero_run----')
        hetero_run(_run, config, _log)
    else:
        print('-----qmix_run-----')
        run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)  # TODO
    # params = ['main.py', '--config=hetero_qmix_latent', '--env-config=sc2', 'with', 'env_args.map_name=MMM2', 'n_medivacs=1', 't_max=5050000']

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict['unique_token'] = "{}__{}".format(config_dict['name'], datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # now add all the config to sacred

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred/map_name/unique_token.")
    results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
    map_name = None
    for _, temp in enumerate(params):
        if temp.split("=")[0] == "env_args.map_name":
            map_name = temp.split("=")[1]
            break
    try:
        if map_name is None:
            map_name = 'default'
            raise Warning("Command_params should include 'env_args.map_name'!")
    except Warning as w:
        print('DEFAULT: ', repr(w))
    test_result_path = os.path.join(results_path, "sacred", map_name, config_dict['unique_token'])
    config_dict['test_result_path'] = test_result_path
    print('Test_Result_Path: ', test_result_path)
    ex.add_config(config_dict)
    ex.observers.append(FileStorageObserver.create(test_result_path))

    ex.run_commandline(params)

