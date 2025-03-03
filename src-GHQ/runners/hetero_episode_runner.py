from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import logging
import shutil


class HeteroEpisodeRunner:
    def __init__(self, args, logger):
        self.args = args  # 初始化时的args缺少了大量重要参数！需要在run()内补充完整！
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0  # n_step
        self.t_env = 0
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.special_agent_id = []  # medivac or marauder
        self.set_special_agent_id()

        # Log the first run
        self.log_train_stats_t = -1000000
        self.log_train_stats_t_model = -1000000

        self.verbose = args.verbose

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def set_special_agent_id(self):
        self.env.reset()
        assert "MM" in self.env.map_type, 'Map_Type Error!'
        env_info = self.get_env_info()
        n_agents = env_info["n_agents"]
        for agent_id in range(n_agents):
            unit = self.env.agents[agent_id]
            print('agent_id: %d, unit_type: %s, health_max: %s' % (agent_id, unit.unit_type, unit.health_max))
            # ID序号和地图创建单位的顺序无关，是编辑器兵种单位的序号，按照字母顺序排列！
            if unit.unit_type == self.env.medivac_id and unit.is_flying:
                self.special_agent_id.append(agent_id)  # medivac

        if self.env.map_type == "MM" or self.env.map_type == "MMM":
            assert len(self.special_agent_id) == self.args.n_medivacs, 'args.n_medivacs Error!'

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self, replay_dir):
        self.env.replay_dir = replay_dir
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def get_avail_actions(self):
        avail_actions = self.env.get_avail_actions()
        agent_avail_actions, spec_avail_actions = [], []
        if self.env.map_type == "MM" or self.env.map_type == "MMM":  # medivac
            for i in range(self.args.n_agents):
                if i not in self.special_agent_id:
                    agent_avail_actions.append(avail_actions[i])
                else:
                    medivac_avail_action = self.env.get_avail_agent_actions(i)
                    spec_avail_actions.append(medivac_avail_action[:self.args.n_special_actions])
        elif self.env.map_type == "MMT":  # marauder
            for i in range(self.args.n_agents):
                if i not in self.special_agent_id:
                    agent_avail_actions.append(avail_actions[i])
                else:
                    spec_avail_actions.append(avail_actions[i])
        return agent_avail_actions, spec_avail_actions

    def get_obs(self):
        obs = self.env.get_obs()
        agent_obs, spec_obs = [], []
        for i in range(self.args.n_agents):
            if i not in self.special_agent_id:
                agent_obs.append(obs[i])
            else:
                spec_obs.append(self.env.get_obs_agent(i))
        return agent_obs, spec_obs

    def run(self, test_mode=False, t_episode=0):
        self.reset()
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        if self.args.mac == "hetero_latent_mac":
            self.mac.init_latent()
        replay_data = []

        if self.verbose:
            if ((t_episode % 10) < 2) and test_mode:  # plot
                save_path = os.path.join(self.args.local_results_path, "pic_replays", self.args.unique_token, str(t_episode))
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                os.makedirs(save_path)
                role_color = np.array(['r', 'y', 'b', 'c', 'm', 'g'])
                logging.getLogger('matplotlib.font_manager').disabled = True

        while not terminated:
            agent_avail_actions, spec_avail_actions = self.get_avail_actions()
            agent_obs, spec_obs = self.get_obs()
            pre_transition_data = None
            if self.env.map_type == "MMM" or self.env.map_type == "MM" or self.env.map_type == "MMT":
                pre_transition_data = {
                    "state": np.array(self.env.get_state()),
                    "avail_actions": np.array(agent_avail_actions),
                    "special_avail_actions": np.array(spec_avail_actions),
                    "obs": np.array(agent_obs),
                    "special_obs": np.array(spec_obs)
                }
            if self.verbose:
                # These outputs are designed for SMAC
                ally_info, enemy_info = self.env.get_structured_state()  # plot
                replay_data.append([ally_info, enemy_info])

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            agent_actions, spec_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            actions = None
            if self.env.map_type == "MMM" or self.env.map_type == "MM":  # marine+medivac
                actions = torch.cat((agent_actions, spec_actions), dim=1)
            elif self.env.map_type == "MMT":  # marauder+marine
                actions = torch.cat((spec_actions, agent_actions), dim=1)

            if self.verbose:
                if ((t_episode % 10) < 2) and test_mode:  # plot
                    ally_info = replay_data[-1][0]

                    figure = plt.figure()
                    ally_health = ally_info['health']
                    ally_health_max = ally_info['health_max']
                    if 'shield' in ally_info.keys():
                        ally_health += ally_info['shield']
                        ally_health_max += ally_info['shield_max']
                    ally_health_status = ally_health / ally_health_max
                    for agent_i in range(self.args.n_agents):
                        plt.text(ally_info['x'][agent_i], ally_info['y'][agent_i], '{:d}'.format(agent_i+1), c='y')

                    enemy_info = replay_data[-1][1]
                    enemy_health = enemy_info['health']
                    enemy_health_max = enemy_info['health_max']
                    if 'shield' in enemy_info.keys():
                        enemy_health += enemy_info['shield']
                        enemy_health_max += enemy_info['shield_max']
                    enemy_health_status = enemy_health / enemy_health_max
                    plt.scatter(enemy_info['x'], enemy_info['y'], s=20*enemy_health_status, c='k')
                    for enemy_i in range(len(enemy_info['x'])):
                        plt.text(enemy_info['x'][enemy_i], enemy_info['y'][enemy_i], '{:d}'.format(enemy_i+1))

                    plt.xlim(0, 32)
                    plt.ylim(0, 32)
                    plt.title('t={:d}'.format(self.t))
                    pic_name = os.path.join(save_path, str(self.t) + '.png')
                    plt.savefig(pic_name)
                    plt.close()

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": agent_actions,
                "special_actions": spec_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1

        agent_avail_actions, spec_avail_actions = self.get_avail_actions()
        agent_obs, spec_obs = self.get_obs()
        last_data = {
            "state": np.array(self.env.get_state()),
            "avail_actions": np.array(agent_avail_actions),
            "special_avail_actions": np.array(spec_avail_actions),
            "obs": np.array(agent_obs),
            "special_obs": np.array(spec_obs)
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        agent_actions, spec_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": agent_actions, "special_actions": spec_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        t = self.t_env
        self.logger.log_stat(prefix + "return_mean", np.mean(returns).item(), t)
        self.logger.log_stat(prefix + "return_std", np.std(returns).item(), t)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v/stats["n_episodes"], t)
            if k == 'battle_won':
                key = prefix + k + "_mean"
                battle_won = np.array(self.logger.sacred_info[key])
                if battle_won.shape != (0, ):
                    save_path = os.path.join(self.args.test_result_path, prefix + k)
                    np.save(save_path, battle_won)
                    self.logger.console_logger.info("Saving prefix_battle_won.npy to {}".format(save_path))
                else:
                    raise ValueError('runner._log: logger.sacred_info damaged!')
        stats.clear()
