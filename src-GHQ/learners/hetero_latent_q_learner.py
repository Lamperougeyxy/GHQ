import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.hetero_qmix import HeteroQMixer
from .q_learner import QLearner
import torch as th
from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Normal, kl_divergence
# import time
# import numpy as np
# import matplotlib.pyplot as plt


class HeteroLatentQLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super(HeteroLatentQLearner, self).__init__(mac, scheme, logger, args)
        assert 'hetero' in args.name, 'Illegal Usage of HeteroLatentQLearner'
        self.args = args
        self.mac = mac
        self.logger = logger
        self.latent_out_dim = args.latent_dim  # Compute KL_Loss
        if self.args.MI_Ablation or self.args.MI_Disable:
            self.args.kl_loss_weight = 0.0
        self.agent_latent_infer = th.rand((self.args.n_agents - self.args.n_specialists), args.latent_dim * 2)
        self.spec_latent_infer = th.rand(self.args.n_specialists, args.latent_dim * 2)

        agent_params, spec_params = mac.parameters()
        self.params = list(agent_params)
        self.spec_params = list(spec_params)
        self.last_target_update_episode = 0
        self.spec_last_target_update_episode = 0

        assert not args.universal_qmix, "Unsupported feature args.universal_qmix"
        assert 'hetero' in args.mixer, 'Illegal args.mixer'
        self.mixer = HeteroQMixer(args, mixer_type=0)  # marines
        self.spec_mixer = HeteroQMixer(args, mixer_type=1)  # medivacs
        self.params += list(self.mixer.parameters())
        self.spec_params += list(self.spec_mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)
        self.spec_target_mixer = copy.deepcopy(self.spec_mixer)

        self.lr_scheduler = None
        if args.adam_lr > 0:
            self.optimiser = Adam(params=self.params, lr=args.adam_lr)
            self.spec_optimiser = Adam(params=self.spec_params, lr=args.adam_lr)
            if args.lr_episode_size > 0:
                self.lr_scheduler = StepLR(self.optimiser, args.lr_episode_size, gamma=args.lr_scheduler_gamma)
                self.spec_lr_scheduler = StepLR(self.spec_optimiser, args.lr_episode_size, gamma=args.lr_scheduler_gamma)
        else:  # Default optimizer, args.lr=0.0005
            assert (args.adam_lr == -1), 'Should Set args.adam_lr=-1 to use Default Settings'
            args.lr_episode_size = -1
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.spec_optimiser = RMSprop(params=self.spec_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.spec_log_stats_t = -self.args.learner_log_interval - 1

    def uni_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        spec_actions = batch["special_actions"][:, :-1]
        avail_actions = batch["avail_actions"]
        spec_avail_actions = batch["special_avail_actions"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Calculate estimated Q-Values
        spec_mac_out = []
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        spec_var_mean, spec_latent = self.mac.special_agent.init_latent()
        var_mean, latent = self.mac.agent.init_latent()
        KL_loss = th.tensor(0.0).to(self.args.device)
        spec_KL_loss = th.tensor(0.0).to(self.args.device)

        for t in range(batch.max_seq_length):
            if not self.args.MI_Disable:  # Compute KL_Loss
                agent_outs, agent_input, agent_rnn_hidden, agent_embed, agent_latent, spec_outs, spec_input, \
                    spec_rnn_hidden, spec_embed, spec_latent = self.mac.forward(batch, t=t, train_mode=True)
                mac_out.append(agent_outs)  # [t,(bs,n,n_actions)]
                spec_mac_out.append(spec_outs)  # [t,(bs,n,n_actions)]
                spec_latent = spec_latent.reshape(batch.batch_size, -1)
                agent_latent = agent_latent.reshape(batch.batch_size, -1)
                if self.args.input_latent:  # obs_i
                    agent_input = agent_input.reshape(batch.batch_size, -1)
                    spec_input = spec_input.reshape(batch.batch_size, -1)
                    agent_infer_input = th.cat([spec_latent.clone().detach(), agent_input], dim=1)
                    spec_infer_input = th.cat([agent_latent.clone().detach(), spec_input], dim=1)
                else:  # rnn_i
                    agent_rnn_hidden = agent_rnn_hidden.reshape(batch.batch_size, -1)
                    spec_rnn_hidden = spec_rnn_hidden.reshape(batch.batch_size, -1)
                    agent_infer_input = th.cat([spec_latent.clone().detach(), agent_rnn_hidden], dim=1)
                    spec_infer_input = th.cat([agent_latent.clone().detach(), spec_rnn_hidden], dim=1)

                self.agent_latent_infer = self.mac.agent.inference_net(agent_infer_input)
                self.agent_latent_infer[:, -self.latent_out_dim:] = th.clamp(
                    th.exp(self.agent_latent_infer[:, -self.latent_out_dim:]), min=self.args.var_floor)  # ([32, 32])
                agent_gaussian_infer = Normal(self.agent_latent_infer[:, :self.latent_out_dim],
                                              (self.agent_latent_infer[:, self.latent_out_dim:]) ** (1 / 2))
                loss_kl = kl_divergence(agent_embed, agent_gaussian_infer).sum(dim=-1).mean()
                KL_loss += loss_kl

                self.spec_latent_infer = self.mac.special_agent.inference_net(spec_infer_input)
                self.spec_latent_infer[:, -self.latent_out_dim:] = th.clamp(
                    th.exp(self.spec_latent_infer[:, -self.latent_out_dim:]), min=self.args.var_floor)
                spec_gaussian_infer = Normal(self.spec_latent_infer[:, :self.latent_out_dim],
                                             (self.spec_latent_infer[:, self.latent_out_dim:]) ** (1 / 2))
                spec_loss_kl = kl_divergence(spec_embed, spec_gaussian_infer).sum(dim=-1).mean()
                spec_KL_loss += spec_loss_kl
            else:
                agent_outs, _, _, _, _, spec_outs, _, _, _, _ = self.mac.forward(batch, t=t, train_mode=True)
                mac_out.append(agent_outs)  # [t,(bs,n,n_actions)]
                spec_mac_out.append(spec_outs)  # [t,(bs,n,n_actions)]

        KL_loss /= batch.max_seq_length
        spec_KL_loss /= batch.max_seq_length

        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        target_mac_out = []
        spec_mac_out = th.stack(spec_mac_out, dim=1)
        spec_chosen_action_qvals = th.gather(spec_mac_out[:, :-1], dim=3, index=spec_actions).squeeze(3)
        spec_target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        self.target_mac.init_latent()
        for t in range(batch.max_seq_length):
            target_agent_outs, _, _, _, _, target_spec_outs, _, _, _, _ = self.target_mac.forward(batch, t=t, train_mode=True)
            target_mac_out.append(target_agent_outs)  # [t,(bs,n,n_actions)]
            spec_target_mac_out.append(target_spec_outs)  # [t,(bs,n,n_actions)]

        target_mac_out = th.stack(target_mac_out[1:], dim=1)
        spec_target_mac_out = th.stack(spec_target_mac_out[1:], dim=1)
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        spec_target_mac_out[spec_avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            spec_mac_out_detach = spec_mac_out.clone().detach()
            spec_mac_out_detach[spec_avail_actions == 0] = -9999999
            spec_cur_max_actions = spec_mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            spec_target_max_qvals = th.gather(spec_target_mac_out, 3, spec_cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
            spec_target_max_qvals = spec_target_mac_out.max(dim=3)[0]

        # Mix
        # if self.mixer is not None:
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
        spec_chosen_action_qvals = self.spec_mixer(spec_chosen_action_qvals, batch["state"][:, :-1])
        spec_target_max_qvals = self.spec_target_mixer(spec_target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning spec_targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        spec_targets = rewards + self.args.gamma * (1 - terminated) * spec_target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())  # no gradient through target net
        spec_td_error = (spec_chosen_action_qvals - spec_targets.detach())  # no gradient through target net
        # (bs,t,1)
        mask = mask.clone().expand_as(td_error)
        spec_mask = mask.expand_as(spec_td_error)
        # 0-out the spec_targets that came from padded data
        masked_td_error = td_error * mask
        spec_masked_td_error = spec_td_error * spec_mask

        # Normal L2 spec_loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        spec_loss = (spec_masked_td_error ** 2).sum() / spec_mask.sum()
        if not self.args.MI_Disable:  # Compute KL_Loss
            loss += self.args.kl_loss_weight * KL_loss  # default kl_loss_weight=1
            spec_loss += self.args.kl_loss_weight * spec_KL_loss

        # Optimise
        self.optimiser.zero_grad()
        self.spec_optimiser.zero_grad()
        loss.backward()
        spec_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)  # max_norm
        spec_grad_norm = th.nn.utils.clip_grad_norm_(self.spec_params, self.args.grad_norm_clip)  # max_norm
        self.optimiser.step()
        self.spec_optimiser.step()
        if self.args.lr_episode_size > 0 and self.args.adam_lr > 0:
            self.lr_scheduler.step()
            self.spec_lr_scheduler.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self._update_spec_targets()
            self.last_target_update_episode = episode_num
            if self.args.lr_episode_size > 0 and self.args.adam_lr > 0:
                print('-----' * 5)
                print('LR_scheduler Working! Marine_lr=%.7f, Specialist_lr=%.7f' %
                      (self.lr_scheduler.get_lr()[0], self.spec_lr_scheduler.get_lr()[0]))
                print('-----' * 5)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_total", loss.item(), t_env)
            self.logger.log_stat("w * loss_KL", self.args.kl_loss_weight * KL_loss.item(), t_env)
            self.logger.log_stat("var_mean", var_mean.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (
                    mask_elems * (self.args.n_agents - self.args.n_specialists)), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (
                    mask_elems * (self.args.n_agents - self.args.n_specialists)), t_env)
            self.logger.log_stat("spec_loss_total", spec_loss.item(), t_env)
            self.logger.log_stat("w * spec_loss_KL", self.args.kl_loss_weight * spec_KL_loss.item(), t_env)
            self.logger.log_stat("spec_var_mean", spec_var_mean.item(), t_env)
            self.logger.log_stat("spec_grad_norm", spec_grad_norm.item(), t_env)
            self.logger.log_stat("spec_td_error_abs", (spec_masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("spec_q_taken_mean", (spec_chosen_action_qvals * mask).sum().item() / (
                    mask_elems * self.args.n_specialists), t_env)
            self.logger.log_stat("spec_target_mean", (spec_targets * mask).sum().item() / (
                    mask_elems * self.args.n_specialists), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _update_spec_targets(self):
        if not self.args.universal_training:
            self.target_mac.load_state(self.mac)
        # assert self.spec_mixer is not None
        self.spec_target_mixer.load_state_dict(self.spec_mixer.state_dict())
        self.logger.console_logger.info("Updated spec_target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        assert self.mixer is not None
        self.mixer.cuda()
        self.target_mixer.cuda()
        if not self.args.universal_qmix:
            self.spec_mixer.cuda()
            self.spec_target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
            if not self.args.universal_qmix:
                th.save(self.spec_mixer.state_dict(), "{}/spec_mixer.th".format(path))
                th.save(self.spec_optimiser.state_dict(), "{}/spec_opt.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            if not self.args.universal_qmix:
                self.spec_mixer.load_state_dict(
                    th.load("{}/spec_mixer.th".format(path), map_location=lambda storage, loc: storage))
                self.spec_optimiser.load_state_dict(
                    th.load("{}/spec_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        var_mean, latent = self.mac.agent.init_latent()

        KL_loss = th.tensor(0.0).to(self.args.device)
        for t in range(batch.max_seq_length):
            agent_outs, agent_input, agent_embed, _, _, _, _, med_latent = self.mac.forward(batch, t=t, train_mode=True)
            if not self.args.MI_Disable:  # Compute KL_Loss
                med_latent = med_latent.reshape(batch.batch_size, -1)
                agent_input = agent_input.reshape(batch.batch_size, -1)
                agent_infer_input = th.cat([med_latent.clone().detach(), agent_input], dim=1)
                self.agent_latent_infer = self.mac.agent.inference_net(agent_infer_input)
                self.agent_latent_infer[:, -self.latent_out_dim:] = th.clamp(
                    th.exp(self.agent_latent_infer[:, -self.latent_out_dim:]), min=self.args.var_floor)  # ([32, 32])
                agent_gaussian_infer = Normal(self.agent_latent_infer[:, :self.latent_out_dim],
                                              (self.agent_latent_infer[:, self.latent_out_dim:]) ** (1 / 2))
                loss_kl = kl_divergence(agent_embed, agent_gaussian_infer).sum(dim=-1).mean() * self.args.kl_loss_weight
                KL_loss += loss_kl
            mac_out.append(agent_outs)  # [t,(bs,n,n_actions)]

        KL_loss /= batch.max_seq_length
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # (bs,t,n) Q value of an action

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)  # (bs,n,hidden_size)
        self.target_mac.init_latent()  # (bs,n,latent_size)

        for t in range(batch.max_seq_length):
            target_agent_outs, _, _, _, _, _, _, _ = self.target_mac.forward(batch, t=t, train_mode=True)  # (bs,n,n_actions), (bs,n,latent_out_dim)
            target_mac_out.append(target_agent_outs)  # [t,(bs,n,n_actions)]

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time, dim=1 is time index
        # (bs,t,n,n_actions)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # Q values

        # Max over target Q-Values
        if self.args.double_q:  # True for QMix
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()  # return a new Tensor, detached from the current graph
            mac_out_detach[avail_actions == 0] = -9999999
            # (bs,t,n,n_actions), discard t=0
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]  # indices instead of values
            # (bs,t,n,1)
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            # (bs,t,n,n_actions) ==> (bs,t,n,1) ==> (bs,t,n) max target-Q
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        # if self.mixer is not None:
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])  # (bs,t,1)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())  # no gradient through target net
        # (bs,t,1)
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        loss += KL_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)  # max_norm
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_total", loss.item(), t_env)
            self.logger.log_stat("loss_KL", KL_loss.item(), t_env)
            self.logger.log_stat("var_mean", var_mean.item(), t_env)

            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (
                    mask_elems * (self.args.n_agents - self.args.n_medivacs)), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (
                    mask_elems * (self.args.n_agents - self.args.n_medivacs)), t_env)
            if self.args.use_tensorboard:
                pass
            self.log_stats_t = t_env
