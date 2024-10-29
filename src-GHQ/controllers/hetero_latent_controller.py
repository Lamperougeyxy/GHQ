from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th


class HeteroLatentMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(HeteroLatentMAC, self).__init__(scheme, groups, args)
        self.n_agents = args.n_agents  # Total Agent Num!!!
        self.n_specialists = args.n_specialists
        print('controller.n_agents: %d, n_specialists: %d' % (self.n_agents, self.n_specialists))
        # 核心差异是obs，所以在runner.get_obs()处将默认的obs拆分成2组不同的。action维度差异由agent.n_action的差异给定。
        self.args = args
        self.agent_input_shape, self.spec_input_shape = self._get_input_shape(scheme)
        self._build_agents(self.agent_input_shape, self.spec_input_shape)
        # self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None
        self.spec_hidden_states = None
        print('controller.agent_input_shape: ', self.agent_input_shape)  # 42=30+3+9, obs_shape+ n_agents+ n_actions
        print('controller.spec_input_shape: ', self.spec_input_shape)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        spec_avail_actions = ep_batch["special_avail_actions"][:, t_ep]
        agent_outputs, _, _, _, _, spec_outputs, _, _, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        agent_chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        spec_chosen_actions = self.action_selector.select_action(spec_outputs[bs], spec_avail_actions[bs], t_env,
                                                                 test_mode=test_mode)
        return agent_chosen_actions, spec_chosen_actions

    def forward(self, ep_batch, t, train_mode=False, test_mode=False):
        batch_size = ep_batch.batch_size
        agent_inputs, spec_inputs = self._build_inputs(ep_batch, t)  # (bs*n,(obs+act+id))
        # (bs*n,(obs+act+id)), (bs,n,hidden_size), (bs,n,latent_out_dim)
        agent_outs, self.hidden_states, agent_embed, agent_latent_sample = self.agent.forward(
            agent_inputs, self.hidden_states, t=t, batch_size=batch_size, train_mode=train_mode)
        spec_outs, self.spec_hidden_states, spec_embed, spec_latent_sample = self.special_agent.forward(
            spec_inputs, self.spec_hidden_states, t=t, batch_size=batch_size, train_mode=train_mode)
        return agent_outs.view(ep_batch.batch_size, (self.n_agents - self.n_specialists),
                               -1), agent_inputs, self.hidden_states, agent_embed, agent_latent_sample, spec_outs.view(
            ep_batch.batch_size, self.n_specialists, -1), spec_inputs, self.spec_hidden_states, spec_embed, spec_latent_sample

    def init_hidden(self, batch_size):
        if self.args.use_cuda:
            self.hidden_states = th.zeros(batch_size, (self.n_agents - self.n_specialists),
                                          self.args.rnn_hidden_dim).cuda()
            self.spec_hidden_states = th.zeros(batch_size, self.n_specialists, self.args.rnn_hidden_dim).cuda()
        else:
            self.hidden_states = th.zeros(batch_size, (self.n_agents - self.n_specialists), self.args.rnn_hidden_dim)
            self.spec_hidden_states = th.zeros(batch_size, self.n_specialists, self.args.rnn_hidden_dim)

    def init_latent(self):
        return self.agent.init_latent(), self.special_agent.init_latent()

    def parameters(self):
        return self.agent.parameters(), self.special_agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.special_agent.load_state_dict(other_mac.special_agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.special_agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.special_agent.state_dict(), "{}/special_agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.special_agent.load_state_dict(
            th.load("{}/special_agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape, spec_input_shape=None):
        assert spec_input_shape is not None, 'spec_input_shape is not assigned!'
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args, agent_type=0)  # marine
        self.special_agent = agent_REGISTRY[self.args.agent](spec_input_shape, self.args, agent_type=1)  # specialists

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        spec_inputs = [batch["special_obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
                spec_inputs.append(th.zeros_like(batch["special_actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
                spec_inputs.append(batch["special_actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye((self.n_agents - self.n_specialists), device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            spec_inputs.append(th.eye(self.n_specialists, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs * (self.n_agents - self.n_specialists), -1) for x in inputs], dim=1)
        spec_inputs = th.cat([x.reshape(bs * self.n_specialists, -1) for x in spec_inputs], dim=1)  # (bs*n, obs+act+id)
        if self.args.use_cuda:
            inputs = inputs.cuda()
            spec_inputs = spec_inputs.cuda()
        return inputs, spec_inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        spec_input_shape = scheme["special_obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
            spec_input_shape += scheme["special_actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += (self.n_agents - self.n_specialists)
            spec_input_shape += self.n_specialists
        return input_shape, spec_input_shape
