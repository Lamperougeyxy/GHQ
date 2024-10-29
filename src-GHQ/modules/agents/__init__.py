from .rnn_agent import RNNAgent
from .rode_agent import RODEAgent
from .hetero_qmix_agent import HeteroQmixAgent
from .hetero_latent_agent import HeteroLatentAgent
from .central_rnn_agent import CentralRNNAgent
from .roma_agent import ROMAAgent

REGISTRY = {"rnn": RNNAgent, "rode": RODEAgent, "hetero_qmix": HeteroQmixAgent, "hetero_latent": HeteroLatentAgent,
            "central_rnn": CentralRNNAgent, "roma": ROMAAgent}
