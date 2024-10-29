from .q_learner import QLearner
from .coma_learner import COMALearner
from .hetero_latent_q_learner import HeteroLatentQLearner

REGISTRY = {"q_learner": QLearner, "coma_learner": COMALearner,
            "hetero_latent_q_learner": HeteroLatentQLearner, }
