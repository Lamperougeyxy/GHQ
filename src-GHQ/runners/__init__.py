from .parallel_runner import ParallelRunner
from .qmix_episode_runner import QmixEpisodeRunner
from .hetero_episode_runner import HeteroEpisodeRunner

REGISTRY = {"parallel": ParallelRunner, "qmix_episode": QmixEpisodeRunner,
            'hetero_episode': HeteroEpisodeRunner, "episode": QmixEpisodeRunner}
