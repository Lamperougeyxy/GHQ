from .qmix import QMixer
from .vdn import VDNMixer
from .hetero_qmix import HeteroQMixer


REGISTRY = {"qmix": QMixer, 'vdn': VDNMixer, "hetero_qmix": HeteroQMixer}
