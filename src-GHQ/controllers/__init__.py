from .basic_controller import BasicMAC
from .central_basic_controller import CentralBasicMAC
from .hetero_latent_controller import HeteroLatentMAC
from .separate_controller import SeparateMAC


REGISTRY = {"basic_mac": BasicMAC, "central_basic_mac": CentralBasicMAC,
            'hetero_latent_mac': HeteroLatentMAC, "separate_mac": SeparateMAC}
