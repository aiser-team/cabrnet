import torch.nn as nn


# Layer initialisation functions
def init_weights_xavier(module: nn.Module):
    r"""Xavier initialization (used in ProtoTree).

    Args:
        module (module): Target module.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("sigmoid"))


def init_weights_kaiming_normal(module: nn.Module):
    r"""Kaiming initialization (used in ProtoPNet).

    Args:
        module (module): Target module.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, val=0)


def init_weights_batchnorm_identity(module: nn.Module):
    r"""Batch-norm initialization (used in ProtoPNet).

    Args:
        module (module): Target module.
    """
    if isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, val=1)
        nn.init.constant_(module.bias, val=0)


def init_weights_protopnet(module: nn.Module):
    r"""Weight initialization used in ProtoPNet.

    Args:
        module (module): Target module.
    """
    init_weights_kaiming_normal(module)
    init_weights_batchnorm_identity(module)


layer_init_functions = {
    "XAVIER": init_weights_xavier,
    "PROTOPNET": init_weights_protopnet,
}
