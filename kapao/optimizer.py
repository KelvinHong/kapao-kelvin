from torch import nn
from torch.optim import Adam, SGD, lr_scheduler
from .utils import one_cycle

def build_optimizer_schedulers(model, hyp: dict = None, adam: bool = False, linear_lr: bool = False, epochs: int = 300):
    batchnorm_group, weights_group, bias_group = [], [], []
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            bias_group.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            batchnorm_group.append(v.weight)
        elif hasattr(v, "weight") and isinstance(
            v.weight, nn.Parameter
        ):
            weights_group.append(v.weight)

    if adam:
        optimizer = Adam(
            batchnorm_group, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999)
        )
    else:
        optimizer = SGD(batchnorm_group, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

    optimizer.add_param_group(
        {"params": weights_group, "weight_decay": hyp["weight_decay"]}
    )
    optimizer.add_param_group({"params": bias_group})

    # TODO: Remove this for now, not sure what effect it has.
    # del batchnorm_group, weights_group, bias_group

    if linear_lr:
        lr_function = (
            lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp["lrf"]) + hyp["lrf"]
        )
    else:
        lr_function = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_function
    )

    return optimizer, scheduler
