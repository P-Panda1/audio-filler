import torch
import torch.nn as nn
import ast  # For safely parsing tuples written as strings


class ConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        def to_tuple_if_needed(x):
            """
            Converts YAML strings like '(3, 5)' or lists [3,5] to proper tuples.
            If x is already int or tuple, returns it unchanged.
            """
            if isinstance(x, str):
                try:
                    return tuple(ast.literal_eval(x))
                except Exception:
                    raise ValueError(f"Invalid tuple format: {x}")
            elif isinstance(x, list):
                return tuple(x)
            return x

        in_c = config["in_channels"]
        out_c = config["out_channels"]
        k = to_tuple_if_needed(config.get("kernel_size", 3))
        s = to_tuple_if_needed(config.get("stride", 1))
        p = to_tuple_if_needed(config.get("padding", 0))
        use_bn = config.get("batch_norm", False)
        act_name = config.get("activation", "ReLU")

        layers = [nn.Conv2d(in_c, out_c, kernel_size=k,
                            stride=s, padding=p, bias=not use_bn)]

        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))

        if act_name:
            act_layer = getattr(nn, act_name, None)
            if act_layer is None:
                raise ValueError(f"Unknown activation: {act_name}")
            layers.append(act_layer())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
