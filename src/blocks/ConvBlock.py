import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, config):
        """
        Generic convolutional block builder.
        Expects a dict with:
            {
              "in_channels": int,
              "out_channels": int,
              "kernel_size": int or tuple,
              "stride": int or tuple,
              "padding": int or tuple,
              "activation": str,
              "batch_norm": bool
            }
        """
        super().__init__()

        in_c = config["in_channels"]
        out_c = config["out_channels"]
        k = config.get("kernel_size", 3)
        s = config.get("stride", 1)
        p = config.get("padding", 0)
        use_bn = config.get("batch_norm", False)
        act_name = config.get("activation", "ReLU")

        # --- Core convolution ---
        layers = [nn.Conv2d(in_c, out_c, kernel_size=k,
                            stride=s, padding=p, bias=not use_bn)]

        # --- Optional BatchNorm ---
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))

        # --- Activation function ---
        if act_name:
            act_layer = getattr(nn, act_name, None)
            if act_layer is None:
                raise ValueError(f"Unknown activation: {act_name}")
            layers.append(act_layer())

        # --- Combine into sequential ---
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
