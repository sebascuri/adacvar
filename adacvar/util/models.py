"""Implementation of different NN models."""
import torch.nn as nn

__author__ = "Sebastian Curi"
__all__ = ["LinearNet", "ReLUNet"]


def _num_flat_features(in_tensor):
    sizes = in_tensor.size()[1:]  # The first index is the batch size.
    num_features = 1
    for s in sizes:
        num_features *= s
    return num_features


def _update_shape(shape, padding=0, kernel_size=1, dilation=1, stride=1):
    num_dim = len(shape)
    if type(padding) == int:
        padding = [padding] * num_dim
    if type(kernel_size) == int:
        kernel_size = [kernel_size] * num_dim
    if type(dilation) == int:
        dilation = [dilation] * num_dim
    if type(stride) == int:
        stride = [stride] * num_dim

    out_shape = [0] * num_dim
    for idx in range(num_dim):
        out_shape[idx] = int(
            (shape[idx] + 2 * padding[idx] - dilation[idx] * (kernel_size[idx] - 1) - 1)
            / stride[idx]
            + 1
        )

    return out_shape


class View(nn.Module):
    """Module that implements a view operation."""

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        return x


class AbstractNetwork(nn.Module):
    """Abstract Neural Network.

    Parameters
    ----------
    name: str
        name of the implementation.
    in_channels: int
        number of channels
    in_features: tuple of int
        number of features
    out_features: int
        number of outputs.
    """

    def __init__(self, name, in_channels, in_features, out_features):
        super(AbstractNetwork, self).__init__()
        self.name = name
        self._in_channels = in_channels
        self._in_features = in_features
        self._out_features = out_features
        self._layers = nn.ModuleList()

    def _make_layers(self, config):
        layers = nn.ModuleList()
        if self._in_channels == 0:
            in_features = self._in_features[0]
        else:
            view_features = [*self._in_features]
            in_channels = self._in_channels

        for layer in config:
            if layer["name"] == "MaxPool2d":
                layers += [
                    nn.MaxPool2d(
                        kernel_size=layer.get("kernel_size", 1),
                        padding=layer.get("padding", 0),
                        dilation=layer.get("dilation", 1),
                        stride=layer.get("stride", 1),
                    )
                ]
                view_features = _update_shape(
                    view_features,
                    kernel_size=layer.get("kernel_size", 1),
                    padding=layer.get("padding", 0),
                    dilation=layer.get("dilation", 1),
                    stride=layer.get("stride", 1),
                )
            elif layer["name"] == "AvgPool2d":
                layers += [
                    nn.AvgPool2d(
                        kernel_size=layer.get("kernel_size", 1),
                        padding=layer.get("padding", 0),
                        stride=layer.get("stride", 1),
                    )
                ]
                view_features = _update_shape(
                    view_features,
                    kernel_size=layer.get("kernel_size", 1),
                    padding=layer.get("padding", 0),
                    stride=layer.get("stride", 1),
                )
            elif layer["name"] == "Conv2d":
                layers += [
                    nn.Conv2d(
                        in_channels,
                        layer["out_channels"],
                        kernel_size=layer.get("kernel_size", 1),
                        padding=layer.get("padding", 0),
                        dilation=layer.get("dilation", 1),
                        stride=layer.get("stride", 1),
                    )
                ]
                view_features = _update_shape(
                    view_features,
                    kernel_size=layer.get("kernel_size", 1),
                    padding=layer.get("padding", 0),
                    dilation=layer.get("dilation", 1),
                    stride=layer.get("stride", 1),
                )

                in_channels = layer["out_channels"]
            elif layer["name"] == "Fully Connected":
                layers += [
                    nn.Linear(
                        in_features=in_features, out_features=layer["out_features"]
                    )
                ]
                in_features = layer["out_features"]
            elif layer["name"] == "ReLU":
                layers += [nn.ReLU(inplace=True)]
            elif layer["name"] == "BatchNorm2d":
                layers += [nn.BatchNorm2d(num_features=layer["num_features"])]
            elif layer["name"] == "Dropout2d":
                layers += [nn.Dropout2d(p=layer.get("p", 0.5))]
            elif layer["name"] == "View":
                layers += [View()]
                size = 1
                for dim_size in view_features:
                    size *= dim_size
                in_features = size * in_channels
            else:
                raise ValueError("Did not parse {} correctly".format(layer["name"]))

        self._layers = layers
        self._classifier = nn.Linear(
            in_features=in_features, out_features=self._out_features
        )

    def forward(self, x, training=None):
        if training is None:
            training = self.training
        for layer in self._layers:
            if not ("dropout" in str(type(layer))) or training:
                x = layer(x)

        x = self._classifier(x)
        if x.shape[1] == 1:  # This is needed for 1-D outputs.
            x = x[:, 0]

        return x


class LinearNet(AbstractNetwork):
    """Implementation of a linear neural network."""

    def __init__(self, name, in_channels, in_features, out_features):
        super(LinearNet, self).__init__(name, in_channels, in_features, out_features)
        configuration = []

        if in_channels > 0:
            view_features = [*in_features]
            size = 1
            for dim_size in view_features:
                size *= dim_size
            configuration += [{"name": "View", "out_features": size * in_channels}]

        self._make_layers(configuration)


class ReLUNet(AbstractNetwork):
    """Implementation of a neural network with ReLU activations."""

    implementations = {
        "C10C20L50": [("C", 10), ("D",), ("C", 20), ("D",), ("FC", 50)],
        "FC320FC50": [("FC", 320), ("FC", 50)],
        "FC20FC20": [("FC", 20), ("FC", 20)],
        "LeNet-5": [("C", 6), ("C", 16), ("FC", 120), ("FC", 84)],
        "LeNet-5d": [("C", 6), ("C", 16), ("FC", 120), ("D",), ("FC", 84), ("D",)],
    }

    def __init__(self, name, in_channels, in_features, out_features):
        super(ReLUNet, self).__init__(name, in_channels, in_features, out_features)
        configuration = []
        if in_channels > 0:
            view_features = [*in_features]
        in_channels = in_channels

        for layer in self.implementations[name]:
            if layer[0] == "C":
                configuration += [
                    {"name": "Conv2d", "out_channels": layer[1], "kernel_size": 5},
                    {"name": "MaxPool2d", "kernel_size": 2},
                    {"name": "ReLU"},
                ]
                in_channels = layer[1]
                view_features = _update_shape(view_features, kernel_size=5)
                view_features = _update_shape(view_features, kernel_size=2)

            elif layer[0] == "D":
                configuration += [{"name": "Dropout2d"}]
            elif layer[0] == "FC":
                if in_channels > 0:  # Add a View Layer
                    size = 1
                    for dim_size in view_features:
                        size *= dim_size
                    configuration += [
                        {"name": "View", "out_features": size * in_channels}
                    ]
                    in_channels = 0

                configuration += [
                    {"name": "Fully Connected", "out_features": layer[1]},
                    {"name": "ReLU"},
                ]
            else:
                raise ValueError("Layer name {} not parsed correctly".format(layer[0]))

        self._make_layers(configuration)
