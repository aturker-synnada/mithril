# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

from mithril import IOKey
from mithril.models import Add, Convolution2D, Flatten, Linear, Model, Relu


def basic_block(
    out_channels: int, stride: int = 1, downsample: Model | None = None
) -> Model:
    block = Model()
    block += Convolution2D(
        kernel_size=3, out_channels=out_channels, padding=1, stride=stride
    )
    model_input = block.cin
    block += Relu()
    block += Convolution2D(kernel_size=3, out_channels=out_channels, padding=1)
    skip_in = block.cout

    if downsample is not None:
        block |= downsample.connect(input=model_input)
        block |= Add().connect(left=downsample.cout, right=skip_in)
    else:
        block |= Add().connect(left=model_input, right=skip_in)

    block += Relu()
    return block


def bottleneck(
    out_channels: int, stride: int = 1, downsample: Model | None = None
) -> Model:
    model = Model()
    model += Convolution2D(kernel_size=1, out_channels=out_channels, stride=stride)
    model += Relu()
    model += Convolution2D(
        kernel_size=3, out_channels=out_channels, padding=1, stride=stride
    )
    model += Relu()
    model += Convolution2D(kernel_size=1, out_channels=out_channels, stride=stride)
    skip_in = model.cout

    if downsample is not None:
        model |= downsample.connect(input=model.cin)
        model |= Add().connect(left=downsample.cout, right=skip_in)
    else:
        model |= Add().connect(left=model.cin, right=skip_in)

    model += Relu()
    return model


def make_layer(
    out_channels: int, block: Callable, n_blocks: int = 2, stride: int = 1
) -> Model:
    layer = Model()
    downsample = Convolution2D(kernel_size=1, out_channels=out_channels, stride=stride)
    layer += block(out_channels, stride, downsample)
    for _ in range(n_blocks - 1):
        layer += block(out_channels)
    return layer


def resnet(n_classes: int, block: Callable, layers: list[int]) -> Model:
    resnet = Model()
    resnet += Convolution2D(kernel_size=7, out_channels=64, stride=2, padding=3)
    resnet += make_layer(64, block, n_blocks=layers[0], stride=1)
    resnet += make_layer(128, block, n_blocks=layers[1], stride=2)
    resnet += make_layer(256, block, n_blocks=layers[2], stride=2)
    resnet += make_layer(512, block, n_blocks=layers[3], stride=2)
    resnet += Flatten(start_dim=1)

    resnet |= Linear(dimension=n_classes).connect(
        input=resnet.cout, output=IOKey(name="output")
    )
    return resnet


def resnet18(n_classes: int):
    return resnet(n_classes=n_classes, block=basic_block, layers=[2, 2, 2, 2])


def resnet34(n_classes: int):
    return resnet(n_classes=n_classes, block=basic_block, layers=[2, 4, 6, 3])


def resnet50(n_classes: int):
    return resnet(n_classes=n_classes, block=bottleneck, layers=[2, 4, 6, 3])


def resnet101(n_classes: int):
    return resnet(n_classes=n_classes, block=bottleneck, layers=[3, 4, 23, 3])


def resnet152(n_classes: int):
    return resnet(n_classes=n_classes, block=bottleneck, layers=[3, 8, 36, 3])


resnet_model = resnet18(10)
