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

import torch
import torch.nn as nn

import mithril as ml

params = {
    "ch": 128,
    "out_ch": 3,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "in_channels": 3,
    "resolution": 256,
    "z_channels": 16,
}


class SmallCNNForward(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5 = nn.Conv2d(512, 256, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class BigCNNForward(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = SmallCNNForward()
        self.conv2 = SmallCNNForward()
        self.conv3 = SmallCNNForward()
        self.conv4 = SmallCNNForward()
        self.conv5 = SmallCNNForward()
        self.conv6 = SmallCNNForward()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


def run_torch():
    torch_inp = torch.randn(8, 256, 256, 128)
    o_model = BigCNNForward()
    o_model(torch_inp)


def small_cnn_forward():
    model = ml.models.Model()
    model += ml.models.Convolution2D(3, 256, 1, 1)(
        input=ml.IOKey("input", shape=(None, 256, None, None))
    )
    model += ml.models.Convolution2D(3, 512, 1, 1)
    model += ml.models.Convolution2D(3, 512, 1, 1)
    model += ml.models.Convolution2D(3, 512, 1, 1)
    model += ml.models.Convolution2D(3, 256, 1, 1)
    return model


def big_cnn_forward():
    model = ml.models.Model()
    model += small_cnn_forward()(input=ml.IOKey("input", shape=(None, 256, None, None)))
    model += small_cnn_forward()
    model += small_cnn_forward()
    model += small_cnn_forward()
    model += small_cnn_forward()
    model += small_cnn_forward()
    return model


def run_mithril():
    model = big_cnn_forward()
    pm = ml.compile(
        model,
        ml.TorchBackend(),
        jit=False,
        shapes={"input": [8, 256, 256, 128]},
        data_keys={"input"},
        file_path="ekmek.py",
    )
    model_params = pm.randomize_params()
    pm.evaluate(model_params, {"input": torch.randn(8, 256, 256, 128)})


if __name__ == "__main__":
    # Get args from command line and run wrt arg
    import sys

    if len(sys.argv) != 2:
        print("Usage: python mem_test.py torch/mithril")
        sys.exit(1)
    if sys.argv[1] == "torch":
        print("Running torch")
        run_torch()
    elif sys.argv[1] == "mithril":
        print("Running mithril")
        run_mithril()
