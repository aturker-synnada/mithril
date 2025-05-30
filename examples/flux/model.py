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

from dataclasses import dataclass

from examples.flux.layers import (
    double_stream_block,
    embed_nd,
    last_layer,
    mlp_embedder,
    single_stream_block,
    timestep_embedding,
)
from mithril import IOKey
from mithril.models import Add, Concat, Linear, Model


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


def flux(params: FluxParams):
    flux = Model()

    img = IOKey("img", shape=[1, 4096, 64])
    txt = IOKey("txt", shape=[1, 512, 4096])

    img_ids = IOKey("img_ids", shape=[1, 4096, 3])
    txt_ids = IOKey("txt_ids", shape=[1, 512, 3])

    timesteps = IOKey("timesteps", shape=[1])
    y = IOKey("y", shape=[1, 768])

    flux |= timestep_embedding(dim=256).connect(input=timesteps, output="time_embed")
    flux |= mlp_embedder(params.hidden_size, name="time_in").connect(
        input="time_embed", output="time_vec"
    )
    flux |= mlp_embedder(params.hidden_size, name="vector_in").connect(
        input=y, output="y_vec"
    )
    flux |= Add().connect(left="time_vec", right="y_vec", output="vec")

    if params.guidance_embed:
        guidance = IOKey("guidance", shape=[1])
        flux |= timestep_embedding(dim=256).connect(
            input=guidance, output="guidance_embed"
        )
        flux |= mlp_embedder(params.hidden_size, name="guidance_in").connect(
            input="guidance_embed", output="guidance_vec"
        )
        flux |= Add().connect(left="vec", right="guidance_vec", output="guided_vec")

    flux |= Linear(params.hidden_size, name="img_in").connect(
        input=img, output="img_vec"
    )
    flux |= Linear(params.hidden_size, name="txt_in").connect(
        input=txt, output="txt_vec"
    )
    flux |= Concat(axis=1).connect(input=[txt_ids, img_ids], output="ids")

    flux |= embed_nd(params.theta, params.axes_dim).connect(input="ids", output="pe")

    img_name = "img_vec"
    txt_name = "txt_vec"
    for i in range(params.depth):
        flux |= double_stream_block(
            params.hidden_size,
            params.num_heads,
            params.mlp_ratio,
            params.qkv_bias,
            name=f"double_blocks_{i}",
        ).connect(
            img=img_name,
            txt=txt_name,
            pe="pe",
            vec="vec" if not params.guidance_embed else "guided_vec",
            img_out=f"img{i}",
            txt_out=f"txt{i}",
        )
        img_name = f"img{i}"
        txt_name = f"txt{i}"

    flux |= Concat(axis=1).connect(
        input=[getattr(flux, txt_name), getattr(flux, img_name)], output="img_concat"
    )

    img_name = "img_concat"
    for i in range(params.depth_single_blocks):
        flux |= single_stream_block(
            hidden_size=params.hidden_size,
            num_heads=params.num_heads,
            mlp_ratio=params.mlp_ratio,
            name=f"single_blocks_{i}",
        ).connect(
            input=img_name,
            pe="pe",
            vec="vec" if not params.guidance_embed else "guided_vec",
            output=f"img_single_{i}",
        )
        img_name = f"img_single_{i}"

    img = getattr(flux, img_name)
    # TODO: [:, txt.shape[1] :, ...]
    img = img[:, 512:, ...]  # type: ignore

    flux |= last_layer(
        params.hidden_size, 1, params.in_channels, name="final_layer"
    ).connect(input=img, vec="vec", output=IOKey("output"))
    return flux
