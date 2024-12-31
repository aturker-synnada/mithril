from dataclasses import dataclass

from mithril import IOKey
from mithril.models import Add, Concat, Linear, Model, Buffer

from examples.flux.layers import (
    double_stream_block,
    embed_nd,
    last_layer,
    mlp_embedder,
    single_stream_block,
    timestep_embedding,
)


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

    img = IOKey("img", shape=[1, 1024, 64])
    txt = IOKey("txt", shape=[1, 256, 4096])

    img_ids = IOKey("img_ids", shape=[1, 1024, 3])
    txt_ids = IOKey("txt_ids", shape=[1, 256, 3])

    timesteps = IOKey("timesteps", shape=[1])
    y = IOKey("y", shape=[1, 768])

    flux += timestep_embedding(dim=256)(input=timesteps, output="time_embed")
    flux += mlp_embedder(params.hidden_size, name="time_in")(
        input="time_embed", output="time_vec"
    )
    flux += mlp_embedder(params.hidden_size, name="vector_in")(input=y, output="y_vec")
    flux += Add()(left="time_vec", right="y_vec", output="vec")

    flux += Linear(params.hidden_size, name="img_in")(input=img, output="img_vec")
    flux += Linear(params.hidden_size, name="txt_in")(input=txt, output="txt_vec")
    flux += Concat(n=2, axis=1)(input1=txt_ids, input2=img_ids, output="ids")

    flux += embed_nd(params.theta, params.axes_dim)(input="ids", output="pe")

    img_name = "img_vec"
    txt_name = "txt_vec"
    for i in range(params.depth):
        flux += double_stream_block(
            params.hidden_size,
            params.num_heads,
            params.mlp_ratio,
            params.qkv_bias,
            name=f"double_blocks_{i}",
        )(
            img=img_name,
            txt=txt_name,
            pe="pe",
            vec="vec",
            img_out=f"img{i}",
            txt_out=f"txt{i}",
        )
        img_name = f"img{i}"
        txt_name = f"txt{i}"


    flux += Concat(n=2, axis=1)(input1=img_name, input2=txt_name, output="img_concat")
    img_name = "img_concat"
    for i in range(params.depth_single_blocks):
        flux += single_stream_block(
            hidden_size=params.hidden_size,
            num_heads=params.num_heads,
            mlp_ratio=params.mlp_ratio,
            name=f"single_blocks_{i}",
        )(
            input=img_name,
            vec="vec",
            pe="pe",
            output=f"img_single_{i}",
        )
        img_name = f"img_single_{i}"

    flux += Buffer()(input=img_name, output=IOKey("img_buffer"))
    img = getattr(flux, img_name)
    # TODO: [:, txt.shape[1] :, ...]
    img = img[:, 256:, ...]

    flux += last_layer(params.hidden_size, 1, params.in_channels, name="final_layer")(
        input=img, vec="vec", output=IOKey("output")
    )
    return flux