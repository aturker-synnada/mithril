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

import numpy as np
import torch
import gc
import os

import mithril as ml
from examples.flux.auto_encoder import (
    attn_block,
    decoder,
    downsample,
    encoder,
    resnet_block,
    upsample,
)
from examples.flux.layers import apply_rope as apply_rope_mithril
from examples.flux.layers import attention as attention_mithril
from examples.flux.layers import double_stream_block as double_stream_block_mithril
from examples.flux.layers import embed_nd as embed_nd_mithril
from examples.flux.layers import last_layer as last_layer_mithril
from examples.flux.layers import mlp_embedder as mlp_embedder_mithril
from examples.flux.layers import modulation as modulation_mithril
from examples.flux.layers import qk_norm as qk_norm_mithril
from examples.flux.layers import rms_norm as rms_norm_mithril
from examples.flux.layers import rope as rope_mithril
from examples.flux.layers import single_stream_block as single_stream_block_mithril
from examples.flux.layers import timestep_embedding as timestep_embedding_mithril
from examples.flux.model import flux as flux_mithril
from examples.flux.model import FluxParams
from examples.flux.original_impl import (
    AttnBlock,
    Decoder,
    DoubleStreamBlock,
    Downsample,
    EmbedND,
    Encoder,
    Flux,
    LastLayer,
    MLPEmbedder,
    Modulation,
    QKNorm,
    ResnetBlock,
    RMSNorm,
    SingleStreamBlock,
    Upsample,
    apply_rope,
    attention,
    rope,
    timestep_embedding,
)

default_backends = [
    ml.TorchBackend(),
    ml.JaxBackend(),
    # ml.NumpyBackend(),
    ml.MlxBackend(),
]


def load_weights(
    ml_params: dict,
    torch_model: torch.nn.Module,
):
    torch_state_dict = torch_model.state_dict()

    for torch_key in torch_state_dict:
        ml_key = torch_key.replace(".", "_").lower()
        if ml_key not in ml_params:
            continue
        parameter = ml_params[ml_key]

        if torch_state_dict[torch_key].shape != parameter.shape:
            parameter = torch_state_dict[torch_key].clone().reshape(parameter.shape)
        else:
            parameter = torch_state_dict[torch_key].clone().reshape(parameter.shape)
        ml_params[ml_key] = parameter



def test_resnet_block():
    m_model = resnet_block(64, 64)
    o_model = ResnetBlock(64, 64)

    torch_inp = torch.randn(8, 64, 32, 48)

    pm = ml.compile(
        m_model,
        backend=ml.TorchBackend(),
        shapes={"input": [8, 64, 32, 48]},
        data_keys={"input"},
        use_short_namings=False,
    )
    params = pm.randomize_params()
    load_weights(params, o_model)

    expected_result = o_model(torch_inp)

    for backend in default_backends:
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [8, 64, 32, 48]},
            data_keys={"input"},
            use_short_namings=False,
        )

        inp = backend.array(torch_inp.numpy())
        params = {key: backend.array(np.array(value)) for key, value in params.items()}

        res = pm(params, {"input": inp})["output"]

        np.testing.assert_allclose(res, expected_result.detach(), 1e-5, 1e-5)


def test_attn_block():
    m_model = attn_block(512)
    o_model = AttnBlock(512)
    m_model.set_shapes({"input": [8, 512, 32, 32]})

    torch_inp = torch.randn(8, 512, 32, 32)

    pm = ml.compile(
        m_model,
        backend=ml.TorchBackend(),
        shapes={"input": [8, 512, 32, 32]},
        data_keys={"input"},
        use_short_namings=False,
    )
    params = pm.randomize_params()
    load_weights(params, o_model)

    expected_result = o_model(torch_inp).detach()

    for backend in default_backends:
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [8, 512, 32, 32]},
            data_keys={"input"},
            use_short_namings=False,
        )

        inp = backend.array(torch_inp.numpy())
        params = {key: backend.array(np.array(value)) for key, value in params.items()}

        res = pm(params, {"input": inp})["output"]

        np.testing.assert_allclose(res, expected_result, 1e-4, 1e-4)


def test_downsample():
    m_model = downsample(64)
    o_model = Downsample(64)

    torch_inp = torch.randn(8, 64, 32, 48)

    pm = ml.compile(
        m_model,
        backend=ml.TorchBackend(),
        shapes={"input": [8, 64, 32, 48]},
        data_keys={"input"},
        use_short_namings=False,
    )
    params = pm.randomize_params()
    load_weights(params, o_model)

    expected_result = np.array(o_model(torch_inp).detach())

    for backend in default_backends:
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [8, 64, 32, 48]},
            data_keys={"input"},
            use_short_namings=False,
        )

        inp = backend.array(torch_inp.numpy())
        params = {key: backend.array(np.array(value)) for key, value in params.items()}

        res = pm(params, {"input": inp})["output"]

        np.testing.assert_allclose(res, expected_result, 1e-5, 1e-5)


def test_upsample():
    m_model = upsample(64)
    o_model = Upsample(64)

    torch_inp = torch.randn(8, 64, 32, 48)
    #m_model.set_shapes({"input": [8, 64, 32, 48]})

    pm = ml.compile(
        m_model,
        backend=ml.TorchBackend(),
        shapes={"input": [8, 64, 32, 48]},
        data_keys={"input"},
        use_short_namings=False,
        jit=False,
    )
    params = pm.randomize_params()
    load_weights(params, o_model)

    expected_result = np.array(o_model(torch_inp).detach())

    for backend in default_backends:
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [8, 64, 32, 48]},
            data_keys={"input"},
            use_short_namings=False,
            jit=False
        )

        inp = backend.array(torch_inp.numpy())
        params = {key: backend.array(np.array(value)) for key, value in params.items()}

        res = pm(params, {"input": inp})["output"]

        np.testing.assert_allclose(res, expected_result, 1e-5, 1e-5)


def test_encoder():
    backends = [
        ml.JaxBackend(),
        ml.TorchBackend(),
        ml.MlxBackend(),
    ]

    params = {
        "resolution": 256,
        "in_channels": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "z_channels": 16,
    }
    m_model = encoder(**params)
    o_model = Encoder(**params)
    # TODO: Bug in set_shapes, if I provide shape in compile it cannot resolve all shapes
    m_model.set_shapes({"input": [8, 3, 256, 256]})

    torch_inp = torch.randn(8, 3, 256, 256)

    pm = ml.compile(
        m_model,
        backend=ml.TorchBackend(),
        shapes={"input": [8, 3, 256, 256]},
        data_keys={"input"},
        use_short_namings=False,
        jit=False,
    )

    params = pm.randomize_params()
    load_weights(params, o_model)

    expected_result = np.array(o_model(torch_inp).detach())

    for backend in backends:
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [8, 3, 256, 256]},
            data_keys={"input"},
            use_short_namings=False,
            jit=False,
        )
        inp = backend.array(torch_inp.numpy())
        params = {key: backend.array(np.array(value)) for key, value in params.items()}

        res = pm(params, {"input": inp})["output"]

        np.testing.assert_allclose(res, expected_result, 5e-5, 5e-5)



def test_decoder():
    backends = [
        # ml.JaxBackend(),
        ml.TorchBackend(),
        # ml.NumpyBackend(),
        # ml.MlxBackend(),
    ]

    params = {
        "ch": 128,
        "out_ch": 3,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "in_channels": 3,
        "resolution": 256,
        "z_channels": 16,
    }
    m_model = decoder(**params)
    o_model = Decoder(**params)

    # Bug in prune!
    # m_model.set_shapes({"input": [8, 16, 32, 32]})

    torch_inp = torch.randn(8, 16, 32, 32)
    # TODO: summary model names is not understandable

    pm = ml.compile(
        m_model,
        backend=ml.TorchBackend(),
        shapes={"input": [8, 16, 32, 32]},
        data_keys={"input"},
        jit=False,
        file_path="ekmek.py",
        use_short_namings=False

    )

    params = pm.randomize_params()
    load_weights(params, o_model)

    expected_result = np.array(o_model(torch_inp).detach())

    for backend in backends:
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [8, 16, 32, 32]},
            data_keys={"input"},
            jit=False,
            use_short_namings=False
        )

        inp = backend.array(torch_inp.numpy())
        params = {key: backend.array(np.array(value)) for key, value in params.items()}

        res = pm(params, {"input": inp})["output"]

        np.testing.assert_allclose(res, expected_result, 5e-5, 5e-5)


def test_apply_rope():
    backends = [
        ml.JaxBackend(),
        ml.TorchBackend(),
        ml.MlxBackend(),
    ]
    B, L, H, D = 1, 24, 4336, 128
    q_ref = torch.randn(B, L, H, D)
    k_ref = torch.randn(B, L, H, D)
    pe_ref = torch.randn(1, 1, H, D // 2, 2, 2)
    expected_res = apply_rope(q_ref, k_ref, pe_ref)

    for backend in backends:
        pm = ml.compile(
            apply_rope_mithril(),
            backend=backend,
            shapes={
                "xq": [B, L, H, D],
                "xk": [B, L, H, D],
                "freqs_cis": [1, 1, H, D // 2, 2, 2],
            },
            data_keys={"xq", "xk", "freqs_cis"},
        )

        q = backend.array(q_ref.numpy())
        k = backend.array(k_ref.numpy())
        pe = backend.array(pe_ref.numpy())

        res = pm({}, {"xq": q, "xk": k, "freqs_cis": pe})

        np.testing.assert_allclose(res["xq_out"], expected_res[0], 1e-6, 1e-6)
        np.testing.assert_allclose(res["xk_out"], expected_res[1], 1e-6, 1e-6)


def test_time_embeddings():
    backends = [
        ml.MlxBackend(),
        ml.JaxBackend(),
        ml.TorchBackend(),
    ]

    input_ref = torch.ones([1])

    expected_res = timestep_embedding(input_ref, 256)

    for backend in backends:
        pm = ml.compile(
            timestep_embedding_mithril(256),
            backend=backend,
            shapes={"input": [1]},
            data_keys={"input"},
        )

        input = backend.array(input_ref.numpy())

        res = pm({}, {"input": input})

        np.testing.assert_allclose(res["output"], expected_res, 1e-4, 1e-4)


def test_attention():
    backends = [
        ml.MlxBackend(),
        ml.TorchBackend(),
        ml.JaxBackend(),
    ]
    B, L, H, D = 1, 24, 4336, 128
    q_ref = torch.randn(B, L, H, D)
    k_ref = torch.randn(B, L, H, D)
    v_ref = torch.randn(B, L, H, D)
    pe_ref = torch.randn(1, 1, H, D // 2, 2, 2)
    expected_res = attention(q_ref, k_ref, v_ref, pe_ref)

    for backend in backends:
        pm = ml.compile(
            attention_mithril(),
            backend=backend,
            shapes={
                "q": [B, L, H, D],
                "k": [B, L, H, D],
                "v": [B, L, H, D],
                "pe": [1, 1, H, D // 2, 2, 2],
            },
            data_keys={"q", "k", "v", "pe"},
        )

        q = backend.array(q_ref.numpy())
        k = backend.array(k_ref.numpy())
        v = backend.array(v_ref.numpy())
        pe = backend.array(pe_ref.numpy())

        res = pm({}, {"q": q, "k": k, "v": v, "pe": pe})

        np.testing.assert_allclose(res["output"], expected_res, 1e-5, 1e-5)


def test_mlp_embbeder():
    m_model = mlp_embedder_mithril(64)
    o_model = MLPEmbedder(48, 64)

    torch_inp = torch.randn(8, 64, 32, 48)

    pm = ml.compile(
        m_model,
        backend=ml.TorchBackend(),
        shapes={"input": [8, 64, 32, 48]},
        data_keys={"input"},
        use_short_namings=False,
    )
    params = pm.randomize_params()
    load_weights(params, o_model)

    expected_result = np.array(o_model(torch_inp).detach())

    for backend in default_backends:
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [8, 64, 32, 48]},
            data_keys={"input"},
            use_short_namings=False,
        )

        inp = backend.array(torch_inp.numpy())
        params = {key: backend.array(np.array(value)) for key, value in params.items()}

        res = pm(params, {"input": inp})["output"]

        np.testing.assert_allclose(res, expected_result, 1e-5, 1e-5)


def test_rms_norm():
    backends = [
        ml.JaxBackend(),
        ml.TorchBackend(),
        ml.MlxBackend(),
    ]
    B, L, H, D = 1, 24, 4080, 128

    input_ref = torch.randn(B, L, H, D)
    rms_norm = RMSNorm(dim=128)
    expected_res = rms_norm(input_ref)

    for backend in backends:
        pm = ml.compile(
            rms_norm_mithril(dim=128),
            backend=backend,
            shapes={"input": [B, L, H, D], "scale": [128]},
            data_keys={"input", "scale"},
        )

        input = backend.array(input_ref.numpy())
        scale = backend.ones(128)

        res = pm({}, {"input": input, "scale": scale})

        np.testing.assert_allclose(
            res["output"], expected_res.detach().cpu(), 1e-6, 1e-6
        )


def test_qk_norm():
    backends = [
        ml.JaxBackend(),
        ml.TorchBackend(),
        ml.MlxBackend(),
    ]
    B, L, H, D = 1, 24, 4080, 128
    q_ref = torch.randn(B, L, H, D)
    k_ref = torch.randn(B, L, H, D)
    v_ref = torch.randn(B, L, H, D)
    qk_norm = QKNorm(dim=128)
    q_out_ref, k_out_ref = qk_norm(q_ref, k_ref, v_ref)

    for backend in backends:
        pm = ml.compile(
            qk_norm_mithril(128),
            backend=backend,
            shapes={"q_in": [B, L, H, D], "k_in": [B, L, H, D]},
            data_keys={"q_in", "k_in"},
        )

        q_in = backend.array(q_ref.numpy())
        k_in = backend.array(k_ref.numpy())
        scale = backend.ones(128)

        res = pm({"scale_0": scale, "scale_1": scale}, {"q_in": q_in, "k_in": k_in})

        np.testing.assert_allclose(res["q_out"], q_out_ref.detach().cpu(), 1e-6, 1e-6)
        np.testing.assert_allclose(res["k_out"], k_out_ref.detach().cpu(), 1e-6, 1e-6)


def test_modulation():
    backends = [
        ml.JaxBackend(),
        ml.TorchBackend(),
        ml.MlxBackend(),
    ]
    H, W = 1, 3072
    input_ref = torch.randn(H, W)

    m_model = modulation_mithril(3072, False)
    o_model = Modulation(dim=3072, double=False)

    pm = ml.compile(
        m_model,
        backend=ml.TorchBackend(),
        shapes={"input": [H, W]},
        data_keys={"input"},
        use_short_namings=False,
    )

    params = pm.randomize_params()
    load_weights(params, o_model)

    out = o_model(input_ref)

    for backend in backends:
        pm = ml.compile(
            modulation_mithril(3072, False),
            backend=backend,
            shapes={"input": [H, W]},
            data_keys={"input"},
            use_short_namings=False,
        )

        input = backend.array(input_ref.numpy())
        params = {key: backend.array(np.array(value)) for key, value in params.items()}

        res = pm(params, {"input": input})

        np.testing.assert_allclose(
            res["mod_1"][0], out[0].shift.detach().cpu(), 1e-5, 1e-5
        )
        np.testing.assert_allclose(
            res["mod_1"][1], out[0].scale.detach().cpu(), 1e-5, 1e-5
        )
        np.testing.assert_allclose(
            res["mod_1"][2], out[0].gate.detach().cpu(), 1e-5, 1e-5
        )


def test_double_stream_block():
    backends = [
        ml.JaxBackend(),
        ml.TorchBackend(),
        ml.MlxBackend(),
    ]

    hidden_size = 3072
    num_heads = 24
    mlp_ratip = 4.0

    img_ref = torch.randn(1, 4080, 3072)
    txt_ref = torch.randn(1, 256, 3072)
    vec_ref = torch.randn(1, 3072)
    pe_ref = torch.randn(1, 1, 4336, 64, 2, 2)

    m_model = double_stream_block_mithril(hidden_size, num_heads, mlp_ratip)
    o_model = DoubleStreamBlock(hidden_size, num_heads, mlp_ratip)

    pm = ml.compile(
        m_model,
        backend=ml.TorchBackend(precision=32),
        shapes={
            "img": [1, 4080, 3072],
            "txt": [1, 256, 3072],
            "vec": [1, 3072],
            "pe": [1, 1, 4336, 64, 2, 2],
        },
        data_keys={"img", "txt", "vec", "pe"},
        use_short_namings=False,
    )

    params = pm.randomize_params()
    load_weights(params, o_model)

    out = o_model(img_ref, txt_ref, vec_ref, pe_ref)

    for backend in backends:
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={
                "img": [1, 4080, 3072],
                "txt": [1, 256, 3072],
                "vec": [1, 3072],
                "pe": [1, 1, 4336, 64, 2, 2],
            },
            data_keys={"img", "txt", "vec", "pe"},
            use_short_namings=False,
        )

        img = backend.array(img_ref.numpy())
        txt = backend.array(txt_ref.numpy())
        vec = backend.array(vec_ref.numpy())
        pe = backend.array(pe_ref.numpy())

        params = {key: backend.array(np.array(value)) for key, value in params.items()}

        res = pm(params, {"img": img, "txt": txt, "vec": vec, "pe": pe})

        np.testing.assert_allclose(res["img_out"], out[0].detach().cpu(), 1e-5, 1e-5)
        np.testing.assert_allclose(res["txt_out"], out[1].detach().cpu(), 1e-5, 1e-5)


# single stream:
# x-> (1, 4336, 3072), vec->(1, 3072) , pe->(1, 1, 4336,64, 2, 2)


def test_single_stream_block():
    backends = [
        ml.JaxBackend(),
        ml.TorchBackend(),
        ml.MlxBackend(),
    ]

    hidden_size = 3072
    num_heads = 24
    mlp_ratip = 4.0

    input_ref = torch.randn(1, 4336, 3072)
    vec_ref = torch.randn(1, 3072)
    pe_ref = torch.randn(1, 1, 4336, 64, 2, 2)

    m_model = single_stream_block_mithril(hidden_size, num_heads, mlp_ratip)
    o_model = SingleStreamBlock(hidden_size, num_heads, mlp_ratip)

    pm = ml.compile(
        m_model,
        backend=ml.TorchBackend(),
        shapes={
            "input": [1, 4336, 3072],
            "vec": [1, 3072],
            "pe": [1, 1, 4336, 64, 2, 2],
        },
        data_keys={"input", "vec", "pe"},
        use_short_namings=False,
    )

    params = pm.randomize_params()
    load_weights(params, o_model)

    out = o_model(input_ref, vec_ref, pe_ref)

    for backend in backends:
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={
                "input": [1, 4336, 3072],
                "vec": [1, 3072],
                "pe": [1, 1, 4336, 64, 2, 2],
            },
            data_keys={"input", "vec", "pe"},
            use_short_namings=False,

        )

        input = backend.array(input_ref.numpy())
        vec = backend.array(vec_ref.numpy())
        pe = backend.array(pe_ref.numpy())

        params = {key: backend.array(np.array(value)) for key, value in params.items()}

        res = pm(params, {"input": input, "vec": vec, "pe": pe})

        np.testing.assert_allclose(res["output"], out.detach().cpu(), 1e-5, 1e-5)


def test_last_layer():
    backends = [
        ml.JaxBackend(),
        ml.TorchBackend(),
        ml.MlxBackend(),
    ]

    hidden_size = 256
    out_channels = 3
    patch_size = 2

    input_ref = torch.randn(1, 512, 256)
    vec_ref = torch.randn(1, 256)

    m_model = last_layer_mithril(hidden_size, patch_size, out_channels)
    o_model = LastLayer(hidden_size, patch_size, out_channels)

    pm = ml.compile(
        m_model,
        backend=ml.TorchBackend(),
        shapes={"input": [1, 512, 256], "vec": [1, 256]},
        data_keys={"input", "vec"},
        use_short_namings=False,
    )

    params = pm.randomize_params()
    load_weights(params, o_model)

    out = o_model(input_ref, vec_ref)

    for backend in backends:
        pm = ml.compile(
            m_model,
            backend=backend,
            shapes={"input": [1, 512, 256], "vec": [1, 256]},
            data_keys={"input", "vec"},
            use_short_namings=False,
        )

        input = backend.array(input_ref.numpy())
        vec = backend.array(vec_ref.numpy())

        params = {key: backend.array(np.array(value)) for key, value in params.items()}

        res = pm(params, {"input": input, "vec": vec})

        np.testing.assert_allclose(res["output"], out.detach().cpu(), 1e-5, 1e-5)


def test_rope():
    dim = 256
    theta = 4
    pos_ref = torch.rand(dim, 2)
    torch_out = rope(pos_ref, dim, theta)

    for backend in default_backends:
        mithril_rope = rope_mithril(dim, theta)
        pm = ml.compile(mithril_rope, backend=backend, shapes={"input": [dim, 2]})

        pos = backend.array(pos_ref.numpy())
        mithril_out = pm.evaluate({"input": pos})

        np.testing.assert_allclose(
            mithril_out["output"], torch_out, rtol=1e-5, atol=1e-5
        )


def test_embednd():
    dim = 128
    theta = 10_000
    axes_dim = [16, 56, 56]
    input_ref = torch.rand(1, 4336, 3)
    torch_out = EmbedND(dim=dim, theta=theta, axes_dim=axes_dim)(input_ref)

    for backend in default_backends:
        mithril_rope = embed_nd_mithril(theta=theta, axes_dim=axes_dim)
        pm = ml.compile(mithril_rope, backend=backend, shapes={"input": [1, 4336, 3]})

        input = backend.array(input_ref.numpy())
        mithril_out = pm.evaluate({"input": input})

        np.testing.assert_allclose(
            mithril_out["output"], torch_out, rtol=1e-5, atol=1e-5
        )


def test_flux():
    flux_params = FluxParams(
        in_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
    )

    use_torch = False
    if not use_torch:
        flux_m = flux_mithril(flux_params)
        backend = ml.TorchBackend(precision=16)
        pm = ml.compile(flux_m, backend=backend, data_keys={"img","txt", "img_ids", "txt_ids", "timesteps", "y"}, use_short_namings=False, jit=False, file_path="flux.py")
        params = {}
        
        for key in os.listdir("flux_weights"):
            weight = np.load(f"flux_weights/{key}")["arr_0"]
            params[key.replace(".npz","")] = backend.array(weight)
            del weight

        img = params.pop("img")
        txt = params.pop("txt")
        img_ids = params.pop("img_ids")
        txt_ids = params.pop("txt_ids")
        timesteps = params.pop("timesteps")
        y = params.pop("y")
        img_out = params.pop("imgout")
        txt_out = params.pop("txtout")
        #txt_out = params.pop("txtout")
        data = {"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids, "timesteps": timesteps, "y": y}
        res = pm.evaluate(params, data)
    else:
        flux_torch = Flux(flux_params).half().eval()

        img_shape = [1, 1024, 64]
        txt_shape = [1, 256, 4096]
        img_ids_shape = [1, 1024, 3]
        txt_ids_shape = [1, 256, 3]
        timesteps_shape = [1]
        y_shape = [1, 768]

        img = torch.rand(img_shape, dtype=torch.float16)
        txt = torch.rand(txt_shape, dtype=torch.float16)
        img_ids = torch.rand(img_ids_shape, dtype=torch.float16)
        txt_ids = torch.rand(txt_ids_shape, dtype=torch.float16)
        timesteps = torch.rand(timesteps_shape, dtype=torch.float16)
        y = torch.rand(y_shape, dtype=torch.float16)

        np.savez("flux_weights/img", img)
        np.savez("flux_weights/txt", txt)
        np.savez("flux_weights/img_ids", img_ids)
        np.savez("flux_weights/txt_ids", txt_ids)
        np.savez("flux_weights/timesteps", timesteps)
        np.savez("flux_weights/y", y)

        img_out, txt_out = flux_torch(img, img_ids, txt, txt_ids, timesteps, y)
        np.savez("flux_weights/imgout", img_out.detach().numpy())
        np.savez("flux_weights/txtout", txt_out.detach().numpy())
        #np.savez("flux_weights/txtout", txt_out.detach().numpy())
        for key, value in flux_torch.state_dict().items():
            np.savez(f"flux_weights/{key.replace(".","_").lower()}.npz", value.numpy())

    ...
