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

import os
from dataclasses import dataclass

from PIL import Image
from sampling import denoise, get_noise, get_schedule, prepare, rearrange, unpack
from util import configs, load_clip, load_decoder, load_flow_model, load_t5

import mithril as ml


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None = None


def run(
    model_name: str = "flux-schnell",
    width: int = 512,
    height: int = 512,
    prompt: str = "A photo q of a cat",
    device: str = "cuda",
    output_dir: str = "temp",
    num_steps: int | None = None,
    guidance: float = 3.5,
    seed: int = 42,
):
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(
            f"Got unknown model name: {model_name}, chose from {available}"
        )

    if num_steps is None:
        num_steps = 4 if model_name == "flux-schnell" else 50

    # allow for packing and conversion to latent space
    height = 16 * (height // 16)
    width = 16 * (width // 16)

    output_name = os.path.join(output_dir, "img.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    backend = ml.TorchBackend(device=device, precision=16)

    t5 = load_t5(
        device=device, max_length=256 if model_name == "flux-schnell" else 512
    ).to("cuda")
    clip = load_clip(device=device).to("cuda")
    flow_model, flow_params = load_flow_model(model_name, backend=backend)
    decoder, decoder_params = load_decoder("flux-schnell", backend=backend)

    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    noise = get_noise(1, opts.height, opts.width, device, seed)
    inp = prepare(t5, clip, noise, prompt=opts.prompt)
    t5 = t5.to("cpu")
    clip = clip.to("cpu")

    timesteps = get_schedule(
        opts.num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell")
    )
    del t5
    del clip

    x = denoise(flow_model, flow_params, **inp, timesteps=timesteps)
    x = unpack(x, opts.height, opts.width)
    x = decoder.evaluate(decoder_params, {"input": x.float()})["output"]
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())


if __name__ == "__main__":
    run()
