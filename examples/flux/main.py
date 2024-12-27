import os
import mithril as ml
from util import configs, load_decoder, load_clip, load_t5, load_flow_model
from sampling import get_noise, prepare, denoise, get_schedule
from dataclasses import dataclass
import gc



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
    prompt: str = "A photo of a cat",
    device: str = "cpu",
    output_dir: str = "temp",
    num_steps: int | None = None,
    guidance: float = 3.5,
    seed: int = 42,

):
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {model_name}, chose from {available}")
    

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
    
    t5 = load_t5(device=device, max_length=256 if model_name == "flux-schnell" else 512)
    clip = load_clip(device=device)
    flow_model, flow_params = load_flow_model(model_name, backend=backend)
    #decoder_model, decoder_params = load_decoder(model_name, backend=backend)


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

    timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell"))
    del t5
    del clip
    denoise(flow_model, flow_params, **inp, timesteps=timesteps)









if __name__ == '__main__':
    run()