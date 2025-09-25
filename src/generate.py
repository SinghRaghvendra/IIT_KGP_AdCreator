import torch
from diffusers import StableDiffusionPipeline
import os

def load_model(model_id="runwayml/stable-diffusion-v1-5"):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

def generate_ad(pipe, prompt, out_dir="outputs", n=1, steps=20):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(n):
        out = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5)
        img = out.images[0]
        path = os.path.join(out_dir, f"ad_{i}.png")
        img.save(path)
        paths.append(path)
    return paths

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    pipe = load_model()
    files = generate_ad(pipe, args.prompt, args.out_dir, n=3)
    print("Generated:", files)
