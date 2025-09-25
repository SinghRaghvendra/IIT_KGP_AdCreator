import gradio as gr
from src.generate import load_model, generate_ad

pipe = load_model()

def generate_interface(prompt, num_images=1, steps=20):
    files = generate_ad(pipe, prompt, out_dir="outputs", n=num_images, steps=steps)
    return files

with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ AI Advertisement Creator")

    prompt = gr.Textbox(label="Enter Ad Prompt", placeholder="e.g. Ad for luxury watch, premium style")
    num_images = gr.Slider(1, 5, value=1, step=1, label="Number of Ads")
    steps = gr.Slider(10, 50, value=20, step=1, label="Inference Steps")

    generate_btn = gr.Button("Generate Ads")
    gallery = gr.Gallery(label="Generated Ads").style(grid=[2], height="auto")

    generate_btn.click(fn=generate_interface, inputs=[prompt, num_images, steps], outputs=[gallery])

if __name__ == "__main__":
    demo.launch()
