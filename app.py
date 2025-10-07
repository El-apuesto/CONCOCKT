import gradio as gr
from gtts import gTTS
from PIL import Image
import io
import requests
import os
import numpy as np
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, TextClip, CompositeVideoClip
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

model = Transformer("models/mistral_nemo", tokenizer="tekken", instruct=True)
HF_TOKEN = os.getenv("HF_TOKEN")
ENDPOINT_URL = "https://api-inference.huggingface.co/models/your-username/your-model"

def generate_narration(theme):
    prompt = f"Write a vivid, emotional narration about: {theme}"
    return generate(model, prompt, max_tokens=512, temperature=0.7)

def extract_visual_prompts(narration):
    prompt = f"""Extract 3 vivid, photorealistic visual prompts from the following narration.
Narration:
{narration}
Respond with a Python list of strings."""
    response = generate(model, prompt, max_tokens=256, temperature=0.5)
    try:
        prompts = eval(response)
        return prompts if isinstance(prompts, list) else []
    except:
        return ["misty jungle ruins", "sunset horizon", "ancient stone statue"]

def generate_images(prompts):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    images = []
    for prompt in prompts:
        try:
            response = requests.post(ENDPOINT_URL, headers=headers, json={"inputs": prompt})
            images.append(response.content if response.status_code == 200 else None)
        except:
            images.append(None)
    return images

def narration_to_audio(narration, filename="narration.mp3"):
    try:
        tts = gTTS(text=narration, lang='en')
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"Voiceover generation failed: {e}")
        return None

def render_slideshow(images, audio_path, narration, duration_per_slide=3):
    narration_lines = narration.split(". ")
    clips = []
    for i, img_bytes in enumerate(images):
        if img_bytes:
            img = Image.open(io.BytesIO(img_bytes))
            clip = ImageClip(img).set_duration(duration_per_slide).resize(width=1280)

            # Pan/Zoom effect
            clip = clip.set_position(lambda t: ('center', 100 + t*20)).resize(lambda t: 1 + 0.02*t)

            # Fade in/out
            clip = clip.fadein(0.5).fadeout(0.5)

            # Subtitle overlay
            if i < len(narration_lines):
                subtitle = TextClip(narration_lines[i], fontsize=40, color='white', bg_color='black', size=(1280, 80))
                subtitle = subtitle.set_duration(duration_per_slide).set_position(('center', 'bottom'))
                clip = CompositeVideoClip([clip, subtitle])

            clips.append(clip)

    final = concatenate_videoclips(clips, method="compose", padding=-1)
    if audio_path:
        final = final.set_audio(AudioFileClip(audio_path))
    final.write_videofile("slideshow.mp4", fps=24)
    return "slideshow.mp4"

def pipeline(theme):
    narration = generate_narration(theme)
    prompts = extract_visual_prompts(narration)
    images = generate_images(prompts)
    audio_path = narration_to_audio(narration)
    video_path = render_slideshow(images, audio_path, narration)
    return narration, video_path, audio_path

gr.Interface(
    fn=pipeline,
    inputs=gr.Textbox(label="Theme"),
    outputs=[
        gr.Textbox(label="Narration"),
        gr.Video(label="Slideshow Preview"),
        gr.File(label="Download Narration Audio")
    ],
    title="Concoct: Narration to Animation",
    description="Enter a theme to generate narration, visuals, and cinematic animation"
).launch()
