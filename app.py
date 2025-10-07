import gradio as gr
from gtts import gTTS
from PIL import Image
import io
import requests
import os
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, TextClip, CompositeVideoClip

# Pull API key from environment (set in HF Space secrets)
GROK_API_KEY = os.getenv("GROK_API_KEY")
if not GROK_API_KEY:
    raise ValueError("GROK_API_KEY missing. Add it to HF Space secrets, you absolute legend.")

XAI_API_URL = "https://api.x.ai/v1/grok"  # Grok text generation endpoint

# Absurdist comedian voice system prompt
SYSTEM_PROMPT = """
You are a comedian blending Bill Hicks' cynical truth-telling, George Carlin's language deconstruction, Bo Burnham's meta-humor, Mitch Hedberg's deadpan non-sequiturs, and Kids in the Hall's surreal darkness. Your narration must:
1. Take the theme and spiral it into absurd, dark, paranoid conclusions with airtight but deranged logic.
2. Use literal interpretations of phrases (e.g., "birthday party" becomes surviving another year without dying).
3. Ramble stream-of-consciousness, doubling back with "I mean" and "you know."
4. Drop casual profanity naturally, not for shock.
5. Make authority figures (parents, clowns, etc.) complicit in the absurdity.
6. End with a delayed-realization punchline where the narrator is the butt of the joke.
7. Keep it 150-200 words for 3-4 scenes.
Deliver it like you're three whiskeys deep at a dingy bar at 2 AM.
"""

def generate_narration(theme):
    """Generate narration using Grok API."""
    try:
        headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "grok-3",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Write a vivid, emotional narration about: {theme}"}
            ],
            "max_tokens": 512,
            "temperature": 0.7
        }
        response = requests.post(XAI_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"API error: {response.status_code}")
            return "API’s drunk again. Imagine a clown ranting about cake, okay?"
    except Exception as e:
        print(f"Narration failed: {e}")
        return "Tech gods are pissed. Picture a piñata full of regret."

def extract_visual_prompts(narration):
    """Extract 3 visual prompts from narration."""
    try:
        headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "grok-3",
            "messages": [
                {"role": "system", "content": "Extract 3 vivid, photorealistic visual prompts from the narration. Respond with a Python list of strings."},
                {"role": "user", "content": f"Narration: {narration}"}
            ],
            "max_tokens": 256,
            "temperature": 0.5
        }
        response = requests.post(XAI_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            prompts = eval(response.json()["choices"][0]["message"]["content"])
            return prompts if isinstance(prompts, list) else ["creepy clown at dusk", "overdecorated birthday cake", "abandoned party room"]
        else:
            return ["creepy clown at dusk", "overdecorated birthday cake", "abandoned party room"]
    except:
        return ["creepy clown at dusk", "overdecorated birthday cake", "abandoned party room"]

def generate_images_grok(prompts):
    """Placeholder for images (Pexels fallback until Aurora API is public)."""
    images = []
    for prompt in prompts:
        try:
            response = requests.get(f"https://api.pexels.com/v1/search?query={prompt}&per_page=1", 
                                   headers={"Authorization": "563492ad6f91700001000001your-pexels-key"})
            if response.status_code == 200:
                img_url = response.json()["photos"][0]["src"]["medium"]
                img_data = requests.get(img_url).content
                images.append(img_data)
            else:
                images.append(None)
        except:
            images.append(None)
    return images

def narration_to_audio(narration, filename="narration.mp3"):
    """Convert narration to audio."""
    try:
        tts = gTTS(text=narration, lang='en')
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"Voiceover failed: {e}")
        return None

def generate_script(narration, prompts):
    """Generate scene-by-scene script."""
    narration_lines = narration.split(". ")
    script = "# VIDEO SCRIPT\n\n**Full Narration:**\n" + narration + "\n\n---\n\n## Scene Breakdown:\n"
    for i, (line, prompt) in enumerate(zip(narration_lines[:3], prompts)):
        script += f"\n**Scene {i+1}:**\n- Visual: {prompt}\n- Narration: {line}\n"
    return script

def render_slideshow(images, audio_path, narration, duration_per_slide=3):
    """Render slideshow with images, audio, and subtitles."""
    narration_lines = narration.split(". ")
    clips = []
    for i, img_bytes in enumerate(images):
        if img_bytes:
            img = Image.open(io.BytesIO(img_bytes))
            clip = ImageClip(np.array(img)).set_duration(duration_per_slide).resize(width=1280)
            clip = clip.fadein(0.5).fadeout(0.5)
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
    """Main pipeline: narration, images, video, script."""
    narration = generate_narration(theme)
    prompts = extract_visual_prompts(narration)
    images = generate_images_grok(prompts)
    audio_path = narration_to_audio(narration)
    video_path = render_slideshow(images, audio_path, narration)
    script = generate_script(narration, prompts)
    return narration, script, video_path, audio_path

# Gradio interface
iface = gr.Interface(
    fn=pipeline,
    inputs=gr.Textbox(label="Theme", placeholder="e.g., childhood birthday parties"),
    outputs=[
        gr.Textbox(label="Narration"),
        gr.Textbox(label="Script"),
        gr.Video(label="Slideshow Preview"),
        gr.File(label="Download Narration Audio")
    ],
    title="Concockt: Narration to Animation",
    description="Enter a theme for absurdist, dark-comedy narration and video."
)

if __name__ == "__main__":
    iface.launch()
