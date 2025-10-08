import gradio as gr
from gtts import gTTS
from PIL import Image
import io
import requests
import os
import time
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip, TextClip
import numpy as np
import spaces
from diffusers import StableDiffusionXLPipeline
import torch

# API Keys from environment
XAI_API_KEY = os.getenv("XAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY not found! Add it to environment/secrets.")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found! Add it to environment/secrets.")

# Load Stable Diffusion XL model
print("Loading Stable Diffusion XL model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    token=HF_TOKEN
)

def grok_text(prompt, max_tokens=400):
    """Call Grok for text generation"""
    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "messages": [{"role": "user", "content": prompt}],
                "model": "grok-beta",
                "max_tokens": max_tokens,
                "temperature": 0.8
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Grok error: {e}")
        return None

def generate_narration(theme):
    """Generate absurdist comedian narration"""
    system_context = """You're a dark absurdist comedian (Bill Hicks + George Carlin + Bo Burnham style).

Write 4-5 sentences that:
- Start mundane, spiral into dark absurdity through paranoid logic
- Use "I mean" and "you know" naturally
- Include casual profanity (naturally, not forced)
- Follow airtight but deranged internal logic
- Deadpan delivery, no apologies
- Make authority figures complicit in the absurdity

You're three whiskeys deep at 2 AM at a dingy bar."""

    prompt = f"""{system_context}

Theme: {theme}

Write the narration:"""
    
    result = grok_text(prompt, max_tokens=350)
    return result if result else f"So, {theme}, right? I mean, nobody asked for this but here we fucking are, and somehow we're all complicit."

def get_visual_prompts(narration, num_scenes=4):
    """Extract visual prompts from narration"""
    prompt = f"""From this narration, extract {num_scenes} visual prompts for dark, cinematic image generation.

Narration: {narration}

Each prompt should be:
- 8-12 words
- Photorealistic, cinematic
- Include lighting/mood (dramatic, moody, dark, atmospheric)
- Match the absurdist/dark tone

Format as numbered list:
1. prompt here
2. prompt here
3. prompt here
4. prompt here"""
    
    result = grok_text(prompt, max_tokens=250)
    
    if result:
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        prompts = []
        for line in lines:
            clean = line.split(".", 1)[-1].strip().strip("[]\"'-")
            if clean and len(clean) > 10:
                prompts.append(clean)
        if len(prompts) >= num_scenes:
            return prompts[:num_scenes]
    
    # Fallback prompts
    words = narration.split()
    return [
        f"dark cinematic scene, {' '.join(words[i:i+4])}, moody lighting, photorealistic"
        for i in range(0, min(len(words), num_scenes*5), max(len(words)//num_scenes, 5))
    ][:num_scenes]

@spaces.GPU
def generate_image_sdxl(prompt):
    """Generate image using Stable Diffusion XL on Hugging Face GPU"""
    try:
        # Generate image
        image = pipe(
            prompt=prompt,
            negative_prompt="ugly, blurry, low quality, distorted, deformed",
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=576  # 16:9 aspect ratio
        ).images[0]
        
        return image
        
    except Exception as e:
        print(f"Image generation error: {e}")
        return create_fallback_image(prompt)

def create_fallback_image(text):
    """Create fallback image if generation fails"""
    img = Image.new('RGB', (1280, 720), color=(20, 20, 30))
    return img

def make_audio(text):
    """Generate voiceover audio"""
    try:
        tts = gTTS(text, lang='en', slow=False)
        filename = f"audio_{int(time.time())}.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"Audio error: {e}")
        return None

def apply_ken_burns(clip, zoom_direction="in"):
    """Apply Ken Burns pan and zoom effect"""
    duration = clip.duration
    
    if zoom_direction == "in":
        # Zoom in while panning
        zoom_effect = lambda t: 1 + 0.3 * (t / duration)
        pan_effect = lambda t: ('center', int(50 * (t / duration)))
    else:
        # Zoom out while panning
        zoom_effect = lambda t: 1.3 - 0.3 * (t / duration)
        pan_effect = lambda t: ('center', int(50 - 50 * (t / duration)))
    
    return clip.resize(zoom_effect).set_position(pan_effect)

def make_video(narration, images, audio_path, duration_per_scene=5):
    """Create cinematic video with Ken Burns effects - supports 16:9 and 9:16"""
    clips_16_9 = []
    clips_9_16 = []
    sentences = [s.strip() + "." for s in narration.split(".") if s.strip()]
    
    for i, img in enumerate(images):
        try:
            # Convert PIL to numpy
            if isinstance(img, Image.Image):
                img_array = np.array(img)
            else:
                img_array = img
            
            # Create 16:9 clip (full landscape)
            clip_landscape = ImageClip(img_array).set_duration(duration_per_scene)
            clip_landscape = apply_ken_burns(clip_landscape, "in" if i % 2 == 0 else "out")
            clip_landscape = clip_landscape.crossfadein(0.8).crossfadeout(0.8)
            
            # Create 9:16 clip (center crop for portrait)
            h, w = img_array.shape[:2]
            crop_width = int(h * 9 / 16)
            x_center = w // 2
            x1 = max(0, x_center - crop_width // 2)
            x2 = min(w, x_center + crop_width // 2)
            
            img_portrait = img_array[:, x1:x2]
            clip_portrait = ImageClip(img_portrait).set_duration(duration_per_scene)
            clip_portrait = apply_ken_burns(clip_portrait, "in" if i % 2 == 0 else "out")
            clip_portrait = clip_portrait.crossfadein(0.8).crossfadeout(0.8)
            
            # Add subtitles to both versions
            if i < len(sentences):
                subtitle_text = sentences[i]
                if len(subtitle_text) > 100:
                    subtitle_text = subtitle_text[:97] + "..."
                
                try:
                    # Landscape subtitle
                    txt_landscape = TextClip(
                        subtitle_text,
                        fontsize=38,
                        color='white',
                        font='Arial-Bold',
                        stroke_color='black',
                        stroke_width=2,
                        method='caption',
                        size=(1100, None),
                        align='center'
                    )
                    txt_landscape = (txt_landscape
                                    .set_duration(duration_per_scene)
                                    .set_position(('center', 480))
                                    .crossfadein(0.5)
                                    .crossfadeout(0.5))
                    
                    clip_landscape = CompositeVideoClip([clip_landscape, txt_landscape])
                    
                    # Portrait subtitle (narrower)
                    txt_portrait = TextClip(
                        subtitle_text,
                        fontsize=32,
                        color='white',
                        font='Arial-Bold',
                        stroke_color='black',
                        stroke_width=2,
                        method='caption',
                        size=(650, None),
                        align='center'
                    )
                    txt_portrait = (txt_portrait
                                   .set_duration(duration_per_scene)
                                   .set_position(('center', 900))
                                   .crossfadein(0.5)
                                   .crossfadeout(0.5))
                    
                    clip_portrait = CompositeVideoClip([clip_portrait, txt_portrait])
                    
                except Exception as e:
                    print(f"Subtitle error: {e}")
            
            clips_16_9.append(clip_landscape)
            clips_9_16.append(clip_portrait)
            
        except Exception as e:
            print(f"Clip error: {e}")
            continue
    
    if not clips_16_9:
        return None, None
    
    # Concatenate both versions
    final_16_9 = concatenate_videoclips(clips_16_9, method="compose")
    final_9_16 = concatenate_videoclips(clips_9_16, method="compose")
    
    # Add audio to both
    if audio_path and os.path.exists(audio_path):
        try:
            audio = AudioFileClip(audio_path)
            final_16_9 = final_16_9.set_audio(audio)
            final_9_16 = final_9_16.set_audio(audio.copy())
        except Exception as e:
            print(f"Audio sync error: {e}")
    
    # Render both videos
    timestamp = int(time.time())
    output_16_9 = f"video_16x9_{timestamp}.mp4"
    output_9_16 = f"video_9x16_{timestamp}.mp4"
    
    final_16_9.write_videofile(
        output_16_9,
        fps=30,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile=f'temp-audio-16-9-{timestamp}.m4a',
        remove_temp=True,
        logger=None
    )
    
    final_9_16.write_videofile(
        output_9_16,
        fps=30,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile=f'temp-audio-9-16-{timestamp}.m4a',
        remove_temp=True,
        logger=None
    )
    
    return output_16_9, output_9_16

def make_script(narration, prompts):
    """Generate formatted script"""
    script = f"""# VIDEO SCRIPT

**FULL NARRATION:**
{narration}

---

## SCENE BREAKDOWN:

"""
    
    sentences = [s.strip() + "." for s in narration.split(".") if s.strip()]
    
    for i, prompt in enumerate(prompts, 1):
        script += f"**Scene {i}:**\n"
        script += f"- Visual: {prompt}\n"
        if i-1 < len(sentences):
            script += f"- Narration: {sentences[i-1]}\n"
        script += f"- Effect: {'Ken Burns zoom in + pan' if i % 2 == 1 else 'Ken Burns zoom out + pan'}\n\n"
    
    return script

def pipeline(theme, num_scenes=4, seconds_per_scene=5, progress=gr.Progress()):
    """Main pipeline with progress tracking - generates BOTH aspect ratios"""
    try:
        # Step 1: Generate narration
        progress(0.1, desc="Generating narration with Grok...")
        narration = generate_narration(theme)
        if not narration:
            return "Narration generation failed", "", None, None, None
        
        # Step 2: Extract visual prompts
        progress(0.2, desc="Extracting visual prompts...")
        prompts = get_visual_prompts(narration, num_scenes)
        
        # Step 3: Generate images with SDXL
        images = []
        for i, prompt in enumerate(prompts):
            progress(0.3 + (i / num_scenes) * 0.4, desc=f"Generating image {i+1}/{num_scenes} with SDXL...")
            img = generate_image_sdxl(prompt)
            images.append(img)
        
        # Step 4: Create script
        progress(0.75, desc="Formatting script...")
        script = make_script(narration, prompts)
        
        # Step 5: Generate audio
        progress(0.8, desc="Creating voiceover...")
        audio_path = make_audio(narration)
        
        # Step 6: Render BOTH videos (16:9 and 9:16)
        progress(0.85, desc="Rendering videos (landscape + portrait)...")
        video_16_9, video_9_16 = make_video(narration, images, audio_path, seconds_per_scene)
        
        progress(1.0, desc="Complete!")
        
        return narration, script, video_16_9, video_9_16, audio_path
        
    except Exception as e:
        return f"Error: {str(e)}", "", None, None, None

# Gradio Interface
with gr.Blocks(theme=gr.themes.Monochrome(), css="""
    .gradio-container {
        font-family: 'Courier New', monospace !important;
    }
    h1 {
        color: #ff4444 !important;
        text-align: center;
        font-weight: bold;
        font-size: 3em !important;
    }
    .generate-btn {
        background: #ff4444 !important;
        font-size: 1.2em !important;
    }
""") as demo:
    
    gr.Markdown("""
    # 🎬 CONCOCKT
    ### Absurdist AI Narration → Cinematic Dark Comedy
    
    *Powered by Grok + Stable Diffusion XL on Hugging Face GPU*
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            theme_input = gr.Textbox(
                label="Theme",
                placeholder="childhood birthday parties, gym memberships, motivational speakers, wedding traditions...",
                lines=2
            )
        
        with gr.Column(scale=1):
            num_scenes = gr.Slider(
                label="Number of Scenes",
                minimum=3,
                maximum=6,
                value=4,
                step=1
            )
            duration = gr.Slider(
                label="Seconds per Scene",
                minimum=3,
                maximum=8,
                value=5,
                step=1
            )
    
    generate_btn = gr.Button("🔥 GENERATE VIDEO", variant="primary", size="lg", elem_classes="generate-btn")
    
    with gr.Row():
        with gr.Column():
            narration_out = gr.Textbox(label="Generated Narration", lines=8)
            script_out = gr.Textbox(label="Scene-by-Scene Script", lines=12)
            audio_out = gr.Audio(label="Voiceover Audio")
        
        with gr.Column():
            gr.Markdown("### 🖥️ Desktop / YouTube (16:9)")
            video_16_9_out = gr.Video(label="Landscape Video")
            
            gr.Markdown("### 📱 TikTok / Reels / Shorts (9:16)")
            video_9_16_out = gr.Video(label="Portrait Video")
    
    gr.Markdown("""
    ---
    ### Features:
    - **Dual aspect ratios**: Get BOTH 16:9 (desktop/YouTube) and 9:16 (TikTok/Reels) from one generation
    - **Absurdist narration** via Grok (dark comedian voice)
    - **Cinematic images** via Stable Diffusion XL (FREE on HF GPU!)
    - **Ken Burns effects** (zoom + pan on each scene)
    - **Crossfade transitions** between scenes
    - **Subtitle overlays** adapted for each format
    - **Professional audio sync** on both versions
    
    ### Setup:
    Add to HF Space Secrets:
    - `XAI_API_KEY` - Your Grok API key
    - `HF_TOKEN` - Your Hugging Face token (for downloading SDXL model)
    
    ### Hardware:
    Make sure your Space is set to **ZeroGPU** in Settings → Hardware
    """)
    
    generate_btn.click(
        fn=pipeline,
        inputs=[theme_input, num_scenes, duration],
        outputs=[narration_out, script_out, video_16_9_out, video_9_16_out, audio_out]
    )

if __name__ == "__main__":
    demo.launch()
