import gradio as gr
from gtts import gTTS
from PIL import Image
import io
import requests
import os
import time
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
import numpy as np

# Get API key from HF Secrets or environment variable
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY not found! Add it to HF Space secrets.")

def grok_text(prompt, max_tokens=400):
    """Call Grok API for text"""
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
    system_context = """You're a dark absurdist comedian (Bill Hicks + George Carlin + Bo Burnham). 
Write 4-5 sentences that:
- Start mundane, spiral into dark absurdity
- Use "I mean" and "you know" naturally  
- Include casual profanity
- Follow paranoid logic to insane conclusions
- Deadpan delivery, no apologies

You're three whiskeys deep at 2 AM telling some SHIT."""

    prompt = f"""{system_context}

Theme: {theme}

Write the narration now:"""
    
    result = grok_text(prompt, max_tokens=300)
    return result if result else f"So, {theme}, right? I mean, nobody asked for this but here we fucking are."

def get_image_prompts(narration):
    """Extract visual prompts"""
    prompt = f"""From this narration, extract 4 SHORT visual prompts (5-7 words each) for image generation.

Narration: {narration}

Format as simple list:
1. [prompt]
2. [prompt]  
3. [prompt]
4. [prompt]"""
    
    result = grok_text(prompt, max_tokens=200)
    
    if result:
        lines = [l.strip() for l in result.split("\n") if l.strip() and not l.strip().startswith("#")]
        prompts = []
        for line in lines:
            # Remove numbering
            clean = line.split(".", 1)[-1].strip().strip("[]\"'")
            if clean:
                prompts.append(clean)
        if len(prompts) >= 4:
            return prompts[:4]
    
    # Fallback
    words = narration.split()[:20]
    return [
        f"dark cinematic {' '.join(words[0:3])}",
        f"dramatic {' '.join(words[5:8])}",
        f"moody {' '.join(words[10:13])}",
        f"atmospheric {' '.join(words[15:18])}"
    ]

def create_placeholder_image(text, index):
    """Create a styled placeholder image"""
    colors = [
        (20, 20, 40),   # dark blue
        (40, 20, 20),   # dark red
        (20, 40, 20),   # dark green
        (40, 30, 20),   # dark brown
    ]
    img = Image.new('RGB', (1280, 720), color=colors[index % 4])
    return np.array(img)

def make_audio(text):
    """Generate TTS audio"""
    try:
        tts = gTTS(text, lang='en', slow=False)
        filename = f"audio_{int(time.time())}.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"Audio error: {e}")
        return None

def make_video(narration, prompts, audio_path):
    """Create video from images and audio"""
    clips = []
    
    # Create images
    for i in range(len(prompts)):
        img = create_placeholder_image(prompts[i], i)
        clip = ImageClip(img).set_duration(4)
        clip = clip.crossfadein(0.5).crossfadeout(0.5)
        clips.append(clip)
    
    final = concatenate_videoclips(clips, method="compose")
    
    # Add audio
    if audio_path and os.path.exists(audio_path):
        try:
            from moviepy.editor import AudioFileClip
            audio = AudioFileClip(audio_path)
            final = final.set_audio(audio)
        except Exception as e:
            print(f"Audio sync error: {e}")
    
    output = f"video_{int(time.time())}.mp4"
    final.write_videofile(output, fps=24, codec='libx264', audio_codec='aac', logger=None)
    return output

def make_script(narration, prompts):
    """Format script breakdown"""
    script = f"""# VIDEO SCRIPT

**FULL NARRATION:**
{narration}

---

## SCENE BREAKDOWN:
"""
    
    sentences = [s.strip() + "." for s in narration.split(".") if s.strip()]
    
    for i, prompt in enumerate(prompts, 1):
        script += f"\n**Scene {i}:**\n"
        script += f"Visual: {prompt}\n"
        if i-1 < len(sentences):
            script += f"Narration: {sentences[i-1]}\n"
    
    return script

def pipeline(theme):
    """Main pipeline"""
    try:
        # Generate narration
        narration = generate_narration(theme)
        if not narration:
            return "Narration failed", "", None, None
        
        # Get visual prompts
        prompts = get_image_prompts(narration)
        
        # Create script
        script = make_script(narration, prompts)
        
        # Generate audio
        audio_path = make_audio(narration)
        
        # Create video
        video_path = make_video(narration, prompts, audio_path)
        
        return narration, script, video_path, audio_path
        
    except Exception as e:
        return f"Error: {str(e)}", "", None, None

# Gradio UI with better styling
with gr.Blocks(theme=gr.themes.Monochrome(), css="""
    .gradio-container {
        font-family: 'Courier New', monospace !important;
    }
    h1 {
        color: #ff4444 !important;
        text-align: center;
        font-weight: bold;
    }
""") as demo:
    
    gr.Markdown("""
    # 🎬 CONCOCKT
    ### Absurdist AI Narration → Dark Comedy Video
    
    *Powered by Grok. Three whiskeys deep.*
    """)
    
    with gr.Row():
        theme_input = gr.Textbox(
            label="Enter Theme",
            placeholder="childhood birthday parties, gym memberships, motivational speakers...",
            lines=2,
            scale=4
        )
    
    generate_btn = gr.Button("🔥 GENERATE", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            narration_out = gr.Textbox(label="Generated Narration", lines=8)
            script_out = gr.Textbox(label="Scene Breakdown", lines=12)
        
        with gr.Column():
            video_out = gr.Video(label="Video Output")
            audio_out = gr.Audio(label="Voiceover Audio")
    
    gr.Markdown("""
    ---
    **NOTE:** Currently using placeholder images (colored frames) until Grok Aurora API is publicly available.
    The narration and audio are fully functional with Grok AI.
    """)
    
    generate_btn.click(
        fn=pipeline,
        inputs=[theme_input],
        outputs=[narration_out, script_out, video_out, audio_out]
    )

if __name__ == "__main__":
    demo.launch()
