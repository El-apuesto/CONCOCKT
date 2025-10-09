import React, { useState, useRef } from 'react';
import { Download, Loader } from 'lucide-react';

export default function Konkokt() {
  const [scriptInput, setScriptInput] = useState('');
  const [aspectRatio, setAspectRatio] = useState('both');
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState('');
  const [scenes, setScenes] = useState([]);
  const [videoBlobs, setVideoBlobs] = useState({ wide: null, tall: null });
  
  const canvasRef = useRef(null);

  const breakIntoScenes = (text) => {
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    const scenes = [];
    let currentScene = '';
    
    sentences.forEach(sentence => {
      currentScene += sentence;
      const wordCount = currentScene.split(' ').length;
      
      if (wordCount >= 15 || sentence === sentences[sentences.length - 1]) {
        scenes.push(currentScene.trim());
        currentScene = '';
      }
    });
    
    return scenes.length > 0 ? scenes : [text];
  };

  const generatePromptFromScript = (sceneText) => {
    const words = sceneText.split(' ').slice(0, 10).join(' ');
    return `cinematic scene: ${words}`;
  };

  const generateImage = async (prompt) => {
    const url = `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}?width=1024&height=1024&nologo=true`;
    
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve({ url, img });
      img.onerror = () => reject(new Error('Image generation failed'));
      img.src = url;
    });
  };

  const speak = (text, voiceType) => {
    return new Promise((resolve) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      const voices = speechSynthesis.getVoices();
      
      if (voiceType === 'female') {
        utterance.voice = voices.find(v => v.name.includes('Female') || v.name.includes('Samantha')) || voices[0];
      } else {
        utterance.voice = voices.find(v => v.name.includes('Male') || v.name.includes('Daniel')) || voices[0];
      }
      
      utterance.onend = resolve;
      speechSynthesis.speak(utterance);
    });
  };

  const drawKenBurns = (canvas, img, progress, aspectRatio = 'square') => {
    const ctx = canvas.getContext('2d');
    
    if (aspectRatio === '16:9') {
      canvas.width = 1920;
      canvas.height = 1080;
    } else if (aspectRatio === '9:16') {
      canvas.width = 1080;
      canvas.height = 1920;
    } else {
      canvas.width = 1024;
      canvas.height = 1024;
    }

    const scale = 1 + (progress * 0.2);
    const offsetX = -progress * 50;
    const offsetY = -progress * 30;

    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.scale(scale, scale);
    
    const imgAspect = img.width / img.height;
    const canvasAspect = canvas.width / canvas.height;
    let drawWidth, drawHeight;
    
    if (imgAspect > canvasAspect) {
      drawHeight = canvas.height;
      drawWidth = drawHeight * imgAspect;
    } else {
      drawWidth = canvas.width;
      drawHeight = drawWidth / imgAspect;
    }
    
    ctx.translate(offsetX, offsetY);
    ctx.drawImage(img, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
    ctx.restore();
  };

  const recordVideo = async (scenes, aspectRatio) => {
    const canvas = canvasRef.current;
    const stream = canvas.captureStream(30);
    const mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' });
    
    const chunks = [];
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    
    return new Promise(async (resolve) => {
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        resolve(blob);
      };

      mediaRecorder.start();

      for (let i = 0; i < scenes.length; i++) {
        const scene = scenes[i];
        const duration = Math.max(4, Math.min(8, scene.script.split(' ').length * 0.3));
        
        const startTime = Date.now();
        const endTime = startTime + (duration * 1000);
        
        const animate = () => {
          const elapsed = (Date.now() - startTime) / 1000;
          const progress = Math.min(elapsed / duration, 1);
          
          drawKenBurns(canvas, scene.img, progress, aspectRatio);
          
          if (Date.now() < endTime) {
            requestAnimationFrame(animate);
          }
        };
        
        animate();
        await speak(scene.script, scene.voice);
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      mediaRecorder.stop();
    });
  };

  const generateStory = async () => {
    if (!scriptInput.trim()) {
      alert('Enter a script or topic');
      return;
    }

    setGenerating(true);
    setProgress('Breaking script into scenes...');
    setVideoBlobs({ wide: null, tall: null });

    try {
      const sceneTexts = breakIntoScenes(scriptInput);
      setProgress(`Generating ${sceneTexts.length} images...`);

      const generatedScenes = [];
      for (let i = 0; i < sceneTexts.length; i++) {
        setProgress(`Generating image ${i + 1}/${sceneTexts.length}...`);
        const prompt = generatePromptFromScript(sceneTexts[i]);
        const { url, img } = await generateImage(prompt);
        
        generatedScenes.push({
          script: sceneTexts[i],
          imageUrl: url,
          img: img,
          voice: i % 2 === 0 ? 'male' : 'female'
        });
      }

      setScenes(generatedScenes);
      setProgress('Creating videos...');

      const videos = {};
      
      if (aspectRatio === '16:9' || aspectRatio === 'both') {
        setProgress('Recording 16:9 video...');
        videos.wide = await recordVideo(generatedScenes, '16:9');
      }
      
      if (aspectRatio === '9:16' || aspectRatio === 'both') {
        setProgress('Recording 9:16 video...');
        videos.tall = await recordVideo(generatedScenes, '9:16');
      }

      setVideoBlobs(videos);
      setProgress('Done! Download your video(s).');
      setGenerating(false);

    } catch (error) {
      alert('Error: ' + error.message);
      setGenerating(false);
      setProgress('');
    }
  };

  const downloadVideo = (blob, format) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `konkokt-${format}.webm`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-8 text-center">KONKOKT</h1>

        <div className="bg-gray-800 p-6 rounded-lg mb-6">
          <label className="block mb-2 font-semibold">Script or Topic:</label>
          <textarea
            value={scriptInput}
            onChange={(e) => setScriptInput(e.target.value)}
            placeholder="Paste your full script here, or just type a topic like 'a dragon's adventure'..."
            className="w-full h-40 p-4 bg-gray-700 rounded text-white resize-none"
            disabled={generating}
          />
        </div>

        <div className="bg-gray-800 p-6 rounded-lg mb-6">
          <label className="block mb-3 font-semibold">Output Format:</label>
          <div className="flex gap-4">
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                value="16:9"
                checked={aspectRatio === '16:9'}
                onChange={(e) => setAspectRatio(e.target.value)}
                disabled={generating}
                className="mr-2"
              />
              <span>16:9 (Desktop)</span>
            </label>
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                value="9:16"
                checked={aspectRatio === '9:16'}
                onChange={(e) => setAspectRatio(e.target.value)}
                disabled={generating}
                className="mr-2"
              />
              <span>9:16 (Mobile)</span>
            </label>
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                value="both"
                checked={aspectRatio === 'both'}
                onChange={(e) => setAspectRatio(e.target.value)}
                disabled={generating}
                className="mr-2"
              />
              <span>Both</span>
            </label>
          </div>
        </div>

        <button
          onClick={generateStory}
          disabled={generating}
          className="w-full p-4 bg-green-600 hover:bg-green-700 rounded-lg font-bold text-lg disabled:bg-gray-600 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {generating ? (
            <>
              <Loader className="animate-spin" size={24} />
              Generating...
            </>
          ) : (
            'Generate'
          )}
        </button>

        {progress && (
          <div className="mt-6 p-4 bg-blue-900 rounded-lg text-center">
            <p className="text-lg">{progress}</p>
          </div>
        )}

        {(videoBlobs.wide || videoBlobs.tall) && (
          <div className="mt-6 bg-gray-800 p-6 rounded-lg">
            <h2 className="text-2xl font-bold mb-4">Download Your Videos:</h2>
            <div className="flex gap-4">
              {videoBlobs.wide && (
                <button
                  onClick={() => downloadVideo(videoBlobs.wide, '16-9')}
                  className="flex-1 p-4 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center justify-center gap-2"
                >
                  <Download size={20} />
                  16:9 Desktop
                </button>
              )}
              {videoBlobs.tall && (
                <button
                  onClick={() => downloadVideo(videoBlobs.tall, '9-16')}
                  className="flex-1 p-4 bg-purple-600 hover:bg-purple-700 rounded-lg flex items-center justify-center gap-2"
                >
                  <Download size={20} />
                  9:16 Mobile
                </button>
              )}
            </div>
          </div>
        )}

        {scenes.length > 0 && (
          <div className="mt-6 bg-gray-800 p-6 rounded-lg">
            <h2 className="text-xl font-bold mb-4">Generated Scenes:</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {scenes.map((scene, i) => (
                <div key={i} className="bg-gray-700 p-2 rounded">
                  <img src={scene.imageUrl} alt={`Scene ${i + 1}`} className="w-full rounded mb-2" />
                  <p className="text-xs">{scene.script.substring(0, 50)}...</p>
                </div>
              ))}
            </div>
          </div>
        )}

        <canvas ref={canvasRef} className="hidden" />
      </div>
    </div>
  );
}
