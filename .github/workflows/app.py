import os
import cv2
import torch
import numpy as np
import librosa
import tempfile
from moviepy.editor import VideoFileClip, AudioFileClip
import face_alignment
import gradio as gr

# --- CONFIG ---
DEVICE = 'cpu'
BATCH_SIZE = 1
PADS = [-15, 15, -15, 15]
FPS = 25
WEIGHTS_PATH = 'checkpoints/wav2lip_gan.pth'
PREDICTOR_PATH = 'checkpoints/shape_predictor_68_face_landmarks.dat'

# --- LOAD MODEL & DETECTOR ---
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=DEVICE)
model = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.eval().to(DEVICE)

def get_landmarks(image):
    preds = fa.get_landmarks_from_image(image)
    return preds[0] if preds else None

def process_audio(audio_path):
    wav, sr = librosa.load(audio_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80).T
    mel = (mel - np.mean(mel)) / np.std(mel)
    return mel

def generate_talking_head(image_path, audio_path):
    img = cv2.imread(image_path)
    if img is None: raise gr.Error("Image not found")
    lmk = get_landmarks(img)
    if lmk is None: raise gr.Error("Face not detected in image")

    # Crop & Pad
    h, w = img.shape[:2]
    y1, y2, x1, x2 = min(lmk[:,1])-PADS[0], max(lmk[:,1])+PADS[1], min(lmk[:,0])-PADS[2], max(lmk[:,0])+PADS[3]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    face = img[y1:y2, x1:x2]
    face_resized = cv2.resize(face, (96, 96))

    # Audio
    mel = process_audio(audio_path)
    n_frames = mel.shape[0]
    
    # Inference Loop
    out_frames = []
    with torch.no_grad():
        for i in range(0, n_frames, BATCH_SIZE):
            batch_mel = mel[i:i+BATCH_SIZE].unsqueeze(0).float().to(DEVICE)
            batch_face = cv2.resize(face_resized, (96, 96)) / 255.0
            batch_face = torch.tensor(batch_face.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
            
            pred = model(batch_mel, batch_face)
            pred = pred.squeeze().cpu().numpy().transpose(1, 2, 0)
            pred = np.clip(pred * 255, 0, 255).astype(np.uint8)
            
            # Paste back
            h_f, w_f = face.shape[:2]
            pred_resized = cv2.resize(pred, (w_f, h_f))
            img_out = img.copy()
            img_out[y1:y2, x1:x2] = pred_resized
            out_frames.append(img_out)

    # Save temp video
    temp_video = "temp_no_audio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, FPS, (img.shape[1], img.shape[0]))
    for f in out_frames:
        out.write(f)
    out.release()

    # Merge Audio
    final_video = "output_result.mp4"
    video_clip = VideoFileClip(temp_video)
    audio_clip = AudioFileClip(audio_path)
    final = video_clip.set_audio(audio_clip)
    final.write_videofile(final_video, codec='libx264', audio_codec='aac')
    
    os.remove(temp_video)
    return final_video

# --- DEMO MODE FOR ACTIONS ---
if __name__ == "__main__" and "--demo" in __import__("sys").argv:
    import urllib.request
    urllib.request.urlretrieve("https://picsum.photos/id/1025/800/800", "demo_img.jpg")
    urllib.request.urlretrieve("https://www2.cs.uic.edu/~i101/SoundFiles/taunt.wav", "demo_audio.wav")
    print("⏳ Generating demo video...")
    res = generate_talking_head("demo_img.jpg", "demo_audio.wav")
    print(f"✅ Saved to {res}")

# --- GRADIO INTERFACE (LOCAL USE) ---
with gr.Blocks(title="Wav2Lip CPU") as demo:
    gr.Markdown("# 🎬 Wav2Lip Talking Head (CPU Mode)")
    with gr.Row():
        img_in = gr.Image(type="filepath", label="Face Image")
        aud_in = gr.Audio(type="filepath", label="Voice Audio")
    btn = gr.Button("Generate", variant="primary")
    out_vid = gr.Video(label="Result")
    btn.click(generate_talking_head, inputs=[img_in, aud_in], outputs=out_vid)

if __name__ == "__main__":
    demo.launch(share=False)
