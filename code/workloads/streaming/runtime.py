import sys
import os
import numpy as np
import time
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import librosa
from datetime import timedelta
from mtcnn import MTCNN
import cv2
from deepface import DeepFace
import torch
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# initialize the models
global r, audio_emb_model, text_emb_model, detector

r = sr.Recognizer()

# Audio embedding
print("Load the audio embedding model...")
audio_emb_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
audio_emb_model.eval()
print("Audio embedding model loaded.")

# GloVe for text embedding
# We just need to run this code once, the function glove2word2vec saves the Glove embeddings in the word2vec format 
# that will be loaded in the next section
glove_dir = "./cache/glove"
glove_filename = os.path.join(glove_dir, 'glove.6B.300d.txt')
if not os.path.isfile(glove_filename):
    print(f"{glove_filename} doesn't exist. Need to be downloaded from the website.")
else:
    word2vec_output_file = glove_filename+'.word2vec'
    if not os.path.isfile(word2vec_output_file):
        print(f"{word2vec_output_file} doesn't exist. Call glove2word2vec() to save the Glove embeddings in the word2vec format...")
        glove2word2vec(glove_filename, word2vec_output_file)
    else:
        # load the Stanford GloVe model
        print("Load the Stanford GloVe model...")
        text_emb_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        print("Stanford GloVe model loaded.")

detector = MTCNN()

def convert_video_to_audio(video_file, output_ext="wav"):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    filename, ext = os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    outputname = f"{filename}.{output_ext}"
    clip.audio.write_audiofile(outputname)
    return outputname

def audio_to_text(audio_file):
    duration = librosa.get_duration(filename=audio_file)
    with sr.AudioFile(audio_file) as source:
         audio = r.record(source, duration=int(duration+1))
    text = r.recognize_google(audio)
    return text, duration

def get_runtime_transcribing(video_file):
    start_time = time.time()
    audio_file = convert_video_to_audio(video_file)
    text, duration = audio_to_text(audio_file)
    runtime = time.time() - start_time
    return runtime, duration

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")

def embed_face(video_file, detector, frames_per_second=1, verbose=False):
    # runtime
    total_det_runtime = 0.0
    total_emb_runtime = 0.0
    # load the video clip
    video_clip = VideoFileClip(video_file)
    # make a folder by the name of the video file
    filename, _ = os.path.splitext(video_file)
    filename += "-moviepy"
    if not os.path.isdir(filename):
        os.mkdir(filename)

    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(video_clip.fps, frames_per_second)
    # if SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
    step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second
    # iterate over each possible frame
    for current_duration in np.arange(0, video_clip.duration, step):
        # format the file name and save it
        frame_duration_formatted = format_timedelta(timedelta(seconds=current_duration)).replace(":", "-")
        frame_filename = os.path.join(filename, f"frame{frame_duration_formatted}.jpg")
        
        # save the frame with the current duration
        video_clip.save_frame(frame_filename, current_duration)
        
        det_start_time = time.time()
        img = cv2.imread(frame_filename)
        detections = detector.detect_faces(img)
        
        for detection in detections:
            score = detection["confidence"]
            if score > 0.9:
                x, y, w, h = detection["box"]
                detected_face = img[int(y):int(y+h), int(x):int(x+w)]
        det_runtime = time.time() - det_start_time
        total_det_runtime += det_runtime
        if verbose:
            print("Face detection with MTCNN runtime: ", det_runtime)
        
        #obj = DeepFace.analyze(img_path = frame_filename, actions = ['emotion'])
        emb_start_time = time.time()
        embedding = DeepFace.represent(img_path = frame_filename, model_name = "DeepFace")
        emb_runtime = time.time() - emb_start_time
        total_emb_runtime += emb_runtime
        if verbose:
            print("Face embedding with DeepFace runtime: ", emb_runtime)
    
    return total_det_runtime, total_emb_runtime

def get_runtime_embeddings(video_file, audio_file=None, text=None, frames_per_second=1, verbose=False):
    start_time = time.time()
    if audio_file is None:
        audio_file = convert_video_to_audio(video_file)
        if verbose:
            print("Finished extracting audio from video.")
    if text is None:
        text, duration = audio_to_text(audio_file)
        if verbose:
            print("Finished extracting text from audio.")
    parse_runtime = time.time() - start_time
    
    if verbose:
        print("Start embedding.")
    audio_emb_start_time = time.time()
    audio_emb = audio_emb_model.forward(audio_file)
    audio_emb_runtime = time.time() - audio_emb_start_time
    
    text_emb_start_time = time.time()
    for word in text.split(" "):
        try:
            text_emb_model.get_vector(word)
        except:
            pass
    text_emb_runtime = time.time() - text_emb_start_time
    
    if verbose:
        print("Visual embedding.")
    vis_det_runtime, vis_emb_runtime = embed_face(video_file, detector, frames_per_second, verbose=False)
        
    runtime = time.time() - start_time
    if verbose:
        print(f"Total runtime: {runtime} seconds.")
        print(f"Parsing runtime: {parse_runtime} seconds.")
        print(f"Audio embedding runtime: {audio_emb_runtime} seconds.")
        print(f"Text embedding: {text_emb_runtime} seconds.")
        print(f"Face detection runtime: {vis_det_runtime} seconds.")
        print(f"Visual embedding: {vis_emb_runtime} seconds.")
    
    return runtime, parse_runtime, audio_emb_runtime, text_emb_runtime, vis_det_runtime, vis_emb_runtime

if __name__ == "__main__":
    frames_per_second = 30
    video_file = './cache/person.mp4'
    #text = "hey everyone it's Steve here from the emotion machine.com and this video is going to be about the power of talking to yourself in the third person now we will talk to herself throughout the day we all have an inner voice that sort of narrates our life as we go about it and naturally when we talk to ourselves we use a first-person perspective we usually use the word I so say you make a mistake"
    get_runtime_embeddings(video_file, frames_per_second = 30, verbose=True)
