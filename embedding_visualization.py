import torch
from transformers import AutoModelForAudioClassification, AutoProcessor
import librosa
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)

    audio_filepath = 'data/audio/interview_audio.wav'  # <--  ***CHANGE THIS TO YOUR AUDIO FILE PATH***
    audio_input, sample_rate = librosa.load(audio_filepath, sr=16000)

    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    embeddings = hidden_states.squeeze(0).cpu().numpy()

    plt.figure(figsize=(10, 6))
    time_steps = np.arange(embeddings.shape[0])
    for dim in range(min(3, embeddings.shape[1])):
        plt.plot(time_steps, embeddings[:, dim], label=f'Dimension {dim}')
    plt.xlabel("Time Step (Audio Frames)")
    plt.ylabel("Embedding Value")
    plt.title("Time Series of Wav2Vec2 Embeddings")
    plt.legend()
    plt.grid(True)
    plt.savefig("wav2vec2_embeddings_timeseries.png")
    plt.close()

    print("Embedding time-series plot saved as wav2vec2_embeddings_timeseries.png")
