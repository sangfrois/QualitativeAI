import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import matplotlib.pyplot as plt
import numpy as np

def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, do_normalize=True)
    id2label = model.config.id2label

    audio_filepath = 'data/psilocybin/audio/Kesem_00.mp4' 
    max_duration = 30.0
    inputs = preprocess_audio(audio_filepath, feature_extractor, max_duration)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    embeddings = hidden_states.squeeze(0).cpu().numpy()
    # infer emotions
    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]

    plt.figure(figsize=(12, 8))
    plt.imshow(embeddings.T, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time Step (Audio Frames)")
    plt.ylabel("Embedding Dimension")
    plt.title("Heatmap of Wav2Vec2 Embeddings")
    plt.savefig("wav2vec2_embeddings_heatmap.png")
    plt.close()

    print("Embedding time-series plot saved as wav2vec2_embeddings_timeseries.png")
    print(f"{predicted_label} emotion detected from audio file.")