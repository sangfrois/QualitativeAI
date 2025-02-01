import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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

    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    embeddings = hidden_states.squeeze(0).cpu().numpy()

    # Dimensionality Reduction with PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Emotion Inference (same as before)
    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]

    # Scatter Plot of Reduced Embeddings
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5, s=10) # Reduced point size
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Whisper Embeddings") # Updated title
    plt.grid(True)
    plt.savefig("whisper_embeddings_pca_scatter.png") # Updated filename
    plt.close()

    print("PCA scatter plot of Whisper embeddings saved as whisper_embeddings_pca_scatter.png") # Updated print message
    print(f"{predicted_label} emotion detected from audio file.")
