import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA # Import PCA

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
    embeddings_filepath = 'data/embeddings/whisper_embeddings.npy' # Path to save/load embeddings
    max_duration = 30.0
    n_components_pca = 10 # Number of PCA components

    if os.path.exists(embeddings_filepath):
        print(f"Loading embeddings from {embeddings_filepath}")
        embeddings = np.load(embeddings_filepath)
    else:
        print("Computing embeddings...")
        inputs = preprocess_audio(audio_filepath, feature_extractor, max_duration)
        model = model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        embeddings = hidden_states.squeeze(0).cpu().numpy()
        np.save(embeddings_filepath, embeddings) # Save embeddings

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components_pca)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Emotion Inference (same as before)
    inputs = preprocess_audio(audio_filepath, feature_extractor, max_duration) # Need to preprocess again for emotion inference
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs) # No need for hidden states here, just logits
    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]

    # Heatmap of PCA Reduced Embeddings
    plt.figure(figsize=(12, 8))
    plt.imshow(reduced_embeddings.T, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis') # Use PCA reduced embeddings
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time Step (Audio Frames)")
    plt.ylabel(f"PCA Components (Top {n_components_pca})") # Updated Y-axis label
    plt.title(f"Heatmap of Whisper Embeddings (PCA - Top {n_components_pca} Components)") # Updated title
    plt.savefig("whisper_embeddings_pca_heatmap.png") # Updated filename
    plt.close()

    print("Heatmap of Whisper embeddings (PCA reduced) saved as whisper_embeddings_pca_heatmap.png") # Updated print message
    print(f"{predicted_label} emotion detected from audio file.")
