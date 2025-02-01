import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import pandas as pd
import glob # Import glob

def preprocess_audio_chunk(audio_chunk, sampling_rate, feature_extractor):
    inputs = feature_extractor(
        audio_chunk,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    return inputs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, do_normalize=True)
    id2label = model.config.id2label

    # Initialize emotion results DataFrame outside the loop
    emotion_df = pd.DataFrame(columns=['filename', 'emotion', 'timestamp'])
    all_embeddings = {} # Dictionary to store embeddings for each file

    audio_dir = 'data/psilocybin/audio/' # Directory containing audio files
    audio_files = glob.glob(os.path.join(audio_dir, '*.mp4')) # Get all .mp4 files

    for audio_filepath in audio_files: # Loop through each audio file
        filename = os.path.basename(audio_filepath).split('.')[0]
        embeddings_filepath = os.path.join('data', 'embeddings', f'whisper_chunked_mean_pca_embeddings_{filename}.npy') # Unique embeddings file for each audio
        heatmap_filepath = f"whisper_chunked_mean_pca_heatmap_{filename}.png" # Unique heatmap file for each audio

        max_duration = 60.0
        n_components_pca = 10

        print(f"Preprocessing audio: {audio_filepath}")
        audio_array, sr = librosa.load(audio_filepath, sr=feature_extractor.sampling_rate)

        mean_pca_embeddings_list = [] # Reset embeddings list for each file

        print("Computing chunked embeddings...")
        chunk_duration = 30.0
        overlap_ratio = 0.5
        chunk_samples = int(chunk_duration * feature_extractor.sampling_rate)
        overlap_samples = int(chunk_samples * overlap_ratio)
        step_samples = chunk_samples - overlap_samples

        for start_sample in range(0, len(audio_array) - chunk_samples + 1, step_samples):
            end_sample = start_sample + chunk_samples
            audio_chunk = audio_array[start_sample:end_sample]

            inputs = preprocess_audio_chunk(audio_chunk, feature_extractor.sampling_rate, feature_extractor)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            logits = outputs.logits
            predicted_id = torch.argmax(logits, dim=-1).item()
            predicted_label = id2label[predicted_id]

            # Append emotion result to DataFrame, including filename
            emotion_df.loc[len(emotion_df)] = [filename, predicted_label, start_sample / sr]

            hidden_states = outputs.hidden_states[-1]
            chunk_embeddings = hidden_states.squeeze(0).cpu().numpy()

            pca = PCA(n_components=n_components_pca)
            reduced_chunk_embeddings = pca.fit_transform(chunk_embeddings)
            mean_pca_embedding = np.mean(reduced_chunk_embeddings, axis=0)
            mean_pca_embeddings_list.append(mean_pca_embedding)

        mean_pca_embeddings_concatenated = np.array(mean_pca_embeddings_list)
        all_embeddings[filename] = mean_pca_embeddings_concatenated # Store embeddings in dictionary
        np.save(embeddings_filepath, mean_pca_embeddings_concatenated)

        # Plotting Heatmap for each file
        plt.figure(figsize=(12, 6))
        plt.imshow(mean_pca_embeddings_concatenated.T, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
        plt.colorbar(format='%+2.0f')
        plt.xlabel("Time Chunk Index (30s chunks with 50% overlap)")
        plt.ylabel("Mean PCA Component")
        plt.yticks(np.arange(n_components_pca), [f'PC{i+1}' for i in range(n_components_pca)])
        plt.title(f"Heatmap of Mean PCA Embeddings - {filename}")
        plt.savefig(heatmap_filepath)
        plt.close()

        print(f"Heatmap of Whisper embeddings (PCA components) saved as {heatmap_filepath}")
        print(f"Predicted emotion for {filename}: {predicted_label}") # Use last predicted label, consider averaging or majority vote if needed

    emotion_results_filepath = 'data/emotion_results.csv' # Path to save combined emotion results
    emotion_df.to_csv(emotion_results_filepath, index=False) # Save all emotion results to a single CSV
    print(f"Emotion results for all files saved to {emotion_results_filepath}")
