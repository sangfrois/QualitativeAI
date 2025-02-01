import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import pandas as pd # Import pandas

def preprocess_audio_chunk(audio_chunk, sampling_rate, feature_extractor):
    inputs = feature_extractor(
        audio_chunk,
        sampling_rate=sampling_rate, # Use the provided sampling_rate
        return_tensors="pt",
    )
    return inputs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, do_normalize=True)
    id2label = model.config.id2label

    # Save emotion results to CSV
    emotion_df = pd.DataFrame(columns=['emotion', 'timestamp'])

    audio_filepath = 'data/psilocybin/audio/Kesem_00.mp4' # Changed to .mp4
    embeddings_filepath = 'data/embeddings/whisper_chunked_mean_pca_embeddings.npy' # Path to save/load embeddings
    emotion_results_filepath = 'data/emotion_results.csv' # Path to save emotion results
    max_duration = 60.0
    n_components_pca = 10 # Number of PCA components

    print("Preprocessing audio...") # Indicate preprocessing start
    audio_array, sr = librosa.load(audio_filepath, sr=feature_extractor.sampling_rate) # Load audio once

    if os.path.exists(embeddings_filepath):
        print(f"Loading embeddings from {embeddings_filepath}")
        mean_pca_embeddings_concatenated = np.load(embeddings_filepath)
    else:
        print("Computing chunked embeddings...")
        chunk_duration = 30.0  # seconds
        overlap_ratio = 0.5
        chunk_samples = int(chunk_duration * feature_extractor.sampling_rate)
        overlap_samples = int(chunk_samples * overlap_ratio)
        step_samples = chunk_samples - overlap_samples

        mean_pca_embeddings_list = []
        for start_sample in range(0, len(audio_array) - chunk_samples + 1, step_samples): # Use pre-loaded audio_array
            end_sample = start_sample + chunk_samples
            audio_chunk = audio_array[start_sample:end_sample]

            inputs = preprocess_audio_chunk(audio_chunk, feature_extractor.sampling_rate, feature_extractor)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            # perform inference
            logits = outputs.logits
            predicted_id = torch.argmax(logits, dim=-1).item()
            predicted_label = id2label[predicted_id]
            # add row to emotion_df
            emotion_df.loc[len(emotion_df)] = [predicted_label, start_sample / sr] # Add row to emotion_df
            emotion_df.to_csv(emotion_results_filepath, index=False)
            print(f"Emotion  {emotion_results_filepath}") # Confirmation message

            hidden_states = outputs.hidden_states[-1]
            chunk_embeddings = hidden_states.squeeze(0).cpu().numpy()

            pca = PCA(n_components=n_components_pca)
            reduced_chunk_embeddings = pca.fit_transform(chunk_embeddings)
            mean_pca_embedding = np.mean(reduced_chunk_embeddings, axis=0) # Mean across time within the chunk
            mean_pca_embeddings_list.append(mean_pca_embedding)

        mean_pca_embeddings_concatenated = np.array(mean_pca_embeddings_list)
        np.save(embeddings_filepath, mean_pca_embeddings_concatenated) # Save embeddings

    # Plotting concatenated mean PCA embeddings as Heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(mean_pca_embeddings_concatenated.T, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
    plt.colorbar(format='%+2.0f') # Add colorbar to show value scale
    plt.xlabel("Time Chunk Index (10s chunks with 50% overlap)") # Updated x-axis label
    plt.ylabel("Mean PCA Component")
    plt.yticks(np.arange(n_components_pca), [f'PC{i+1}' for i in range(n_components_pca)]) # Set y-ticks to PCA component names
    plt.title(f"Heatmap of Mean PCA Embeddings (10s chunks, 50% overlap)") # Updated title
    plt.savefig("whisper_chunked_mean_pca_heatmap.png") # Filename for heatmap
    plt.close()

    print("Heatmap of Whisper embeddings (PCA components) saved as whisper_chunked_mean_pca_heatmap.png") # Updated print message
    print(f"Predicted emotion from audio file: {predicted_label}") # Use pre-computed predicted_label
