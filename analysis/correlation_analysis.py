import numpy as np
import pandas as pd
import os
import glob
import json
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def load_embeddings(embeddings_filepath):
    """Loads PCA embeddings from a .npy file."""
    if not os.path.exists(embeddings_filepath):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_filepath}")
    return np.load(embeddings_filepath)

def load_transcripts_and_compute_sentiments(json_files_path):
    """Loads transcripts and computes sentiment scores for each segment."""
    json_files = glob.glob(os.path.join(json_files_path, '*.json'))
    if not json_files:
        raise FileNotFoundError(f"No JSON transcript files found in: {json_files_path}")

    sia = SentimentIntensityAnalyzer()
    all_sentiments = {}
    for json_file in json_files:
        filename = os.path.basename(json_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)

        sentiments = []
        current_time = 0
        for segment in transcript_data['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            sentiment_score = sia.polarity_scores(text)['compound'] # Compute sentiment score here
            sentiments.append({'start': start_time, 'end': end_time, 'sentiment': sentiment_score})
        all_sentiments[filename] = sentiments
    return all_sentiments

def align_sentiments_to_embeddings(embeddings, all_sentiments, chunk_duration=30.0, overlap_ratio=0.5, sampling_rate=16000): # Assuming 16kHz sampling rate, adjust if needed
    """
    Aligns sentiment scores to embedding chunks by averaging sentiment scores within each chunk.
    """
    chunk_samples = int(chunk_duration * sampling_rate)
    overlap_samples = int(chunk_samples * overlap_ratio)
    step_samples = chunk_samples - overlap_samples
    aligned_sentiments = []
    num_chunks = len(embeddings)

    # Assuming we are using sentiments from the first transcript file.
    # You might need to adjust this if you have multiple transcript files and need specific matching.
    transcript_filename = list(all_sentiments.keys())[0]
    sentiments = all_sentiments[transcript_filename]


    for chunk_index in range(num_chunks):
        start_sample = chunk_index * step_samples
        start_time = start_sample / sampling_rate
        end_time = start_time + chunk_duration

        chunk_sentiments = []
        for sent_data in sentiments:
            if start_time <= sent_data['start'] < end_time or start_time < sent_data['end'] <= end_time or (sent_data['start'] <= start_time and sent_data['end'] >= end_time):
                chunk_sentiments.append(sent_data['sentiment'])

        if chunk_sentiments:
            aligned_sentiments.append(np.mean(chunk_sentiments)) # Average sentiment for the chunk
        else:
            aligned_sentiments.append(np.nan) # No sentiment in this chunk, use NaN

    return aligned_sentiments


def compute_correlation(embeddings, aligned_sentiments):
    """Computes Pearson correlation between each PCA component and aligned sentiment."""
    num_components = embeddings.shape[1]
    correlations = []
    for i in range(num_components):
        component_embedding = embeddings[:, i]
        # Handle NaN values in sentiment by using only non-NaN pairs
        valid_indices = ~np.isnan(aligned_sentiments)
        if np.sum(valid_indices) < 2: # Need at least 2 points for correlation
            correlation = np.nan # Not enough valid data points
        else:
            correlation, _ = pearsonr(component_embedding[valid_indices], np.array(aligned_sentiments)[valid_indices])
        correlations.append(correlation)
    return correlations

def plot_correlation(correlations, n_components_pca):
    """Plots the correlation coefficients for each PCA component."""
    plt.figure(figsize=(10, 6))
    components = range(1, n_components_pca + 1)
    plt.bar(components, correlations)
    plt.xlabel("PCA Component")
    plt.ylabel("Pearson Correlation with Sentiment")
    plt.title("Correlation between PCA Components and Sentiment")
    plt.xticks(components)
    plt.grid(axis='y')
    plt.savefig("pca_sentiment_correlation.png")
    plt.close()
    print("Correlation plot saved as pca_sentiment_correlation.png")


def main():
    embeddings_filepath = 'data/embeddings/whisper_chunked_mean_pca_embeddings.npy' # Path to embeddings
    json_files_path = 'data/psilocybin/processed/' # Path to processed transcript JSONs
    n_components_pca = 10 # Number of PCA components, should match embedding_visualization.py

    try:
        embeddings = load_embeddings(embeddings_filepath)
        all_sentiments = load_transcripts_and_compute_sentiments(json_files_path) # Use the new function

        aligned_sentiments = align_sentiments_to_embeddings(embeddings, all_sentiments) # Pass all_sentiments

        correlations = compute_correlation(embeddings, aligned_sentiments)

        print("\nCorrelation coefficients between PCA components and sentiment:")
        for i, corr in enumerate(correlations):
            print(f"PCA Component {i+1}: {corr:.3f}")

        plot_correlation(correlations, n_components_pca)


    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    nltk.download('vader_lexicon')
    main()
