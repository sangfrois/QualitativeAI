import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np
import glob
import os

def load_transcript(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        utterances = []
        for segment in data['segments']:
            utterances.append({
                'text': segment['text'],
                'speaker': segment['speaker'],
                'metadata': {'timestamp': {'start': segment['start'], 'end': segment['end']}} if 'start' in segment and 'end' in segment else {}
            })
        return utterances
    except Exception as e:
        print(f"Error in load_transcript: {e}")
        return []

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

def standardize_speaker(speaker):
    speaker = speaker.strip().rstrip(':')
    return speaker # Keep original speaker label if not SPEAKER_01 or SPEAKER_00

def process_transcript(transcript):
    try:
        for i, utterance in enumerate(transcript):
            utterance['sentiment'] = analyze_sentiment(utterance['text'])
            utterance['speaker'] = standardize_speaker(utterance['speaker'])
            utterance['sequence'] = i + 1 # Assign sequence number based on order
        return transcript
    except Exception as e:
        print(f"Error in process_transcript: {e}")
        return []

def load_emotion_results(emotion_results_filepath):
    """Loads emotion results from CSV."""
    try:
        if not os.path.exists(emotion_results_filepath):
            raise FileNotFoundError(f"Emotion results file not found: {emotion_results_filepath}")
        emotion_df = pd.read_csv(emotion_results_filepath)
        print(f"Emotion results loaded successfully from {emotion_results_filepath}") # Debug print
        return emotion_df
    except FileNotFoundError as e:
        print(f"FileNotFoundError in load_emotion_results: {e}")
        return pd.DataFrame() # Return empty DataFrame to avoid further errors
    except pd.errors.EmptyDataError as e:
        print(f"EmptyDataError in load_emotion_results: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error in load_emotion_results: {e}")
        return pd.DataFrame()

def align_emotions_to_utterances(transcripts, emotion_results_df):
    """Aligns emotion predictions to utterances based on timestamps."""
    aligned_emotions = {}
    try:
        for filename, utterances in transcripts.items():
            print(f"Processing filename: {filename}") # Debug print
            emotions_for_file = emotion_results_df[emotion_results_df['filename'] == os.path.splitext(filename)[0]] # Match filename
            utterance_emotions = []
            for utterance in utterances:
                utterance_start_time = utterance['metadata'].get('timestamp', {}).get('start', 0)
                utterance_end_time = utterance['metadata'].get('timestamp', {}).get('end', 0)
                print(f"  Utterance start: {utterance_start_time}, end: {utterance_end_time}") # Debug print

                # Find emotion prediction that overlaps with the utterance timestamp
                overlapping_emotion = emotions_for_file[
                    (emotions_for_file['timestamp'] <= utterance_start_time) & # Emotion chunk starts before utterance
                    (emotions_for_file['timestamp'] + 30 >= utterance_end_time) # Emotion chunk ends after utterance (assuming 30s chunks)
                ]
                print(f"  Overlapping emotions:\n{overlapping_emotion}") # Debug print

                if not overlapping_emotion.empty:
                    # Take the first overlapping emotion (if multiple, which shouldn't happen ideally)
                    predicted_emotion = overlapping_emotion.iloc[0]['emotion']
                    utterance_emotions.append({'sequence': utterance['sequence'], 'emotion': predicted_emotion})
                else:
                    utterance_emotions.append({'sequence': utterance['sequence'], 'emotion': None}) # No emotion for this utterance

            aligned_emotions[filename] = utterance_emotions
        return aligned_emotions
    except Exception as e:
        print(f"Error in align_emotions_to_utterances: {e}")
        return {}


def plot_sentiment_analysis(transcripts, aligned_emotions, smoothing_window=150): # Added aligned_emotions
    plt.figure(figsize=(16, 7))

    all_speakers = set()
    for utterances in transcripts.values():
        all_speakers.update(utterance['speaker'] for utterance in utterances)

    color_map = {'Patient': 'blue', 'Interviewer': 'green'}
    other_speakers = sorted([speaker for speaker in all_speakers if speaker not in ['Patient', 'Interviewer']])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(other_speakers)+1))
    color_map.update(dict(zip(other_speakers, colors)))

    emotion_markers = {
        'neutral':         'o',  # Circle
        'calm':            's',  # Square
        'happy':           '^',  # Triangle up
        'sad':             'v',  # Triangle down
        'angry':           '*',  # Star
        'fearful':         'p',  # Pentagon
        'disgust':         'h',  # Hexagon
        'surprised':       'H'   # Rotated Hexagon
    }
    emotion_colors = { # Distinct colors for emotions
        'neutral':         'gray',
        'calm':            'lightgreen',
        'happy':           'yellow',
        'sad':             'lightblue',
        'angry':           'red',
        'fearful':         'purple',
        'disgust':         'brown',
        'surprised':       'orange'
    }

    try:
        for i, (file, utterances) in enumerate(transcripts.items(), 1):
            plt.subplot(1, 2, i)
            ax = plt.gca() # Get current axes
            df = pd.DataFrame(utterances)
            df['sequence'] = pd.to_numeric(df['sequence'], errors='coerce')
            df = df.dropna(subset=['sequence', 'sentiment'])
            df = df.sort_values('sequence')

            # Get aligned emotions for this file
            emotions_for_plot = aligned_emotions.get(file, []) # Use .get() to avoid KeyError if file not in aligned_emotions
            emotion_df = pd.DataFrame(emotions_for_plot)

            for speaker in df['speaker'].unique():
                speaker_data = df[df['speaker'] == speaker]
                speaker_data['smoothed_sentiment'] = speaker_data['sentiment'].rolling(window=smoothing_window, min_periods=1, center=True).mean() # Apply smoothing

                plt.plot(speaker_data['sequence'], speaker_data['smoothed_sentiment'], linestyle='-', label=speaker, color=color_map[speaker]) # Plot smoothed sentiment

                # Overlay emotion markers
                for _, emotion_point in emotion_df.iterrows():
                    sequence_to_mark = emotion_point['sequence']
                    emotion_label = emotion_point['emotion']
                    if emotion_label and sequence_to_mark in speaker_data['sequence'].values: # Ensure sequence exists for this speaker
                        sentiment_value = speaker_data[speaker_data['sequence'] == sequence_to_mark]['smoothed_sentiment'].iloc[0] # Get sentiment at that sequence
                        marker = emotion_markers.get(emotion_label, 'x') # Default marker if emotion not in dict
                        markercolor = emotion_colors.get(emotion_label, 'black') # Default color if emotion not in dict
                        plt.scatter(sequence_to_mark, sentiment_value, marker=marker, color=markercolor, s=50, label=f"Emotion: {emotion_label}" if ax.get_legend() is not None and emotion_label not in [text.get_text() for text in ax.get_legend().get_texts()] else None) # Add marker, label only once per emotion


            plt.title(f"Sentiment Analysis - {file}")
            plt.xlabel("Utterance Index") # Updated X-axis label to Utterance Index
            plt.ylabel("Sentiment Score (Smoothed)") # Updated Y-axis label to indicate smoothing

            # Get existing legend handles and labels to avoid duplicates
            handles, labels = ax.get_legend_handles_labels() # Use ax to get legend handles

            by_label = dict()
            sentiment_handles = []
            emotion_handles = []
            for handle, label in zip(handles, labels):
                if "Speaker" in ax.get_legend().get_title().get_text() and "Speaker" in label: # Check if legend title is "Speaker"
                    if label not in by_label and "Speaker" in label:
                        by_label[label] = handle
                        sentiment_handles.append((handle, label))
                elif "Emotion" in label:
                    emotion_name = label.split(': ')[1]
                    if emotion_name not in [l.split(': ')[1] for _, l in emotion_handles]: # Avoid duplicate emotion labels
                        emotion_handles.append((handle, label))

            # Combine sentiment and emotion legends
            combined_handles_labels = [hl for hl in sentiment_handles] + [hl for hl in emotion_handles]
            final_handles, final_labels = zip(*combined_handles_labels) if combined_handles_labels else ([], [])
            plt.legend(final_handles, final_labels, title="Legend", loc='best') # Updated legend title

        plt.tight_layout()
        plt.savefig("sentiment_and_emotion_analysis_smoothed.png", dpi=300) # Changed filename to indicate emotions
        plt.close()
    except Exception as e:
        print(f"Error in plot_sentiment_analysis: {e}")


def main():
    json_files_path = 'data/psilocybin/processed/'
    emotion_results_filepath = 'data/emotion_results.csv' # Path to emotion results CSV
    json_files = glob.glob(os.path.join(json_files_path, '*.json'))
    transcripts = {}

    try:
        emotion_results_df = load_emotion_results(emotion_results_filepath) # Load emotion results
        if emotion_results_df.empty: # Check if emotion_results_df is empty
            print("Emotion results DataFrame is empty. Please check if emotion results were generated correctly.")
            return

        for json_file in json_files:
            try:
                transcript = load_transcript(json_file)
                processed_transcript = process_transcript(transcript)
                transcripts[os.path.basename(json_file)] = processed_transcript # Use filename as key
                print(f"\nProcessed {json_file}:")
                print(f"- Total utterances: {len(processed_transcript)}")
                if processed_transcript:
                    print("\nSample of utterances with sentiment:")
                    for i, u in enumerate(processed_transcript[:5]):
                        print(f"\n{i+1}. Speaker: {u['speaker']}")
                        print(f"   Text: {u['text'][:50]}...")
                        print(f"   Sentiment: {u['sentiment']:.2f}")
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

        aligned_emotions = align_emotions_to_utterances(transcripts, emotion_results_df) # Align emotions to utterances
        if not aligned_emotions: # Check if aligned_emotions is empty
            print("No emotions aligned. Please check the alignment process and timestamps.")
            return

        plot_sentiment_analysis(transcripts, aligned_emotions, smoothing_window=7) # Call plot_sentiment_analysis with aligned emotions
        print("\nSentiment and emotion analysis plot saved as sentiment_and_emotion_analysis_smoothed.png") # Updated saved filename in print statement

    except FileNotFoundError as e:
        print(f"File not found error in main: {e}")
    except Exception as e:
        print(f"An error occurred in main: {e}")


if __name__ == "__main__":
    nltk.download('vader_lexicon')
    main()
