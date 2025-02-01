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

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

def standardize_speaker(speaker):
    speaker = speaker.strip().rstrip(':')
    if speaker.startswith('SPEAKER_01'):
        return 'Interviewer'
    elif speaker.startswith('SPEAKER_00'):
        return 'Patient'
    return speaker # Keep original speaker label if not SPEAKER_01 or SPEAKER_00

def process_transcript(transcript):
    for i, utterance in enumerate(transcript):
        utterance['sentiment'] = analyze_sentiment(utterance['text'])
        utterance['speaker'] = standardize_speaker(utterance['speaker'])
        utterance['sequence'] = i + 1 # Assign sequence number based on order
    return transcript

def plot_sentiment_analysis(transcripts):
    plt.figure(figsize=(20, 15))

    all_speakers = set()
    for utterances in transcripts.values():
        all_speakers.update(utterance['speaker'] for utterance in utterances)

    color_map = {'Patient': 'red', 'Interviewer': 'blue'} # Define colors for Patient and Interviewer
    other_speakers = sorted([speaker for speaker in all_speakers if speaker not in ['Patient', 'Interviewer']])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(other_speakers)+1))
    color_map.update(dict(zip(other_speakers, colors)))

    for i, (file, utterances) in enumerate(transcripts.items(), 1):
        plt.subplot(3, 2, i)
        df = pd.DataFrame(utterances)
        df['sequence'] = pd.to_numeric(df['sequence'], errors='coerce')
        df = df.dropna(subset=['sequence', 'sentiment'])
        df = df.sort_values('sequence')

        # Removed timestamp related code
        # timestamps = []
        # for idx, row in df.iterrows():
        #     if 'timestamp' in row['metadata'] and row['metadata']['timestamp']:
        #         ts = row['metadata']['timestamp']
        #         if 'start' in ts and ts['start'] is not None: # Check if start is not None
        #             timestamps.append((row['sequence'], ts['start']))
        #             start_seq = row['sequence']
        #             end_seq = row['sequence'] + 1
        #             plt.axvspan(start_seq, end_seq, color='gray', alpha=0.2, label='Timestamped Segment' if not plt.gca().get_legend() else None) # Label only once


        for speaker in df['speaker'].unique():
            speaker_data = df[df['speaker'] == speaker]
            plt.plot(speaker_data['sequence'], speaker_data['sentiment'], marker='o', linestyle='-', label=speaker, color=color_map[speaker])

        # Removed timestamp related code
        # if timestamps:
        #     tick_positions, tick_labels = zip(*timestamps)
        #     tick_labels = [f"{float(ts):.2f}" for ts in tick_labels] # Format timestamps to 2 decimal places
        #     plt.xticks(tick_positions, tick_labels, rotation=45, ha='right') # Rotate and align x-axis labels

        plt.title(f"Sentiment Analysis - {file}")
        plt.xlabel("Utterance Index") # Updated X-axis label to Utterance Index
        plt.ylabel("Sentiment Score")

        # Get existing legend handles and labels to avoid duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles)) # Remove duplicate labels
        plt.legend(by_label.values(), by_label.keys(), title="Speaker/Time Info", loc='best') # Updated legend title

    plt.tight_layout()
    plt.savefig("sentiment_analysis_time.png", dpi=300)
    plt.close()

def main():
    json_files_path = 'data/psilocybin/processed/'
    json_files = glob.glob(os.path.join(json_files_path, '*.json'))
    transcripts = {}

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

    plot_sentiment_analysis(transcripts)
    print("\nSentiment analysis plot saved as sentiment_analysis_time.png")

if __name__ == "__main__":
    nltk.download('vader_lexicon')
    main()
