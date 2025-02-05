import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline

def load_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

def standardize_speaker(speaker):
    speaker = speaker.strip().rstrip(':')
    if speaker.startswith('Interviewer'):
        parts = speaker.split()
        if len(parts) > 1 and parts[1].isdigit():
            return f'Interviewer {parts[1]}'
        return 'Interviewer'
    return 'Patient' if speaker.lower().startswith('patient') else speaker

def process_transcript(transcript):
    for utterance in transcript:
        utterance['sentiment'] = analyze_sentiment(utterance['text'])
        utterance['speaker'] = standardize_speaker(utterance['speaker'])
    return transcript

def plot_sentiment_analysis(transcripts):
    plt.figure(figsize=(20, 15))
    
    all_speakers = set()
    for utterances in transcripts.values():
        all_speakers.update(utterance['speaker'] for utterance in utterances)
    
    color_map = {'Patient': 'red'}
    other_speakers = sorted([speaker for speaker in all_speakers if speaker != 'Patient'])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(other_speakers)+1))
    color_map.update(dict(zip(other_speakers, colors)))
    
    for i, (file, utterances) in enumerate(transcripts.items(), 1):
        plt.subplot(3, 2, i)
        df = pd.DataFrame(utterances)
        df['sequence'] = pd.to_numeric(df['sequence'], errors='coerce')
        df = df.dropna(subset=['sequence', 'sentiment'])
        df = df.sort_values('sequence')
        
        timestamps = []
        for idx, row in df.iterrows():
            if 'timestamp' in row['metadata'] and row['metadata']['timestamp']:
                ts = row['metadata']['timestamp']
                if 'start' in ts and ts['start']:
                    timestamps.append((row['sequence'], ts['start']))
                    start_seq = row['sequence']
                    end_seq = row['sequence'] + 1 if 'end' not in ts or ts['end'] is None else row['sequence'] + 2
                    plt.axvspan(start_seq, end_seq, color='gray', alpha=0.2)
        
        for speaker in df['speaker'].unique():
            speaker_data = df[df['speaker'] == speaker]
            x = speaker_data['sequence'].to_numpy()
            y = speaker_data['sentiment'].to_numpy()
            if len(x) > 3:
                x_smooth = np.linspace(x.min(), x.max(), 300)
                y_smooth = make_interp_spline(x, y, k=3)(x_smooth)
                plt.plot(x_smooth, y_smooth, label=speaker, color=color_map[speaker])
            else:
                plt.plot(x, y, marker='o', linestyle='-', label=speaker, color=color_map[speaker])
        
        if timestamps:
            tick_positions, tick_labels = zip(*timestamps)
            plt.xticks(tick_positions, tick_labels, rotation=45)
        
        plt.title(f"Sentiment Analysis - {file}")
        plt.xlabel("Sequence (Corresponding to Time)")
        plt.ylabel("Sentiment Score")
        plt.legend(title="Speaker", loc='best')
    
    plt.tight_layout()
    plt.savefig("sentiment_analysis_time.png", dpi=300)
    plt.close()

def main():
    json_files = ['transcript_01_time.json', 'transcript_03_time.json', 'transcript_04_time.json', 'transcript_05_time.json',
'transcript_06_time.json']
    transcripts = {}
    
    for json_file in json_files:
        try:
            transcript = load_transcript(json_file)
            processed_transcript = process_transcript(transcript)
            transcripts[json_file] = processed_transcript
            print(f"\nProcessed {json_file}:")
            print(f"- Total utterances: {len(processed_transcript)}")
            if processed_transcript:
                print("\nSample of utterances with sentiment:")
                for i, u in enumerate(processed_transcript[:5]):
                    print(f"\n{i}. Speaker: {u['speaker']}")
                    print(f"   Text: {u['text'][:50]}...")
                    print(f"   Sentiment: {u['sentiment']:.2f}")
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    plot_sentiment_analysis(transcripts)
    print("\nSentiment analysis plot saved as sentiment_analysis_time.png")

if __name__ == "__main__":
    nltk.download('vader_lexicon')
    main()
