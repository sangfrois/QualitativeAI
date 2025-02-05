import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np

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

def time_to_seconds(time_str):
    if not time_str:
        return None
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 2:  # MM:SS
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:  # H:MM:SS
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    return None

def plot_sentiment_analysis(transcripts, smoothing_window=20):
    plt.figure(figsize=(20, 15))

    all_speakers = set()
    for utterances in transcripts.values():
        all_speakers.update(utterance['speaker'] for utterance in utterances)

    color_map = {'Patient': 'red'}
    other_speakers = sorted([speaker for speaker in all_speakers if speaker != 'Patient'])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(other_speakers)+1))
    color_map.update(dict(zip(other_speakers, colors)))

    interview_duration_seconds = 90 * 60  # 1h30 in seconds

    for i, (file, utterances) in enumerate(transcripts.items(), 1):
        ax = plt.subplot(3, 2, i) # Get the axes object

        df = pd.DataFrame(utterances)
        df['sequence'] = pd.to_numeric(df['sequence'], errors='coerce')
        df = df.dropna(subset=['sequence', 'sentiment'])
        df = df.sort_values('sequence')

        time_breaks = []
        timestamp_ticks = {}

        for idx, row in df.iterrows():
            if 'timestamp' in row['metadata'] and row['metadata']['timestamp']:
                ts = row['metadata']['timestamp']
                start_time_str = ts.get('start')
                end_time_str = ts.get('end')

                start_time_sec = time_to_seconds(start_time_str)

                if start_time_sec is not None:
                    normalized_time = start_time_sec / interview_duration_seconds if interview_duration_seconds > 0 else 0
                    alpha_value = 0.5 - (normalized_time * 0.4)

                    if start_time_str:
                        start_seq = row['sequence']
                        timestamp_ticks[start_seq] = start_time_str

                        end_seq_shade = row['sequence'] + 1
                        if end_time_str:
                            end_seq_shade = row['sequence'] + 2

                        time_breaks.append({'start_seq': start_seq, 'end_seq': end_seq_shade,
                                            'start_time': start_time_sec, 'end_time': time_to_seconds(end_time_str),
                                            'start_time_str': start_time_str, 'end_time_str': end_time_str,
                                            'alpha': alpha_value})


        for break_info in time_breaks:
             ax.axvspan(break_info['start_seq'], break_info['end_seq'], color='gray', alpha=break_info['alpha'], label='Time Break' if break_info == time_breaks[0] else None)

        for speaker in df['speaker'].unique():
            speaker_data = df[df['speaker'] == speaker]
            speaker_data['smoothed_sentiment'] = speaker_data['sentiment'].rolling(window=smoothing_window, min_periods=1, center=True).mean()

            ax.plot(speaker_data['sequence'], speaker_data['smoothed_sentiment'], linestyle='-', label=speaker, color=color_map[speaker])
            ax.plot(speaker_data['sequence'], speaker_data['sentiment'], marker='o', linestyle='', color=color_map[speaker], alpha=0.5, markersize=3)


        if timestamp_ticks:
            tick_positions = list(timestamp_ticks.keys())
            tick_labels = [timestamp_ticks[pos] for pos in tick_positions]
            ax.set_xticks(tick_positions) # Use set_xticks and set_xticklabels on the axes object
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize='small') # Reduced fontsize

            ax.set_xlabel("Sequence of Utterances (with Timestamps)")
        else:
            ax.set_xlabel("Sequence of Utterances")

        ax.set_title(f"Sentiment Analysis - {file}")
        ax.set_ylabel("Sentiment Score")

        # Minimalist spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    # Get handles and labels from the last subplot (or any subplot, they should be the same)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    # Overall legend in the bottom right empty space, no box
    plt.legend(unique_handles, unique_labels, title="Legend", loc='lower right', frameon=False, bbox_to_anchor=(1.0, 0.0)) # Moved legend


    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust tight_layout to make space for the legend, reduce right margin
    plt.savefig("sentiment_analysis_time_minimal.png", dpi=300) # Changed filename
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

    plot_sentiment_analysis(transcripts, smoothing_window=7)
    print("\nMinimalist sentiment analysis plot saved as sentiment_analysis_time_minimal.png")


if __name__ == "__main__":
    nltk.download('vader_lexicon')
    main()