import ollama
import json
import os
import pandas as pd

# PROMPT

def load_transcript(file_path):
    """Loads a transcript from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in: {file_path}")
        return None

def analyze_sentiment_rolling_window(utterances):
    """
    Analyzes sentiment of a rolling window of utterances using Ollama and llama3.1.

    Args:
        utterances (list): A list of dictionaries, where each dictionary is an utterance
                           and contains a 'text' key.

    Returns:
        str: The sentiment analysis response from the llama3.1 model.
    """
    model_name = "llama3.1"
    prompt = f"RESPOND ONLY BY ONE VALUE TO THE FOLLOWING. BE CRITICAL AND SEVERE.:Evaluate the therapeutic alliance in the following 20 utterances from a therapy session transcript. Rate each component on a scale of 0-3, where: Bond (0-3): 0 = Negative/hostile relationship, 1 = Neutral/distant relationship, 2 = Warm but surface-level connection, 3 = Strong emotional bond with deep trust and understanding. Goals (0-3): 0 = Misaligned or unclear goals, 1 = Basic agreement but limited shared understanding, 2 = Clear shared goals with good alignment, 3 = Strong mutual investment in and understanding of therapeutic goals. Tasks (0-3): 0 = Resistance or disagreement about therapeutic approach, 1 = Passive acceptance of tasks, 2 = Active participation and agreement on methods, 3 = Strong collaboration and shared belief in therapeutic process. Calculate the total alliance score by summing the three components and dividing by 9, yielding a final score from 0 to 1. Multiply by 10 for a 0-10 scale. Key indicators: Client engagement and openness, Therapist empathy and validation, Mutual respect and collaboration, Clear communication about therapy process, Client buy-in to therapeutic methods, Evidence of trust and safety, Agreement on therapy direction, Signs of rupture or repair. VERY IMPORTANT: Provide ONLY the final 0-10 score. DO NOT SHARE YOUR PROCESS ONLY WRITE THE 0-10 SCORE!!!!; \n\n"
    prompt += "\n".join([f"- {u['text']}. \n\nPROVIDE ONLY THE 0-10 SCORE" for u in utterances])

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error during sentiment analysis: {e}"

def analyze_transcripts_in_directory(directory_path):
    """
    Analyzes all transcript JSON files in the specified directory using a rolling window approach.

    Args:
        directory_path (str): The path to the directory containing transcript JSON files.
    """
    json_files = [f for f in os.listdir(directory_path) if f.startswith('transcript_') and f.endswith('_time.json')]
    if not json_files:
        print(f"No transcript JSON files found in directory: {directory_path}")
        return
    df = pd.DataFrame()
    for transcript_file in json_files:
        file_path = os.path.join(directory_path, transcript_file)
        transcript_data = load_transcript(file_path)

        if transcript_data is not None:
            window_size = 30
            overlap = 5
            step_size = window_size - overlap

            print(f"Analyzing sentiment for: {transcript_file}")
            for i in range(0, len(transcript_data), step_size):
                start_index = i
                end_index = min(i + window_size, len(transcript_data))
                rolling_window = transcript_data[start_index:end_index]

                if len(rolling_window) > 0:
                    sentiment_result = analyze_sentiment_rolling_window(rolling_window)
                    window_utterance_indices = [utt['sequence'] for utt in rolling_window if 'sequence' in utt]
                    window_indices_str = f"Utterances {window_utterance_indices[0]}-{window_utterance_indices[-1]}" if window_utterance_indices else f"Window {start_index}-{end_index}"
                    print(f"\nSentiment Analysis Result for Rolling Window ({window_indices_str}):")
                    print(sentiment_result)
                    # save sentiment result to a csv with pandas
                    new_row = pd.DataFrame([{
                        'transcript_file': transcript_file,
                        'window_indices': window_indices_str,
                        'sentiment_result': sentiment_result
                    }])
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_csv('sentiment_analysis_results.csv', index=False)
                    
                    print("-" * 50)
                else:
                    print("No more utterances to analyze.")
                    break
        else:
            print(f"Failed to load transcript data from: {transcript_file}")
        print("=" * 60) # Separator between files

if __name__ == '__main__':
    directory_path = '.'  # Set the directory path where your transcript files are located (e.g., '.', or './transcripts')
    analyze_transcripts_in_directory(directory_path)
