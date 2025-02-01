import unittest
import subprocess
import os
import json
import tempfile

class TestSentimentScoring(unittest.TestCase):

    def test_sentiment_scoring_with_example_json(self):
        example_json_data = {
            "segments": [
                {"start": 4.275, "end": 4.735, "text": " Okay.", "words": [{"word": "Okay.", "start": 4.275, "end": 4.735, "score": 0.781, "speaker": "SPEAKER_01"}], "speaker": "SPEAKER_01"},
                {"start": 4.755, "end": 7.137, "text": "Hi, Kesemdir.", "words": [{"word": "Hi,", "start": 4.755, "end": 6.156, "score": 0.978, "speaker": "SPEAKER_01"}, {"word": "Kesemdir.", "start": 6.196, "end": 7.137, "score": 0.711, "speaker": "SPEAKER_01"}], "speaker": "SPEAKER_01"},
                {"start": 7.157, "end": 7.197, "text": "Hi.", "words": [{"word": "Hi.", "start": 7.157, "end": 7.197, "score": 0.003, "speaker": "SPEAKER_01"}], "speaker": "SPEAKER_01"},
                {"start": 7.217, "end": 12.18, "text": "I want to ask you about the second ceremony with Gabi.", "words": [{"word": "I", "start": 7.217, "end": 7.237, "score": 0.0, "speaker": "SPEAKER_01"}, {"word": "want", "start": 7.537, "end": 7.657, "score": 0.205, "speaker": "SPEAKER_01"}, {"word": "to", "start": 7.677, "end": 7.877, "score": 0.56, "speaker": "SPEAKER_01"}, {"word": "ask", "start": 8.258, "end": 8.398, "score": 0.385, "speaker": "SPEAKER_01"}, {"word": "you", "start": 8.418, "end": 8.938, "score": 0.587, "speaker": "SPEAKER_01"}, {"word": "about", "start": 9.699, "end": 10.019, "score": 0.917, "speaker": "SPEAKER_01"}, {"word": "the", "start": 10.079, "end": 10.219, "score": 0.813, "speaker": "SPEAKER_01"}, {"word": "second", "start": 10.339, "end": 10.719, "score": 0.943, "speaker": "SPEAKER_01"}, {"word": "ceremony", "start": 10.779, "end": 11.36, "score": 0.902, "speaker": "SPEAKER_01"}, {"word": "with", "start": 11.4, "end": 11.56, "score": 0.468, "speaker": "SPEAKER_01"}, {"word": "Gabi.", "start": 11.66, "end": 12.18, "score": 0.886, "speaker": "SPEAKER_01"}], "speaker": "SPEAKER_01"},
                {"start": 12.761, "end": 21.846, "text": "If you can freely describe your experience, like longitudinal, how was it evolving for you?", "words": [{"word": "If", "start": 12.761, "end": 12.841, "score": 0.996, "speaker": "SPEAKER_01"}, {"word": "you", "start": 12.901, "end": 13.041, "score": 0.999, "speaker": "SPEAKER_01"}, {"word": "can", "start": 13.101, "end": 13.461, "score": 0.82, "speaker": "SPEAKER_01"}, {"word": "freely", "start": 13.801, "end": 14.242, "score": 0.836, "speaker": "SPEAKER_01"}, {"word": "describe", "start": 14.282, "end": 15.102, "score": 0.851, "speaker": "SPEAKER_01"}, {"word": "your", "start": 15.162, "end": 15.382, "score": 0.886, "speaker": "SPEAKER_01"}, {"word": "experience,", "start": 15.442, "end": 16.343, "score": 0.813, "speaker": "SPEAKER_01"}, {"word": "like", "start": 16.383, "end": 16.603, "score": 0.67, "speaker": "SPEAKER_01"}, {"word": "longitudinal,", "start": 16.723, "end": 17.944, "score": 0.842, "speaker": "SPEAKER_01"}, {"word": "how", "start": 19.305, "end": 19.505, "score": 0.994, "speaker": "SPEAKER_01"}, {"word": "was", "start": 19.565, "end": 19.705, "score": 0.919, "speaker": "SPEAKER_01"}, {"word": "it", "start": 19.765, "end": 19.845, "score": 0.83, "speaker": "SPEAKER_01"}, {"word": "evolving", "start": 20.105, "end": 20.866, "score": 0.911, "speaker": "SPEAKER_01"}, {"word": "for", "start": 21.386, "end": 21.546, "score": 0.998, "speaker": "SPEAKER_01"}, {"word": "you?", "start": 21.606, "end": 21.846, "score": 0.894, "speaker": "SPEAKER_01"}], "speaker": "SPEAKER_01"},
                {"start": 21.867, "end": 23.568, "text": "Okay.", "words": [{"word": "Okay.", "start": 21.867, "end": 23.568, "score": 0.856, "speaker": "SPEAKER_01"}], "speaker": "SPEAKER_01"},
                {"start": 23.588, "end": 24.628, "text": "So...", "words": [{"word": "So...", "start": 23.588, "end": 24.628, "score": 0.494, "speaker": "SPEAKER_00"}], "speaker": "SPEAKER_00"},
                {"start": 30.11, "end": 57.99, "text": " The ceremony with Gabi, the second ceremony, it was very different from the previous that we did in a way that the medicine felt, I felt the whole experience generally a little bit less intense, more like waves of...", "words": [{"word": "The", "start": 30.11, "end": 30.211, "score": 0.838, "speaker": "SPEAKER_00"}, {"word": "ceremony", "start": 30.251, "end": 30.931, "score": 0.87, "speaker": "SPE"}
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_json_file_path = os.path.join(temp_dir, "test_transcript.json")
            with open(temp_json_file_path, 'w') as f:
                json.dump(example_json_data, f)

            command = [
                "python",
                os.path.join("analysis", "sentiment_scoring.py"),
                temp_dir # Pass the directory, script will glob for json files
            ]
            process = subprocess.run(command, capture_output=True, text=True, check=False)

            self.assertEqual(process.returncode, 0, f"Script failed with error: {process.stderr}")
            self.assertIn("Sentiment analysis plot saved as sentiment_analysis_time.png", process.stdout)

if __name__ == '__main__':
    unittest.main()
