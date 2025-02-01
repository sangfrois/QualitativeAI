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
                {"start": 30.11, "end": 57.99, "text": " The ceremony with Gabi, the second ceremony, it was very different from the previous that we did in a way that the medicine felt, I felt the whole experience generally a little bit less intense, more like waves of...", "words": [{"word": "The", "start": 30.11, "end": 30.211, "score": 0.838, "speaker": "SPEAKER_00"}, {"word": "ceremony", "start": 30.251, "end": 30.931, "score": 0.87, "speaker": "SPEAKER_00"}, {"word": "ceremony,", "start": 30.971, "end": 31.651, "score": 0.967, "speaker": "SPEAKER_00"}, {"word": "the", "start": 31.711, "end": 31.831, "score": 0.99, "speaker": "SPEAKER_00"}, {"word": "second", "start": 31.911, "end": 32.291, "score": 0.988, "speaker": "SPEAKER_00"}, {"word": "ceremony,", "start": 32.351, "end": 33.152, "score": 0.989, "speaker": "SPEAKER_00"}, {"word": "it", "start": 33.852, "end": 33.952, "score": 0.991, "speaker": "SPEAKER_00"}, {"word": "was", "start": 34.012, "end": 34.172, "score": 0.99, "speaker": "SPEAKER_00"}, {"word": "very", "start": 34.252, "end": 34.552, "score": 0.989, "speaker": "SPEAKER_00"}, {"word": "different", "start": 34.612, "end": 35.252, "score": 0.989, "speaker": "SPEAKER_00"}, {"word": "from", "start": 35.352, "end": 35.552, "score": 0.99, "speaker": "SPEAKER_00"}, {"word": "the", "start": 35.632, "end": 35.732, "score": 0.99, "speaker": "SPEAKER_00"}, {"word": "previous", "start": 35.812, "end": 36.452, "score": 0.988, "speaker": "SPEAKER_00"}, {"word": "that", "start": 36.592, "end": 36.732, "score": 0.99, "speaker": "SPEAKER_00"}, {"word": "we", "start": 36.792, "end": 36.932, "score": 0.99, "speaker": "SPEAKER_00"}, {"word": "did", "start": 36.992, "end": 37.212, "score": 0.99, "speaker": "SPEAKER_00"}, {"word": "in", "start": 37.292, "end": 37.412, "score": 0.991, "speaker": "SPEAKER_00"}, {"word": "a", "start": 37.472, "end": 37.502, "score": 0.992, "speaker": "SPEAKER_00"}, {"word": "way", "start": 37.562, "end": 37.852, "score": 0.991, "speaker": "SPEAKER_00"}, {"word": "that", "start": 38.192, "end": 38.352, "score": 0.993, "speaker": "SPEAKER_00"}, {"word": "the", "start": 38.412, "end": 38.512, "score": 0.993, "speaker": "SPEAKER_00"}, {"word": "medicine", "start": 38.572, "end": 39.253, "score": 0.987, "speaker": "SPEAKER_00"}, {"word": "felt,", "start": 39.293, "end": 39.793, "score": 0.979, "speaker": "SPEAKER_00"}, {"word": "I", "start": 40.493, "end": 40.523, "score": 0.0, "speaker": "SPEAKER_00"}, {"word": "felt", "start": 40.583, "end": 40.853, "score": 0.347, "speaker": "SPEAKER_00"}, {"word": "the", "start": 40.913, "end": 41.013, "score": 0.99, "speaker": "SPEAKER_00"}, {"word": "whole", "start": 41.073, "end": 41.433, "score": 0.989, "speaker": "SPEAKER_00"}, {"word": "experience", "start": 41.493, "end": 42.254, "score": 0.988, "speaker": "SPEAKER_00"}, {"word": "generally", "start": 42.314, "end": 42.974, "score": 0.986, "speaker": "SPEAKER_00"}, {"word": "a", "start": 43.004, "end": 43.034, "score": 0.994, "speaker": "SPEAKER_00"}, {"word": "little", "start": 43.094, "end": 43.434, "score": 0.99, "speaker": "SPEAKER_00"}, {"word": "bit", "start": 43.494, "end": 43.694, "score": 0.991, "speaker": "SPEAKER_00"}, {"word": "less", "start": 43.754, "end": 44.054, "score": 0.991, "speaker": "SPEAKER_00"}, {"word": "intense,", "start": 44.114, "end": 44.835, "score": 0.989, "speaker": "SPEAKER_00"}, {"word": "more", "start": 45.195, "end": 45.455, "score": 0.992, "speaker": "SPEAKER_00"}, {"word": "like", "start": 45.515, "end": 45.735, "score": 0.993, "speaker": "SPEAKER_00"}, {"word": "waves", "start": 45.795, "end": 46.255, "score": 0.991, "speaker": "SPEAKER_00"}, {"word": "of...", "start": 46.315, "end": 46.795, "score": 0.991, "speaker": "SPEAKER_00"}]}
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
