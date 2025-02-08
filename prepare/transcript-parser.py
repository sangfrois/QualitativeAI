import re
from typing import List, Dict
import json
from docx import Document
import os

def read_word_document(file_path: str) -> List[str]:
    """Read text from a Microsoft Word document."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Word document not found at: {file_path}")
    
    doc = Document(file_path)
    paragraphs = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
    
    # Remove first entry if it looks like a file label
    if paragraphs and re.match(r'KET-\d{3}-\d{2}', paragraphs[0]):
        paragraphs.pop(0)
    
    return paragraphs

def extract_speaker(text: str):
    """Extract speaker label from text, even if it appears after a timestamp."""
    match = re.match(r'^(Patient|Interviewer(?:\s*\d*))\s*:\s*(.*)', text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    match = re.search(r'(Patient|Interviewer(?:\s*\d*))\s*:\s*(.*)', text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, text.strip()

def parse_timestamp(text: str) -> Dict:
    """Extract timestamps from text."""
    patterns = [
        r'\[(Silence|Inaudible|Distant conversation|Shuffling, silence)\s+(\d{1,2}:\d{2}(?::\d{2})?)(?:\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?))?\]',
        r'End of recording (\d{1,2}:\d{2}(?::\d{2})?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {
                "type": match.group(1).capitalize(),
                "start": match.group(2) if len(match.groups()) > 1 else None,
                "end": match.group(3) if len(match.groups()) > 2 else None
            }
    return None

def parse_transcript(paragraphs: List[str]) -> List[Dict]:
    """Parse a transcript and return structured utterances."""
    utterances = []
    sequence = 0
    current_speaker = None
    
    for paragraph in paragraphs:
        # Extract timestamp first to prevent interference with speaker extraction
        timestamp = parse_timestamp(paragraph)
        text_cleaned = re.sub(r'\[.*?\]', '', paragraph).strip()
        
        # Extract speaker properly (after cleaning timestamp from text)
        speaker, text = extract_speaker(text_cleaned)
        if speaker:
            current_speaker = speaker
        elif current_speaker:
            speaker = current_speaker  # Retain previous speaker if none found
        
        utterance = {
            "sequence": sequence,
            "speaker": speaker,
            "text": text.strip() if text else "",
            "metadata": {}
        }
        
        if timestamp:
            utterance["metadata"]["timestamp"] = timestamp
        
        if utterance['text'] or utterance['speaker']:
            utterances.append(utterance)
            sequence += 1

    return utterances

def process_word_transcript(word_file: str, output_file: str) -> List[Dict]:
    """Process a Word document transcript and save to JSON."""
    try:
        paragraphs = read_word_document(word_file)
        utterances = parse_transcript(paragraphs)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(utterances, f, indent=2, ensure_ascii=False)
        
        return utterances
    except Exception as e:
        print(f"Error processing transcript: {str(e)}")
        raise

def main():
    input_file = "KET-047/KET-047-04.docx"  # Replace with your file path
    output_file = "transcript_04_time.json"
    
    try:
        utterances = process_word_transcript(input_file, output_file)
        
        print(f"Processing complete. Total utterances: {len(utterances)}")
        for i, u in enumerate(utterances[:5]):
            print(f"\n{i}. Speaker: {u['speaker']}")
            print(f"   Text: {u['text']}")
            if 'metadata' in u and u['metadata']:
                print(f"   Metadata: {u['metadata']}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
