#!/usr/bin/env python3
"""
Module for processing and chunking transcript files.

Supports both plain text (.txt) and JSON transcript formats.
Splits transcripts by speaker and timestamp, then merges small chunks
to create segments of approximately 1000 tokens each while respecting
sentence boundaries where possible.

Example usage:
    python chunk_transcripts.py --input-dir /path/to/transcripts --output-dir /path/to/output
"""

import os
import json
import re
import argparse
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Initialize tokenizer for GPT-4
tokenizer = tiktoken.get_encoding("cl100k_base")

@dataclass
class TranscriptChunk:
    """Represents a chunk of transcript with metadata."""
    text: str
    speaker: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    is_primary: bool = False
    source_file: Optional[str] = None

    @property
    def token_count(self) -> int:
        """Return the number of tokens in the chunk."""
        return len(tokenizer.encode(self.text))

def load_transcript(file_path: str) -> List[Dict]:
    """Load transcript from file, supporting both .txt and .json formats."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {file_path}")
    
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # Handle different JSON structures
                if isinstance(data, list):
                    return data
                elif 'segments' in data:  # Common in Whisper-style transcripts
                    return data['segments']
                else:
                    raise ValueError("Unsupported JSON format: missing 'segments' key")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON file: {file_path}")
    else:  # Assume plain text
        with open(file_path, 'r', encoding='utf-8') as f:
            return [{'text': f.read(), 'speaker': 'speaker_0'}]

def parse_speaker_from_text(text: str) -> Tuple[str, str]:
    """Extract speaker and content from a line of text."""
    # Common patterns for speaker identification
    patterns = [
        r'^(\w+):\s*(.*)',  # Speaker: Text
        r'^\[(.*?)\]\s*(.*)',  # [Speaker] Text
        r'^(\w+)\s*-\s*(.*)'  # Speaker - Text
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text.strip())
        if match:
            return match.group(1).strip(), match.group(2).strip()
    
    return 'unknown', text.strip()

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences while keeping punctuation."""
    # Use regex to split on punctuation followed by whitespace, retaining punctuation
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_pattern.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text: str, max_tokens: int = 1000) -> List[Dict]:
    """Split text into chunks of approximately max_tokens tokens using sentence boundaries."""
    sentences = _split_sentences(text)
    chunks: List[Dict[str, Union[str, int]]] = []
    current_sentences: List[str] = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
        sentence_token_count = len(sentence_tokens)

        # Handle sentences that exceed the chunk size by themselves
        if sentence_token_count > max_tokens:
            if current_sentences:
                chunk_text_value = ' '.join(current_sentences).strip()
                if chunk_text_value:
                    chunks.append({
                        'text': chunk_text_value,
                        'speaker': 'chunked_text',
                        'tokens': current_token_count
                    })
                current_sentences = []
                current_token_count = 0

            # Break the long sentence into sub-chunks by tokens
            for i in range(0, sentence_token_count, max_tokens):
                sub_tokens = sentence_tokens[i:i + max_tokens]
                sub_text = tokenizer.decode(sub_tokens).strip()
                if sub_text:
                    chunks.append({
                        'text': sub_text,
                        'speaker': 'chunked_text',
                        'tokens': len(sub_tokens)
                    })
            continue

        # If adding the sentence would exceed the limit, flush current chunk
        if current_sentences and (current_token_count + sentence_token_count) > max_tokens:
            chunk_text_value = ' '.join(current_sentences).strip()
            if chunk_text_value:
                chunks.append({
                    'text': chunk_text_value,
                    'speaker': 'chunked_text',
                    'tokens': current_token_count
                })
            current_sentences = [sentence]
            current_token_count = sentence_token_count
        else:
            current_sentences.append(sentence)
            current_token_count += sentence_token_count

    # Flush remaining sentences
    if current_sentences:
        chunk_text_value = ' '.join(current_sentences).strip()
        if chunk_text_value:
            chunks.append({
                'text': chunk_text_value,
                'speaker': 'chunked_text',
                'tokens': current_token_count
            })

    # Fallback if no sentences were captured (e.g., no punctuation)
    if not chunks and text.strip():
        tokens = tokenizer.encode(text)
        tokens = tokens[:max_tokens]
        fallback_text = tokenizer.decode(tokens).strip()
        if fallback_text:
            chunks.append({
                'text': fallback_text,
                'speaker': 'chunked_text',
                'tokens': len(tokens)
            })

    return chunks

def process_transcript(file_path: str, primary_speaker: Optional[str] = None, chunk_size: int = 1000) -> List[TranscriptChunk]:
    """Process a single transcript file into chunks."""
    try:
        segments = load_transcript(file_path)
        chunks = []
        
        for segment in segments:
            # Handle different transcript formats
            text = segment.get('text', '').strip()
            if not text:
                continue
                
            speaker = segment.get('speaker', 'unknown')
            start_time = segment.get('start', None)
            end_time = segment.get('end', None)
            
            # If it's a plain text file with speaker in the text
            if speaker == 'speaker_0' and ':' in text:
                speaker, text = parse_speaker_from_text(text)
            
            # Check if this is the primary speaker
            is_primary = (speaker.lower() == primary_speaker.lower()) if primary_speaker else False
            
            # If the text is too long, split it into chunks
            text_chunks = chunk_text(text, max_tokens=chunk_size)
            
            for i, text_chunk in enumerate(text_chunks):
                chunk_start_time = start_time
                chunk_end_time = end_time
                
                # For multiple chunks from same segment, distribute timing if available
                if len(text_chunks) > 1 and start_time is not None and end_time is not None:
                    chunk_duration = (end_time - start_time) / len(text_chunks)
                    chunk_start_time = start_time + (i * chunk_duration)
                    chunk_end_time = chunk_start_time + chunk_duration
                
                chunks.append(TranscriptChunk(
                    text=text_chunk['text'],
                    speaker=speaker if i == 0 else f"{speaker}_chunk_{i+1}",  # Differentiate chunks
                    start_time=chunk_start_time,
                    end_time=chunk_end_time,
                    is_primary=is_primary,
                    source_file=os.path.basename(file_path)
                ))
        
        return chunks
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

def merge_small_chunks(
    chunks: List[TranscriptChunk],
    min_tokens: int = 100,
    max_tokens: int = 1000
) -> List[TranscriptChunk]:
    """Merge small chunks with the same speaker to stay under max_tokens."""
    if not chunks:
        return []
    
    merged = []
    current_chunk = chunks[0]
    
    for chunk in chunks[1:]:
        # If same speaker and combined is still under max tokens, merge
        if (chunk.speaker == current_chunk.speaker and 
            (current_chunk.token_count + chunk.token_count) < max_tokens):
            current_chunk.text += "\n" + chunk.text
            current_chunk.end_time = chunk.end_time
        else:
            merged.append(current_chunk)
            current_chunk = chunk
    
    # Add the last chunk
    merged.append(current_chunk)
    
    return merged

def process_directory(input_dir: str, output_dir: str, primary_speaker: Optional[str] = None, chunk_size: int = 1000) -> None:
    """Process all transcript files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    
    # Process all .txt and .json files in the input directory
    for ext in ['*.txt', '*.json']:
        for file_path in input_path.glob(ext):
            print(f"Processing {file_path.name}...")
            chunks = process_transcript(str(file_path), primary_speaker, chunk_size)
            all_chunks.extend(chunks)
    
    # Merge small chunks
    merged_chunks = merge_small_chunks(all_chunks, max_tokens=chunk_size)
    
    # Prepare output data
    output_data = []
    for i, chunk in enumerate(merged_chunks):
        output_data.append({
            'id': f"chunk_{i:04d}",
            'text': chunk.text,
            'speaker': chunk.speaker,
            'is_primary': chunk.is_primary,
            'start_time': chunk.start_time,
            'end_time': chunk.end_time,
            'source_file': chunk.source_file,
            'token_count': chunk.token_count
        })
    
    # Save to output file
    output_file = output_path / 'chunked_transcripts.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(merged_chunks)} chunks. Output saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Process and chunk transcript files.')
    parser.add_argument('--input-dir', type=str, default='/data/transcripts',
                      help='Directory containing transcript files')
    parser.add_argument('--output-dir', type=str, default='/data/processed',
                      help='Directory to save processed chunks')
    parser.add_argument('--primary-speaker', type=str, default=None,
                      help='Tag this speaker as the primary source of answers')
    parser.add_argument('--chunk-size', type=int, default=1000,
                      help='Maximum tokens per chunk')
    
    args = parser.parse_args()
    
    # Process the directory
    process_directory(args.input_dir, args.output_dir, args.primary_speaker, args.chunk_size)

if __name__ == "__main__":
    main()
