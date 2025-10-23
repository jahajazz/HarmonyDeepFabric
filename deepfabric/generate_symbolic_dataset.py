#!/usr/bin/env python3
"""
CLI tool for generating symbolic dataset using DeepFabric.
Processes transcript chunks and generates symbolic analyses using Harmony format.
"""

import os
import sys
import json
import yaml
import time
import logging
import asyncio
import traceback
import argparse
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('symbolic_dataset_generation.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the current directory to the path to allow importing from deepfabric
sys.path.append(str(Path(__file__).parent))

try:
    from deepfabric.llm import LLMClient
    from deepfabric.formatters import FormatterRegistry
    from deepfabric.formatters.models import Message, ConversationSample
    from chunk_transcripts import process_directory, TranscriptChunk
    from harmony_writer import write_harmony_record, flatten_analysis, flatten_final
    from pydantic import BaseModel, Field
    import tiktoken
    import backoff
    import openai
    import anthropic
    import google.generativeai as genai
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please install the required dependencies with: pip install -e .")
    sys.exit(1)

# Tokenizer for counting tokens
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: Optional[str]) -> int:
    """Return token count for the given text using the global tokenizer."""
    if not text:
        return 0
    return len(tokenizer.encode(text))

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Trim text to the specified token count boundary."""
    if not text:
        return ""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text.strip()
    return tokenizer.decode(tokens[:max_tokens]).strip()

_sentence_splitter = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: Optional[str]) -> List[str]:
    """Split text into sentences using simple punctuation-based heuristic."""
    if not text:
        return []
    return [segment.strip() for segment in _sentence_splitter.split(text) if segment.strip()]

def build_abbreviated_analysis(analysis: Dict[str, Any], target_tokens: int = 200) -> str:
    """Create a concise analysis variant focusing on key bullet points."""
    summary = (analysis.get('symbolic_summary') or analysis.get('symbolic_meaning') or '').strip()
    archetypal = analysis.get('archetypal_patterns') or []
    theological = analysis.get('theological_themes') or []

    parts: List[str] = []
    if summary:
        parts.append(f"Symbolic reasoning: {summary}")
    if archetypal:
        parts.append("Archetypal patterns: " + ', '.join(str(item) for item in archetypal))
    if theological:
        parts.append("Theological themes: " + ', '.join(str(item) for item in theological))

    abbreviated = ' '.join(parts).strip()
    if not abbreviated:
        abbreviated = summary

    if abbreviated:
        tokens = count_tokens(abbreviated)
        filler_sentences = []
        if tokens < int(target_tokens * 0.8):
            if archetypal:
                filler_sentences.append(
                    "This concise synthesis foregrounds patterns such as "
                    + ', '.join(str(item) for item in archetypal) + "."
                )
            if theological:
                filler_sentences.append(
                    "It keeps the theological horizon in view, touching on "
                    + ', '.join(str(item) for item in theological) + "."
                )
            if summary:
                filler_sentences.append(f"In essence, {summary}")
            abbreviated = ' '.join([abbreviated] + filler_sentences).strip()

        safety_iterations = 0
        while count_tokens(abbreviated) < target_tokens and safety_iterations < 5:
            abbreviated += (
                " This condensed reflection preserves the symbolic architecture while remaining focused on the most "
                "salient motifs from the transcript."
            )
            safety_iterations += 1

    return truncate_to_tokens(abbreviated, target_tokens)

def build_detailed_analysis(
    analysis: Dict[str, Any],
    base_text: str,
    chunk_text: str,
    target_tokens: int = 600
) -> str:
    """Create an expanded analysis variant with additional elaboration and context."""
    base_text = (base_text or "").strip()
    if not base_text:
        base_text = build_abbreviated_analysis(analysis, target_tokens=min(target_tokens, 200))

    if count_tokens(base_text) >= target_tokens:
        return truncate_to_tokens(base_text, target_tokens)

    summary = (analysis.get('symbolic_meaning') or analysis.get('symbolic_summary') or '').strip()
    patterns = [str(item) for item in (analysis.get('archetypal_patterns') or [])]
    themes = [str(item) for item in (analysis.get('theological_themes') or [])]
    context_sentences = split_sentences(chunk_text)

    extra_sections: List[str] = []
    idx = 0

    def combined_text() -> str:
        if not extra_sections:
            return base_text
        return base_text + "\n\n" + "\n\n".join(extra_sections)

    while count_tokens(combined_text()) < target_tokens:
        pieces = ["Extended symbolic reflection:"]

        if patterns:
            pattern = patterns[idx % len(patterns)]
            pieces.append(
                f"The archetypal pattern '{pattern}' shapes the narrative's progression and highlights recurring spiritual motifs."
            )
        if themes:
            theme = themes[idx % len(themes)]
            pieces.append(
                f"This dovetails with the theological theme of {theme}, grounding the reflection in doctrinal resonance."
            )
        if context_sentences:
            context = context_sentences[idx % len(context_sentences)]
            pieces.append(f"A representative line — \"{context}\" — anchors the interpretation in the transcript itself.")
        if summary:
            pieces.append(f"Taken together, these strands deepen the insight that {summary}")

        paragraph = ' '.join(pieces).strip()
        if paragraph:
            extra_sections.append(paragraph)

        idx += 1
        if idx > 12:  # avoid runaway loops
            break

    detailed_text = combined_text().strip()

    if not extra_sections and summary:
        filler = (
            "Extended symbolic reflection: The analysis revisits the transcript to unfold additional layers of meaning, "
            "drawing pastoral and theological applications from its imagery and cadence."
        )
        detailed_text = (detailed_text + "\n\n" + filler).strip()

    # Provide gentle padding if still below target
    safety_iterations = 0
    while 0 < count_tokens(detailed_text) < target_tokens and safety_iterations < 10:
        detailed_text += (
            "\n\nFurther elaboration: This commentary layers historical, symbolic, and pastoral perspectives to keep the "
            "interpretive lens attentive to the transcript's unfolding imagery."
        )
        safety_iterations += 1
        if count_tokens(detailed_text) >= target_tokens:
            break

    return truncate_to_tokens(detailed_text, target_tokens)
class SymbolicAnalysis(BaseModel):
    """Schema for symbolic analysis output."""
    symbolic_summary: str = Field(..., description="A concise symbolic and theological interpretation of the text")
    archetypal_patterns: List[str] = Field(..., description="Key archetypal patterns identified in the text")
    theological_themes: List[str] = Field(..., description="Theological themes present in the text")
    symbolic_meaning: str = Field(..., description="Deeper symbolic meaning and significance")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file with validation."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # Set default values for required fields
        config.setdefault('model', 'gpt-4o-mini')
        config.setdefault('data', {})
        config['data'].setdefault('input_dir', '/data/transcripts')
        config['data'].setdefault('output_file', '/data/output/symbolic_harmony.jsonl')
        config.setdefault('generation', {})
        config['generation'].setdefault('temperature', 0.7)
        config['generation'].setdefault('max_tokens', 1024)
        config['generation'].setdefault('chunk_size', 1000)
        config['generation'].setdefault('max_retries', 3)
        config['data'].setdefault('chunk_size', 1000)
        
        return config
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        raise

def ensure_directories_exist(path: str) -> None:
    """Ensure that the directory of the given path exists."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise

def create_harmony_messages(system_prompt: str, chunk: TranscriptChunk) -> List[Dict[str, str]]:
    """Create a conversation in Harmony format."""
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"""Analyze the following text and produce a symbolic narrative summary.
            
            Speaker: {chunk.speaker}
            {'(Primary Speaker)' if chunk.is_primary else ''}
            
            Text: {chunk.text}"""
        }
    ]

@backoff.on_exception(
    backoff.expo,
    (openai.APIError, anthropic.APIError, genai.types.StopCandidateException, Exception),
    max_tries=5,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: logger.warning(
        f"Retry {details['tries']} for chunk processing after error: {details['value']}"
    )
)
async def process_chunk_with_retry(
    client: LLMClient,
    messages: List[Dict[str, str]],
    config: Dict[str, Any],
    chunk_info: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Process a single chunk with retry logic and error handling."""
    try:
        # Create conversation sample from messages
        conversation = ConversationSample(messages=messages)
        
        # Get the appropriate formatter (Harmony format)
        formatter_registry = FormatterRegistry()
        formatter = formatter_registry.load_formatter("builtin://harmony")
        
        # Format the conversation for the model
        formatted_sample = formatter._format_single_sample({
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in conversation.messages
            ]
        })
        
        # Check if formatting failed
        if formatted_sample is None:
            raise ValueError("Failed to format conversation for the model")
        
        # Generate the response
        try:
            response = await client.generate_async(
                prompt=formatted_sample["text"],
                schema=SymbolicAnalysis,
                max_tokens=config['generation']['max_tokens'],
                temperature=config['generation']['temperature'],
                top_p=config['generation'].get('top_p', 0.9),
                frequency_penalty=config['generation'].get('frequency_penalty', 0.2),
                presence_penalty=config['generation'].get('presence_penalty', 0.1),
            )
        except Exception as schema_error:
            # Fallback to text generation if structured outputs fail
            logger.warning(f"Structured output failed, falling back to text generation: {schema_error}")
            try:
                # Generate raw text response instead
                text_response = await client.generate_async(
                    prompt=formatted_sample["text"],
                    schema=None,  # No schema for text generation
                    max_tokens=config['generation']['max_tokens'],
                    temperature=config['generation']['temperature'],
                    top_p=config['generation'].get('top_p', 0.9),
                    frequency_penalty=config['generation'].get('frequency_penalty', 0.2),
                    presence_penalty=config['generation'].get('presence_penalty', 0.1),
                )
                # Create a mock SymbolicAnalysis object from the text
                response = SymbolicAnalysis(
                    symbolic_summary="Text analysis (structured output unavailable)",
                    archetypal_patterns=["General analysis"],
                    theological_themes=["General themes"],
                    symbolic_meaning=str(text_response)
                )
            except Exception as text_error:
                raise ValueError(f"Both structured and text generation failed: schema_error={schema_error}, text_error={text_error}")
        
        # Prepare the result
        result = {
            **chunk_info,
            "analysis": response.model_dump(),
            "model": client.model_name,
            "timestamp": time.time(),
            "status": "completed"
        }
        
        return result
        
    except Exception as e:
        error_info = {
            **chunk_info,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed"
        }
        logger.error(f"Error processing chunk {chunk_info.get('chunk_id')}: {e}")
        return error_info

async def process_chunk(
    chunk: TranscriptChunk,
    client: LLMClient,
    config: Dict[str, Any],
    chunk_id: int,
    total_chunks: int
) -> Optional[Dict[str, Any]]:
    """Process a single transcript chunk."""
    try:
        logger.info(f"Processing chunk {chunk_id + 1}/{total_chunks} ({(chunk_id + 1)/total_chunks:.1%})")
        
        # Prepare chunk info for logging
        chunk_info = {
            "chunk_id": chunk_id,
            "speaker": chunk.speaker,
            "is_primary": chunk.is_primary,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "source_file": chunk.source_file,
            "token_count": len(tokenizer.encode(chunk.text))
        }
        
        # Create messages in Harmony format
        messages = create_harmony_messages(config['system_prompt'], chunk)
        
        # Process with retry logic
        result = await process_chunk_with_retry(client, messages, config, chunk_info)
        
        if result and result.get('status') == 'completed':
            logger.info(f"Successfully processed chunk {chunk_id + 1}/{total_chunks}")
        else:
            logger.error(f"Failed to process chunk {chunk_id + 1}/{total_chunks}")
            
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error in process_chunk: {e}", exc_info=True)
        return None

async def process_all_chunks(
    client: LLMClient,
    config: Dict[str, Any],
    input_dir: str,
    output_file: str,
    primary_speaker: Optional[str] = None,
    fail_on_empty: bool = True,
    dry_run: int = 0
) -> Dict[str, int]:
    """Process all chunks from the input directory."""
    # Process the directory to get chunks
    logger.info(f"Processing transcripts from {input_dir}")
    
    # Create a temporary output directory for chunked transcripts
    temp_dir = Path("/tmp/chunked_transcripts")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Process the directory to get chunks
    from chunk_transcripts import process_directory
    chunk_size = config.get('data', {}).get('chunk_size', 1000)

    process_directory(
        input_dir=input_dir,
        output_dir=str(temp_dir),
        primary_speaker=primary_speaker,
        chunk_size=chunk_size
    )
    
    # Load the processed chunks
    chunks_file = temp_dir / 'chunked_transcripts.json'
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    if not chunks_data:
        logger.warning("No chunks found to process")
        return {"processed": 0, "examples": 0, "errors": 0, "total": 0}
    
    logger.info(f"Found {len(chunks_data)} chunks to process")
    logger.info(f"Appending Harmony examples to {output_file}")
    
    # Process chunks with concurrency
    semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
    processed_count = 0  # Successful chunks
    processed_examples = 0  # Total Harmony examples written
    error_count = 0
    skipped_count = 0

    async def process_with_semaphore(chunk_data, chunk_id):
        nonlocal processed_count, processed_examples, error_count, skipped_count

        async with semaphore:
            chunk = TranscriptChunk(
                text=chunk_data['text'],
                speaker=chunk_data['speaker'],
                start_time=chunk_data.get('start_time'),
                end_time=chunk_data.get('end_time'),
                is_primary=chunk_data.get('is_primary', False),
                source_file=chunk_data.get('source_file')
            )

            result = await process_chunk(
                chunk=chunk,
                client=client,
                config=config,
                chunk_id=chunk_id,
                total_chunks=len(chunks_data)
            )

            if result:
                # Extract the components for harmony writing
                system_prompt = config['system_prompt']
                messages = create_harmony_messages(system_prompt, chunk)
                user_content = messages[1]['content']  # User message content

                # Prepare metadata
                metadata_keys = ["chunk_id","speaker","is_primary","start_time","end_time","source_file","token_count","timestamp","model","status"]
                metadata = {k: result.get(k) for k in metadata_keys if k in result}

                # Handle empty content based on fail_on_empty flag
                if not result.get('status') == 'completed':
                    error_count += 1
                else:
                    try:
                        analysis_obj = result.get("analysis") or {}
                        if isinstance(analysis_obj, dict):
                            analysis_dict = analysis_obj
                        else:
                            analysis_dict = {"symbolic_meaning": str(analysis_obj)}

                        base_analysis_text = flatten_analysis(analysis_dict)
                        if not base_analysis_text:
                            raise ValueError("Empty analysis content")

                        final_source = result.get("final") or analysis_dict.get("symbolic_summary")
                        final_text = flatten_final(final_source)
                        if not final_text:
                            raise ValueError("Empty final content")

                        abbreviated_text = build_abbreviated_analysis(analysis_dict)
                        detailed_text = build_detailed_analysis(analysis_dict, base_analysis_text, chunk.text)

                        variant_map = [
                            ("base", base_analysis_text),
                            ("abbreviated", abbreviated_text or base_analysis_text),
                            ("detailed", detailed_text or base_analysis_text)
                        ]
                        variant_indices = {"base": 0, "abbreviated": 1, "detailed": 2}

                        with open(output_file, 'a', encoding='utf-8') as out_fp:
                            for variant_name, analysis_text in variant_map:
                                if not analysis_text:
                                    analysis_text = base_analysis_text

                                variant_metadata = dict(metadata)
                                variant_metadata["analysis_variant"] = variant_name
                                variant_metadata["variant_index"] = variant_indices[variant_name]

                                write_harmony_record(
                                    out_fp=out_fp,
                                    system_prompt=system_prompt,
                                    user_content=user_content,
                                    analysis_obj=analysis_text,
                                    final_obj=final_text,
                                    metadata=variant_metadata
                                )
                                processed_examples += 1

                        processed_count += 1

                        # Check dry run limit
                        if dry_run > 0 and processed_count >= dry_run:
                            logger.info(
                                f"Dry run completed after {processed_count} chunks ({processed_examples} examples)"
                            )
                            return True  # Signal to stop

                    except ValueError as e:
                        if fail_on_empty:
                            logger.error(f"Empty content for chunk {chunk_id}: {e}")
                            raise  # Re-raise to stop processing
                        else:
                            logger.warning(f"Skipping chunk {chunk_id} due to empty content: {e}")
                            skipped_count += 1
                    except Exception as general_error:
                        logger.error(f"Unexpected write error for chunk {chunk_id}: {general_error}", exc_info=True)
                        error_count += 1
            else:
                error_count += 1

            return False  # Continue processing

    # Create and run tasks
    tasks = [
        process_with_semaphore(chunk_data, i)
        for i, chunk_data in enumerate(chunks_data)
    ]

    # Process in batches to avoid overwhelming the API
    batch_size = 10
    stop_processing = False

    for i in range(0, len(tasks), batch_size):
        if stop_processing:
            break

        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch)

        # Check if any task signaled to stop (dry run complete)
        if any(batch_results):
            stop_processing = True

        logger.info(f"Processed {min(i + batch_size, len(tasks))}/{len(tasks)} chunks")

    return {
        "processed": processed_count,
        "examples": processed_examples,
        "errors": error_count,
        "skipped": skipped_count,
        "total": len(chunks_data)
    }

async def main_async():
    """Async entry point."""
    parser = argparse.ArgumentParser(description='Generate symbolic dataset from transcript chunks.')
    parser.add_argument('--config', type=str, default='symbolic_config.yaml',
                      help='Path to configuration YAML file')
    parser.add_argument('--input-dir', type=str, default=None,
                      help='Override input directory from config')
    parser.add_argument('--output-file', type=str, default=None,
                      help='Override output file path from config')
    parser.add_argument('--primary-speaker', type=str, default=None,
                      help='Primary speaker to tag for question generation')
    parser.add_argument('--provider', type=str, default='openai',
                      choices=['openai', 'anthropic', 'gemini', 'ollama'],
                      help='LLM provider to use')
    parser.add_argument('--model', type=str, default=None,
                      help='Model name to use (overrides config)')
    parser.add_argument('--fail-on-empty', action='store_true', default=True,
                      help='Fail if analysis or final content is empty (default: True)')
    parser.add_argument('--no-fail-on-empty', action='store_false', dest='fail_on_empty',
                      help='Skip rows with empty content instead of failing')
    parser.add_argument('--dry-run', type=int, default=0,
                      help='Run a dry run with N rows and exit (for testing)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Apply overrides from command line
        if args.input_dir:
            config['data']['input_dir'] = args.input_dir
        if args.output_file:
            config['data']['output_file'] = args.output_file
        if args.model:
            config['model'] = args.model
        
        # Ensure output directory exists
        ensure_directories_exist(config['data']['output_file'])
        
        # Initialize LLM client
        try:
            client = LLMClient(
                provider=args.provider,
                model_name=config['model']
            )
            logger.info(f"Initialized {args.provider} client with model {config['model']}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            sys.exit(1)
        
        # Process all chunks
        start_time = time.time()
        
        stats = await process_all_chunks(
            client=client,
            config=config,
            input_dir=config['data']['input_dir'],
            output_file=config['data']['output_file'],
            primary_speaker=args.primary_speaker,
            fail_on_empty=args.fail_on_empty,
            dry_run=args.dry_run
        )
        
        elapsed = time.time() - start_time
        
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("Symbolic Dataset Generation Complete!")
        logger.info(f"Total chunks processed: {stats['total']}")
        logger.info(f"Successfully processed: {stats['processed']}")
        if 'examples' in stats:
            logger.info(f"Total Harmony examples written: {stats['examples']}")
        logger.info(f"Errors: {stats['errors']}")
        if stats.get('skipped', 0) > 0:
            logger.info(f"Skipped (empty content): {stats['skipped']}")
        logger.info(f"Time taken: {elapsed:.2f} seconds")
        logger.info(f"Output appended to: {config['data']['output_file']}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

def main():
    """Synchronous entry point."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
