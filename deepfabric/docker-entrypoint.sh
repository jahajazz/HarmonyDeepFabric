#!/bin/bash
# Startup script for Harmony converter container

set -e

echo "🚀 Starting Harmony Dataset Converter Container"
echo "=============================================="

# Check if input file exists
if [ ! -f "/data/input/symbolic_dataset.jsonl" ]; then
    echo "❌ Input file not found: /data/input/symbolic_dataset.jsonl"
    echo "   Please mount your data directory containing symbolic_dataset.jsonl"
    exit 1
fi

# Build the command with optional arguments
CMD="python convert_to_harmony.py"
CMD="$CMD --input-file /data/input/symbolic_dataset.jsonl"
CMD="$CMD --output-file /data/output/symbolic_harmony.jsonl"

# Add transcript directory if it exists
if [ -d "/data/transcripts" ]; then
    echo "📂 Found transcript directory: /data/transcripts"
    CMD="$CMD --transcript-dir /data/transcripts"
fi

# Add primary speaker if specified via environment variable
if [ ! -z "$PRIMARY_SPEAKER" ]; then
    echo "🎯 Using primary speaker: $PRIMARY_SPEAKER"
    CMD="$CMD --primary-speaker \"$PRIMARY_SPEAKER\""
fi

echo "🔄 Running converter with command:"
echo "   $CMD"
echo ""

# Execute the converter
eval $CMD

echo ""
echo "✅ Conversion completed successfully!"
echo "📋 Check /data/output/symbolic_harmony.jsonl for results"

# Keep container running if requested
if [ "$KEEP_RUNNING" = "true" ]; then
    echo "🔄 Keeping container running for inspection..."
    tail -f /dev/null
fi
