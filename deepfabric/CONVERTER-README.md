# Harmony Dataset Converter - Docker Container

This container provides a self-contained environment for converting symbolic datasets to Harmony conversation format.

## 🚀 Quick Start

### Using Docker Compose (Recommended)

1. **Build and run the container:**
   ```bash
   docker-compose -f docker-compose.converter.yml up --build
   ```

2. **Run with custom primary speaker:**
   ```bash
   PRIMARY_SPEAKER="Fr. Andrew Stephen Damick" docker-compose -f docker-compose.converter.yml up --build
   ```

3. **Keep container running after conversion:**
   ```bash
   KEEP_RUNNING=true docker-compose -f docker-compose.converter.yml up --build
   ```

### Using Docker directly

1. **Build the image:**
   ```bash
   docker build -f Dockerfile.converter -t harmony-converter .
   ```

2. **Run the container:**
   ```bash
   docker run --rm -v $(pwd)/data:/data harmony-converter
   ```

3. **With primary speaker:**
   ```bash
   docker run --rm -v $(pwd)/data:/data -e PRIMARY_SPEAKER="Speaker Name" harmony-converter
   ```

## 📁 Volume Mounting

The container expects the following directory structure:

```
/data/
├── input/
│   └── symbolic_dataset.jsonl    # Input file to convert
├── output/                       # Output directory (created automatically)
│   └── symbolic_harmony.jsonl    # Generated Harmony format
└── transcripts/                  # Optional transcript files
    ├── transcript1.txt
    ├── transcript2.txt
    └── ...
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PRIMARY_SPEAKER` | Primary speaker for analysis focus | `None` |
| `KEEP_RUNNING` | Keep container running after conversion | `false` |

### Command Line Arguments

The container supports all the same arguments as the standalone script:

```bash
# Basic usage
python convert_to_harmony.py --input-file /data/input/symbolic_dataset.jsonl --output-file /data/output/symbolic_harmony.jsonl

# With transcript directory
python convert_to_harmony.py --input-file /data/input/symbolic_dataset.jsonl --output-file /data/output/symbolic_harmony.jsonl --transcript-dir /data/transcripts

# With primary speaker
python convert_to_harmony.py --input-file /data/input/symbolic_dataset.jsonl --output-file /data/output/symbolic_harmony.jsonl --primary-speaker "Speaker Name"
```

## 🔍 Container Workflow

1. **Startup**: Container validates input file exists
2. **Transcript Loading**: Loads transcript files if transcript directory is provided
3. **Conversion**: Processes each line of input file
4. **Validation**: Validates Harmony structure of each entry
5. **Output**: Writes converted entries to output file
6. **Reporting**: Displays conversion statistics and summary

## 📊 Example Output

```
🔄 Converting symbolic dataset to Harmony format...
📂 Input: /data/input/symbolic_dataset.jsonl
📂 Output: /data/output/symbolic_harmony.jsonl
📂 Transcript directory: /data/transcripts
🎯 Primary speaker: Fr. Andrew Stephen Damick
📂 Using transcript directory: /data/transcripts
🎯 Primary speaker preference: Fr. Andrew Stephen Damick
📖 Loaded transcript: Angels and Demons II The Divine Council.txt
✅ Processed line 1 (valid)
...

📊 CONVERSION SUMMARY:
✅ Valid entries: 80
❌ Invalid entries: 0
📁 Output saved to: /data/output/symbolic_harmony.jsonl
📋 Total processed: 80

📜 TRANSCRIPT LOADING:
✅ Transcripts found: 80
❌ Transcripts missing: 0
📂 Transcript directory: /data/transcripts

🎯 PRIMARY SPEAKER PREFERENCE:
👑 Preferred speaker: Fr. Andrew Stephen Damick
📝 System messages updated with speaker preference

✅ Conversion complete!
📋 Check the output file: /data/output/symbolic_harmony.jsonl
```

## 🏗️ Container Architecture

- **Base Image**: `python:3.11-slim` (minimal Python environment)
- **Dependencies**: Only essential Python standard library (no external deps)
- **Volume Mount**: `/data` directory for input/output/transcripts
- **Health Check**: Validates container functionality
- **Entry Point**: Flexible startup script with environment variable support

## 🔧 Troubleshooting

### Common Issues

1. **Input file not found**: Ensure `symbolic_dataset.jsonl` exists in your mounted `/data/input/` directory
2. **Permission errors**: Make sure the mounted directories have proper read/write permissions
3. **Transcript directory not found**: The transcript directory is optional - remove `--transcript-dir` if not needed

### Debug Mode

To troubleshoot issues, run with environment variable to keep container running:

```bash
KEEP_RUNNING=true docker-compose -f docker-compose.converter.yml up --build
```

This will keep the container running after conversion for inspection.
