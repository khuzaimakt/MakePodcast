# AI Podcast Generator

![AI Podcast Generator](https://img.shields.io/badge/AI-Podcast%20Generator-D99B84)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A powerful AI tool that automatically generates professional podcasts from PDF documents. This application uses OpenAI's GPT-4o for content generation and offers two TTS (Text-to-Speech) options:

1. **OpenAI TTS** - High-quality voices from OpenAI
2. **Piper TTS** - Open-source alternative with customizable voices

## 🎙️ Features

- **PDF-to-Podcast Conversion**: Upload any PDF and transform it into a conversational podcast
- **Multiple Voices**: Generate podcasts with different voices for each speaker
- **Configurable Length**: Set the desired podcast length (default: 25 minutes)
- **Conversational Format**: Creates natural-sounding dialogue between hosts
- **Export Options**: Save your podcast as MP3, WAV, or OGG
- **Docker Support**: Easy deployment using Docker
- **Web Interface**: Simple Gradio UI for easy interaction

## 📋 Requirements

- Python 3.10+
- OpenAI API key
- Docker (optional, for containerized deployment)
- For Piper TTS: FFmpeg
- 4GB+ RAM recommended

## 🚀 Quick Start with Docker

The easiest way to get started is with Docker:

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-podcast-generator.git
cd ai-podcast-generator

# Create a .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# Start the container
docker-compose up -d
```

Then open your browser and navigate to `http://localhost:7860`

## 🔧 Manual Installation

If you prefer to run without Docker:

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-podcast-generator.git
cd ai-podcast-generator
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment setup**
```bash
# Create a .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

5. **Run the application**
```bash
python app.py
```

Then open your browser and navigate to `http://localhost:7860`

## 📦 Using Piper TTS (Open Source Option)

If you want to use the open-source Piper TTS instead of OpenAI's TTS:

1. **Install Piper**
   - The Docker container comes with Piper pre-installed
   - For manual installation, follow the [Piper installation guide](https://github.com/rhasspy/piper)

2. **Voice Models**
   - The container includes Ryan (male) and Lessac (female) voice models
   - For manual installation, download voice models from [Piper voices](https://huggingface.co/rhasspy/piper-voices)

3. **In the UI**
   - Select "Piper (Open Source)" as the Text-to-Speech Engine
   - Provide the path to the Piper executable and voices directory

## 🔍 How It Works

1. **Content Generation**
   - The system extracts text from your uploaded PDF
   - GPT-4o generates a podcast script with natural conversation between hosts
   - The script includes timing markers, speech patterns, and natural dialogue flow

2. **Voice Synthesis**
   - Either OpenAI TTS or Piper converts the text to speech
   - Different voices are used for different speakers
   - Audio segments are processed with natural pauses and transitions

3. **Audio Processing**
   - Segments are combined into a single audio file
   - Optional intro/outro music can be added
   - The final podcast is exported in your chosen format

## ⚙️ Configuration

You can customize the podcast generation by modifying the `config.json` file:

```json
{
    "episode_length": "25 minutes",
    "hosts": ["Ryan", "Lessac"],
    "podcast_name": "Your Podcast Name",
    "style": "conversational and informative",
    "audio": {
        "tts_model": "gpt-4o-mini-tts",
        "voices": {
            "default": {
                "voice": "onyx"
            },
            "Ryan": {
                "voice": "onyx"
            },
            "Lessac": {
                "voice": "nova"
            }
        },
        "intro_music": null,
        "outro_music": null,
        "add_silence_between_segments": true,
        "silence_duration": [0.4, 0.2, 0.6]
    },
    "openai_model": "gpt-4o",
    "temperature": 0.7
}
```

## 📝 Output Files

For each generated podcast, the system creates:
- A markdown script file
- An audio file in your chosen format
- A JSON metadata file with generation details

## 🛠️ Advanced Customization

### Custom Voice Models for Piper

1. Download additional voice models from [Piper voices](https://huggingface.co/rhasspy/piper-voices)
2. Place them in the voices directory
3. Update the configuration to use the new models

### Adding Music

1. Place your intro/outro music files in an accessible location
2. Update the configuration to point to these files:

```json
"intro_music": "/path/to/your/intro.mp3",
"outro_music": "/path/to/your/outro.mp3"
```

## 📊 Cost Estimation

The application provides cost estimates for:
- OpenAI API usage (for content generation)
- TTS processing (when using OpenAI TTS)

Approximate costs:
- Content generation: ~$0.05-0.10 per podcast (depends on PDF length)
- OpenAI TTS: ~$0.10-0.30 per podcast (depends on script length)
- Piper TTS: Free (open source)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [OpenAI](https://openai.com/) for GPT-4o and TTS models
- [Piper](https://github.com/rhasspy/piper) for the open-source TTS engine
- [Gradio](https://gradio.app/) for the web interface framework
- [PyDub](https://github.com/jiaaro/pydub) for audio processing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
