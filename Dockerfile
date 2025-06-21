# FROM python:3.10-slim

# # Set working directory
# WORKDIR /app

# # Set environment variables for better Python behavior in containers
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PIP_NO_CACHE_DIR=1 \
#     DEBIAN_FRONTEND=noninteractive

# # Install system dependencies (FFmpeg and other essentials)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ffmpeg \
#     git \
#     wget \
#     unzip \
#     build-essential \
#     libespeak-ng1 \
#     libsndfile1 \
#     espeak-ng \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python requirements
# COPY requirements.txt .
# RUN pip install --upgrade pip && \
#     pip install -r requirements.txt

# # Set up Piper properly with all dependencies
# RUN mkdir -p /app/piper/voices

# # Download and extract Piper with all necessary libraries
# RUN wget -O piper.tar.gz https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz && \
#     mkdir -p /app/piper && \
#     tar -xzf piper.tar.gz -C /app/piper && \
#     rm piper.tar.gz && \
#     chmod +x /app/piper/piper

# # Download voice models directly from HuggingFace
# # Ryan voice (male)
# RUN wget -O /app/piper/voices/en_US-ryan-medium.onnx \
#     "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx" && \
#     wget -O /app/piper/voices/en_US-ryan-medium.onnx.json \
#     "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"

# # Lessac voice (female)
# RUN wget -O /app/piper/voices/en_US-lessac-medium.onnx \
#     "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" && \
#     wget -O /app/piper/voices/en_US-lessac-medium.onnx.json \
#     "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

# # Copy application code
# COPY . .

# # Create directory for output files
# RUN mkdir -p /app/podcast_output

# # Create configuration file
# RUN echo '{ \
#     "episode_length": "25 minutes", \
#     "hosts": ["Ryan", "Lessac"], \
#     "podcast_name": "AI Podcast Generator", \
#     "style": "conversational and informative", \
#     "audio": { \
#         "tts_model": "piper", \
#         "piper_path": "/app/piper/piper", \
#         "voices_dir": "/app/piper/voices", \
#         "voices": { \
#             "default": { \
#                 "model": "en_US-ryan-medium", \
#                 "voice": "onyx", \
#                 "speaking_rate": 1.05 \
#             }, \
#             "Ryan": { \
#                 "model": "en_US-ryan-medium", \
#                 "voice": "onyx", \
#                 "speaking_rate": 1.05 \
#             }, \
#             "Lessac": { \
#                 "model": "en_US-lessac-medium", \
#                 "voice": "nova", \
#                 "speaking_rate": 1.05 \
#             } \
#         }, \
#         "intro_music": null, \
#         "outro_music": null, \
#         "add_silence_between_segments": true, \
#         "silence_duration": [0.4, 0.2, 0.6] \
#     }, \
#     "openai_model": "gpt-4o", \
#     "temperature": 0.7 \
# }' > /app/config.json

# # Test that Piper works correctly
# RUN echo "Testing Piper installation" > test.txt && \
#     /app/piper/piper --model /app/piper/voices/en_US-ryan-medium.onnx --text_file test.txt --output_file test.wav || echo "Piper test failed, but continuing build"

# # Expose Gradio port
# EXPOSE 7860

# # Command to run the Gradio application
# CMD ["python", "app.py"]

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables for better Python behavior in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including sudo
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    unzip \
    build-essential \
    libespeak-ng1 \
    libsndfile1 \
    espeak-ng \
    sudo \
    findutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and add to sudoers
RUN useradd -m appuser && \
    echo "appuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/appuser && \
    chmod 0440 /etc/sudoers.d/appuser

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set up Piper directories
RUN mkdir -p /app/piper && \
    mkdir -p /app/piper/voices && \
    mkdir -p /app/podcast_output

# Download and extract Piper with all necessary libraries
RUN wget -O piper.tar.gz https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz && \
    tar -xzf piper.tar.gz -C /app && \
    rm piper.tar.gz && \
    # Find the actual piper executable and move it to the expected location
    find /app -name piper -type f -exec cp {} /app/piper/piper-exec \; && \
    chmod 755 /app/piper/piper-exec && \
    # Create a symlink that will be used by the application
    ln -sf /app/piper/piper-exec /app/piper/piper && \
    # Make all shared libraries executable
    find /app -type f -name "*.so*" -exec chmod 755 {} \; || true && \
    # Print out the directory structure for debugging
    find /app -name piper -type f -o -name "*.so*" | sort

# Set ownership of all app files
RUN chown -R appuser:appuser /app

# Switch to non-root user for voice downloads
USER appuser

# Download voice models directly from HuggingFace
# Ryan voice (male)
RUN wget -O /app/piper/voices/en_US-ryan-medium.onnx \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx" && \
    wget -O /app/piper/voices/en_US-ryan-medium.onnx.json \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"

# Lessac voice (female)
RUN wget -O /app/piper/voices/en_US-lessac-medium.onnx \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" && \
    wget -O /app/piper/voices/en_US-lessac-medium.onnx.json \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

# Switch back to root user for copying and configuration
USER root

# Copy application code
COPY . .
RUN chown -R appuser:appuser /app

# Create configuration file with corrected piper path
RUN echo '{ \
    "episode_length": "25 minutes", \
    "hosts": ["Ryan", "Lessac"], \
    "podcast_name": "AI Podcast Generator", \
    "style": "conversational and informative", \
    "audio": { \
        "tts_model": "piper", \
        "piper_path": "/app/piper/piper", \
        "voices_dir": "/app/piper/voices", \
        "voices": { \
            "default": { \
                "model": "en_US-ryan-medium", \
                "voice": "onyx", \
                "speaking_rate": 1.05 \
            }, \
            "Ryan": { \
                "model": "en_US-ryan-medium", \
                "voice": "onyx", \
                "speaking_rate": 1.05 \
            }, \
            "Lessac": { \
                "model": "en_US-lessac-medium", \
                "voice": "nova", \
                "speaking_rate": 1.05 \
            } \
        }, \
        "intro_music": null, \
        "outro_music": null, \
        "add_silence_between_segments": true, \
        "silence_duration": [0.4, 0.2, 0.6] \
    }, \
    "openai_model": "gpt-4o", \
    "temperature": 0.7 \
}' > /app/config.json && \
    chown appuser:appuser /app/config.json

# Verify Piper permissions and test the installation
USER appuser
RUN ls -la /app/piper/ && \
    ls -la $(find /app -name piper -type f) && \
    echo "Testing Piper installation..." && \
    echo "This is a test" > test.txt && \
    /app/piper/piper --model /app/piper/voices/en_US-ryan-medium.onnx --text_file test.txt --output_file test.wav || echo "Piper test failed, but continuing build"

# Expose Gradio port
EXPOSE 7860

# Command to run the Gradio application
CMD ["python", "app.py"]