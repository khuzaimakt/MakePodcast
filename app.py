import os
import json
import logging
from dotenv import load_dotenv
from pathlib import Path

# Import the Gradio interface function from the podcast generator module
from podcast_generator import create_gradio_interface

# Load environment variables from .env file if present
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("podcast_generator.log")
    ]
)
logger = logging.getLogger("PodcastGeneratorApp")

def main():
    """Main function to start the Gradio app."""
    logger.info("Starting AI Podcast Generator app")
    
    # Check for configuration file
    config_path = Path("config.json")
    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            config = None
    else:
        logger.warning("No configuration file found")
        config = None
    
    # Check for API key in environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        logger.info("OpenAI API key found in environment variables")
    else:
        logger.warning("No OpenAI API key found in environment variables")
    
    # Create and launch the Gradio interface
    logger.info("Creating Gradio interface")
    app = create_gradio_interface()
    
    # Determine the host and port
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 7860))
    
    # Launch the app
    logger.info(f"Launching Gradio app on {host}:{port}")
    app.launch(server_name=host, server_port=port, share=False)

if __name__ == "__main__":
    main()