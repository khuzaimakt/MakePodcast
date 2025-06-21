import os
import json
import re
import time
import tempfile
import subprocess
import requests
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import random
import PyPDF2
import tiktoken
import csv
import openai
from pydub import AudioSegment
import gradio as gr

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PodcastGenerator')

def calculate_gpt4o_cost(text: str, response_type: str) -> dict:
    """
    Calculates the token count and cost for input or output using GPT-4o pricing.
    """
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(text)
    token_count = len(tokens)

    if response_type.lower() == "input":
        cost_per_million = 5.0  # $5 per 1M tokens
    elif response_type.lower() == "output":
        cost_per_million = 20.0  # $20 per 1M tokens
    else:
        raise ValueError("response_type must be 'input' or 'output'.")

    cost = (token_count / 1_000_000) * cost_per_million

    return {
        "type": response_type,
        "token_count": token_count,
        "estimated_cost_usd": round(cost, 6)
    }

def calculate_tts_cost(text: str, model: str) -> dict:
    """
    Calculates the approximate cost for TTS using OpenAI's pricing.
    """
    char_count = len(text)
    
    # Pricing per 1M characters
    if model == "gpt-4o-mini-tts":
        cost_per_million = 0.60
    elif model == "tts-1":
        cost_per_million = 15.0
    elif model == "tts-1-hd":
        cost_per_million = 30.0
    else:
        raise ValueError(f"Unknown TTS model: {model}")
    
    cost = (char_count / 1_000_000) * cost_per_million
    
    return {
        "model": model,
        "char_count": char_count,
        "estimated_cost_usd": round(cost, 6)
    }

def extract_text_from_pdf(pdf_path):
    """
    Extract all text from a PDF file.
    """
    try:
        # Open the PDF file in read-binary mode
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get the number of pages
            num_pages = len(pdf_reader.pages)
            
            # Initialize an empty string to store text
            text = ""
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
                
            return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

class PodcastGenerator:
    """Base class for podcast generation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the podcast generator with configuration options."""
        self.config = config
    
    def call_openai_all_in_one(self, topic_name: str, content: str = None) -> Dict[str, str]:
        """Make a single API call to OpenAI to generate all podcast content with improved length control."""
        t1 = time.time()
        model = self.config["openai_model"]
        temperature = self.config["temperature"]
        system_prompt = f"""
        You are an expert podcast content creator. You will generate an entire podcast package that is PRECISELY {self.config["episode_length"]} in length when read aloud at a natural speaking pace. 
        
        The podcast should discuss the topic in a well-structured manner, including:
        - Introduction to the topic
        - Key background information
        - Important concepts and definitions
        - Current trends and developments
        - Different perspectives on the topic
        - Practical implications or applications
        - Future outlook
        - Conclusion with key takeaways
        
        Podcast details:
        - Name: {self.config["podcast_name"]}
        - Hosts: {", ".join(self.config["hosts"])}
        - Target length: {self.config["episode_length"]} (STRICTLY enforce this - the final spoken podcast MUST be exactly {self.config["episode_length"]})
        - Style: {self.config["style"]}
        
        To ensure the podcast is EXACTLY {self.config["episode_length"]} in length:
        - A 25-minute podcast should have approximately 3,250-3,750 words of dialogue
        - Include explicit timing guidance throughout the script with [TIME CHECK: X:XX] markers 
        - Balance the speaking time between hosts
        - Include appropriate pacing instructions
        
        Generate a complete, natural conversational script formatted with:
        - Each speaker's lines as "SPEAKER NAME: Their dialogue text"
        - Natural speech markers:
            * [pause-short] for brief pauses (0.3s)
            * [pause-medium] for medium pauses (0.7s)
            * [pause-long] for longer pauses (1.2s)
            * [emphasis] around emphasized words
            * [breath] where speakers would naturally take a breath
            * [thoughtful] for moments of consideration
        - Include natural filler words like "um", "uh", "you know" occasionally
        - Mark sound effects as [SOUND EFFECT: description]
        - Mark transitions as [TRANSITION]
        
        Format your response with these exact headings:
        <PODCAST_SCRIPT>
        [Insert full script here]
        </PODCAST_SCRIPT>
        """
        
        # Create user prompt based on topic and optional content
        user_prompt = f"""
        Create a complete podcast about this topic: "{topic_name}"
        
        Use this content as reference information for the podcast:
        {content}
        
        The podcast MUST be EXACTLY {self.config["episode_length"]} in length when read aloud at a natural pace.
        
        Make the content engaging, informative, and accessible to a general audience. Ensure that:
        1. Technical concepts are explained clearly
        2. Examples and analogies are used to illustrate complex ideas
        3. The conversation flows naturally between the hosts
        4. Different perspectives on the topic are presented
        5. The hosts occasionally ask each other questions
        
        Create a podcast that is both educational and entertaining, with a good balance of facts, analysis, and discussion.
        
        Format your response with these exact headings:
        <PODCAST_SCRIPT>
        [Insert full script here]
        </PODCAST_SCRIPT>
        """
        
        # Rest of your function stays the same...
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 16000
        }
        
        try:
            logger.info(f"Making single comprehensive call to OpenAI API with model: {model}")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            
            full_response = response.json()["choices"][0]["message"]["content"]
            output_tokens = calculate_gpt4o_cost(full_response,'output')
            
            # Extract script from the response
            script_match = re.search(r'<PODCAST_SCRIPT>(.*?)</PODCAST_SCRIPT>', full_response, re.DOTALL)
            
            script = script_match.group(1).strip() if script_match else "Script not found in API response"
            
            logger.info("Successfully extracted podcast content from API response")
            t2 = time.time()
            
            return {
                "topic": topic_name,
                "script": script,
                "openai_time": f"{t2-t1:.2f}",
                "output_tokens": output_tokens
            }
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    def estimate_podcast_length(self, script: str) -> float:
        """
        Estimate the length of the podcast in minutes based on the script word count.
        """
        # Extract just the dialogue text (remove speaker names, sound effects, etc.)
        dialogue_pattern = re.compile(r'^[A-Z][A-Za-z\s\-\']+:\s+(.+)$', re.MULTILINE)
        dialogue_matches = dialogue_pattern.findall(script)
        
        # Join all dialogue
        all_dialogue = ' '.join(dialogue_matches)
        
        # Remove speech markers
        all_dialogue = re.sub(r'\[pause-short\]|\[pause-medium\]|\[pause-long\]|\[emphasis\]|\[/emphasis\]|\[breath\]|\[thoughtful\]', '', all_dialogue)
        
        # Count words
        word_count = len(all_dialogue.split())
        
        # Estimate spoken time
        # Average speaking pace: ~130-150 words per minute
        # We'll use 130 to account for pauses and transitions
        estimated_minutes = word_count / 130
        
        logger.info(f"Script has {word_count} words, estimated podcast length: {estimated_minutes:.2f} minutes")
        
        return estimated_minutes

    def validate_podcast_length(self, script: str) -> Dict[str, Any]:
        """
        Validate if the podcast script meets the target length requirements.
        """
        # Get target length in minutes
        target_length_str = self.config["episode_length"]
        # Extract numeric value (assume format like "25 minutes")
        target_minutes = float(re.search(r'(\d+)', target_length_str).group(1))
        
        # Calculate estimated length
        estimated_minutes = self.estimate_podcast_length(script)
        
        # Calculate difference
        difference = abs(estimated_minutes - target_minutes)
        
        # Set acceptance threshold (10% of target)
        threshold = target_minutes * 0.1
        
        # Check if within acceptable range
        is_valid = difference <= threshold
        
        # Create validation result
        result = {
            "is_valid": is_valid,
            "target_minutes": target_minutes,
            "estimated_minutes": estimated_minutes,
            "difference_minutes": difference,
            "threshold_minutes": threshold,
            "word_count": len(script.split()),
            "message": f"Podcast length {'is' if is_valid else 'is NOT'} within acceptable range. " 
                    f"Target: {target_minutes:.1f} min, Estimated: {estimated_minutes:.1f} min, "
                    f"Difference: {difference:.1f} min, Threshold: {threshold:.1f} min."
        }
        
        logger.info(result["message"])
        return result
    
    def parse_script_for_audio(self, script: str) -> List[Dict[str, str]]:
        """Parse the script to extract dialogue lines and audio cues."""
        
        # Split the script into lines
        lines = script.strip().split('\n')
        parsed_lines = []
        
        # Regular expressions for different line types
        speaker_pattern = re.compile(r'^([A-Z][A-Za-z\s\-\']+):\s+(.+)$')
        sound_effect_pattern = re.compile(r'^\[SOUND EFFECT:([^\]]+)\]$')
        transition_pattern = re.compile(r'^\[TRANSITION\]$')
        timestamp_pattern = re.compile(r'^\[\d+:\d+\]$')  # To ignore timestamp lines
        
        current_speaker = None
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip timestamp lines
            if timestamp_pattern.match(line):
                continue
                
            # Check if it's a speaker line
            speaker_match = speaker_pattern.match(line)
            if speaker_match:
                # If we have accumulated text for the previous speaker, add it
                if current_speaker and current_text:
                    parsed_lines.append({
                        "type": "dialogue",
                        "speaker": current_speaker,
                        "text": " ".join(current_text)
                    })
                    current_text = []
                
                # Set the new current speaker and text
                current_speaker = speaker_match.group(1).strip()  # Strip to remove any extra spaces
                current_text = [speaker_match.group(2)]
                
                # Debug output
                logger.debug(f"Found speaker: '{current_speaker}' with text: '{current_text[0][:30]}...'")
                continue
                
            # Check if it's a sound effect
            sound_effect_match = sound_effect_pattern.match(line)
            if sound_effect_match:
                # If we have accumulated text, add it first
                if current_speaker and current_text:
                    parsed_lines.append({
                        "type": "dialogue",
                        "speaker": current_speaker,
                        "text": " ".join(current_text)
                    })
                    current_text = []
                
                # Add the sound effect
                parsed_lines.append({
                    "type": "sound_effect",
                    "text": sound_effect_match.group(1).strip()
                })
                continue
                
            # Check if it's a transition
            if transition_pattern.match(line):
                # If we have accumulated text, add it first
                if current_speaker and current_text:
                    parsed_lines.append({
                        "type": "dialogue",
                        "speaker": current_speaker,
                        "text": " ".join(current_text)
                    })
                    current_text = []
                
                # Add the transition
                parsed_lines.append({
                    "type": "transition",
                    "text": "Transition"
                })
                continue
                
            # Otherwise, it's continuation of the current speaker's text
            if current_speaker:
                current_text.append(line)
        
        # Add the last speaker's text if any
        if current_speaker and current_text:
            parsed_lines.append({
                "type": "dialogue",
                "speaker": current_speaker,
                "text": " ".join(current_text)
            })
        
        # Debug: Print all parsed speakers
        speakers = set(segment["speaker"] for segment in parsed_lines if segment["type"] == "dialogue")
        logger.info(f"Detected speakers in script: {speakers}")
            
        return parsed_lines
    
    def save_podcast_to_files(self, podcast: Dict[str, str], output_dir: str = "./podcast_output") -> Dict[str, str]:
        """Save podcast components to files with UUID for uniqueness."""
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a safe filename from the topic
        safe_topic = "".join(c if c.isalnum() else "_" for c in podcast["topic"])
        
        # Generate a UUID for the files
        file_uuid = str(uuid.uuid4())
        
        # Save script with UUID
        script_path = f"{output_dir}/{safe_topic}_{file_uuid}_script.md"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(podcast["script"])
            logger.info(f"Saved script to {script_path}")
        
        return {
            "safe_topic": safe_topic,
            "file_uuid": file_uuid,
            "script_path": script_path
        }

    def preprocess_text_for_tts(self, text: str) -> str:
        """Preprocess text for better TTS results."""
        # To be implemented by subclasses
        pass
    
    def generate_audio_from_script(self, script: str, output_dir: str, filename_base: str) -> str:
        """Generate a full podcast audio file from the script."""
        # To be implemented by subclasses
        pass
    
    def create_podcast_with_audio(self, topic_name: str, content: str = None, output_dir: str = "./podcast_output"):
        """Create a complete podcast including audio with length validation."""
        # To be implemented by subclasses
        pass
    
    def generate_metadata_json(self, result: Dict[str, str], output_dir: str):
        """Generate a metadata JSON file for the podcast with all relevant information."""
        # To be implemented by subclasses
        pass


class OpenAIPodcastGenerator(PodcastGenerator):
    """Podcast generator using OpenAI TTS"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, api_key: str = None):
        """Initialize the OpenAI podcast generator."""
        super().__init__(config)
        
        # Set up OpenAI API key
        self.openai_api_key = api_key
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")
        
        # Initialize OpenAI client
        openai.api_key = self.openai_api_key
    
    def preprocess_text_for_tts(self, text: str) -> str:
        """Preprocess text for better TTS results."""
        
        # Process speech markers with direct replacements
        processed_text = text
        
        # Remove speaker prefixes (e.g., "RYAN: ")
        processed_text = re.sub(r'^[A-Z][A-Za-z\s\-\']+:\s+', '', processed_text)
        
        # Replace pause markers with actual periods or commas to force natural pauses
        processed_text = processed_text.replace("[pause-short]", ",")
        processed_text = processed_text.replace("[pause-medium]", ".")
        processed_text = processed_text.replace("[pause-long]", "...")
        
        # Process emphasis - can use SSML-like approach but remove the tags
        processed_text = re.sub(r'\[emphasis\](.*?)\[/emphasis\]', r'\1', processed_text)
        processed_text = re.sub(r'\[emphasis\](.*?)(?!\[/emphasis\])', r'\1', processed_text)
        
        # Process breath markers - add a comma to force a pause
        processed_text = processed_text.replace("[breath]", ",")
        
        # Process thoughtful markers - add ellipsis for longer pause
        processed_text = processed_text.replace("[thoughtful]", "...")
        
        # Clean up any remaining markers that we don't explicitly handle
        processed_text = re.sub(r'\[.*?\]', '', processed_text)
        
        return processed_text
    
    def generate_speech_with_openai(self, text: str, output_file: str, voice: str) -> str:
        """
        Generate speech using OpenAI's TTS API.
        """
        try:
            # Process text for TTS
            processed_text = self.preprocess_text_for_tts(text)
            
            # Calculate TTS cost (for logging purposes)
            tts_cost = calculate_tts_cost(processed_text, self.config["audio"]["tts_model"])
            logger.info(f"TTS request: {len(processed_text)} chars, est. cost: ${tts_cost['estimated_cost_usd']}")
            
            # Create a Path object for the output file
            output_path = Path(output_file)
            
            # Call OpenAI's TTS API and stream to file
            logger.info(f"Generating speech with OpenAI TTS using voice '{voice}'...")
            
            with openai.audio.speech.with_streaming_response.create(
                model=self.config["audio"]["tts_model"],
                voice=voice,
                input=processed_text
            ) as response:
                response.stream_to_file(output_path)
            
            # Verify the output file was created
            if not os.path.exists(output_file):
                logger.error(f"Error: Output file '{output_file}' was not created!")
                return None
                
            logger.info(f"Generated audio file: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error generating speech with OpenAI TTS: {e}")
            return None
    
    def generate_audio_from_script(self, script: str, output_dir: str, filename_base: str) -> str:
        """Generate a full podcast audio file from the script."""
        
        logger.info("Starting audio generation from script...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a temporary directory for audio segments
        temp_audio_dir = tempfile.mkdtemp(prefix="temp_audio_segments_")
        
        # Parse the script into segments
        segments = self.parse_script_for_audio(script)
        
        # Final audio will be built from these segments
        final_audio = AudioSegment.silent(duration=500)  # Start with 0.5s silence
        
        # Add intro music if specified
        if self.config["audio"].get("intro_music") and os.path.exists(self.config["audio"]["intro_music"]):
            try:
                logger.info("Adding intro music...")
                intro_music = AudioSegment.from_file(self.config["audio"]["intro_music"])
                # Fade in and out, limit to 15 seconds
                intro_music = intro_music[:15000].fade_in(1000).fade_out(1000)
                final_audio += intro_music
                final_audio += AudioSegment.silent(duration=1000)  # 1s silence after intro
            except Exception as e:
                logger.warning(f"Couldn't add intro music: {e}")
        
        # Process each segment
        logger.info(f"Processing {len(segments)} script segments...")
        for i, segment in enumerate(segments):
            # Use a temporary file for each segment
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                segment_file = temp_file.name
            
            if segment["type"] == "dialogue":
                # Get the speaker and text
                speaker = segment["speaker"]
                text = segment["text"]
                
                # Get voice configuration for this speaker
                voice_config = None
                
                # Try exact match first
                if speaker in self.config["audio"]["voices"]:
                    voice_config = self.config["audio"]["voices"][speaker]
                    logger.info(f"Using exact voice match for '{speaker}'")
                else:
                    # Try case-insensitive match
                    speaker_lower = speaker.lower()
                    for configured_speaker, config in self.config["audio"]["voices"].items():
                        if configured_speaker.lower() == speaker_lower:
                            voice_config = config
                            logger.info(f"Using case-insensitive voice match for '{speaker}' -> '{configured_speaker}'")
                            break
                
                # If still no match, use default
                if voice_config is None:
                    voice_config = self.config["audio"]["voices"]["default"]
                    logger.warning(f"Using default voice for '{speaker}' (no match found)")
                
                # Generate audio
                try:
                    voice = voice_config.get("voice")
                    
                    logger.info(f"Generating audio for '{speaker}' using voice '{voice}'")
                    
                    success = self.generate_speech_with_openai(
                        text, 
                        segment_file, 
                        voice
                    )
                    
                    # Add the segment to the final audio
                    if success and os.path.exists(segment_file):
                        segment_audio = AudioSegment.from_file(segment_file)
                        final_audio += segment_audio
                        
                        # Add silence between segments if configured
                        if self.config["audio"]["add_silence_between_segments"]:
                            silence_ms = int(random.choice(self.config["audio"]["silence_duration"]) * 1000)
                            final_audio += AudioSegment.silent(duration=silence_ms)
                    else:
                        logger.warning(f"Output file not created for segment {i}, adding silent placeholder")
                        # Add a silent segment as a placeholder (3 seconds per 100 characters of text)
                        silence_duration = min(10000, max(2000, len(text) * 30))  # Between 2-10 seconds
                        final_audio += AudioSegment.silent(duration=silence_duration)
                    
                    # Delete the temporary segment file
                    try:
                        if os.path.exists(segment_file):
                            os.unlink(segment_file)
                    except Exception as e:
                        logger.warning(f"Could not delete temporary file {segment_file}: {e}")
                        
                except Exception as e:
                    logger.warning(f"Couldn't generate audio for segment {i}: {e}")
                    # Add a silent segment as a placeholder
                    silence_duration = min(10000, max(2000, len(text) * 30))  # Between 2-10 seconds
                    final_audio += AudioSegment.silent(duration=silence_duration)
                    
                    # Delete the temporary segment file
                    try:
                        if os.path.exists(segment_file):
                            os.unlink(segment_file)
                    except Exception as e2:
                        logger.warning(f"Could not delete temporary file {segment_file}: {e2}")
            
            elif segment["type"] == "sound_effect":
                # Add a sound effect placeholder (silence with a marker)
                final_audio += AudioSegment.silent(duration=500)
            
            elif segment["type"] == "transition":
                # Add a transition (longer silence)
                final_audio += AudioSegment.silent(duration=1500)
        
        # Add outro music if specified
        if self.config["audio"].get("outro_music") and os.path.exists(self.config["audio"]["outro_music"]):
            try:
                logger.info("Adding outro music...")
                outro_music = AudioSegment.from_file(self.config["audio"]["outro_music"])
                # Fade in and out, limit to 15 seconds
                outro_music = outro_music[:15000].fade_in(1000).fade_out(1000)
                final_audio += AudioSegment.silent(duration=1000)  # 1s silence before outro
                final_audio += outro_music
            except Exception as e:
                logger.warning(f"Couldn't add outro music: {e}")
        
        # Generate a UUID for the output file for uniqueness
        file_uuid = str(uuid.uuid4())
        
        # Export the final audio file with UUID in the name
        final_audio_path = os.path.join(output_dir, f"{filename_base}_{file_uuid}.mp3")
        logger.info(f"Exporting final audio to {final_audio_path}...")
        final_audio.export(final_audio_path, format="mp3", bitrate="192k")
        
        # Try to remove the temporary directory
        try:
            # Cleanup the temporary directory
            import shutil
            shutil.rmtree(temp_audio_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Could not remove temporary directory {temp_audio_dir}: {e}")
        
        logger.info(f"Audio generation complete!")
        return final_audio_path
    
    def create_podcast_with_audio(self, topic_name: str, content: str = None, output_dir: str = "./podcast_output"):
        """Create a complete podcast including audio with length validation."""
        
        # Generate all podcast content in a single API call
        logger.info(f"Generating complete podcast on: {topic_name} with single OpenAI API call")
        podcast = self.call_openai_all_in_one(topic_name, content)
        
        # Validate podcast length
        validation = self.validate_podcast_length(podcast["script"])
        
        # Save text content to files and get file info
        file_info = self.save_podcast_to_files(podcast, output_dir)
        
        # Generate audio using the safe topic and UUID
        audio_path = self.generate_audio_from_script(
            podcast["script"], 
            output_dir, 
            file_info["safe_topic"]
        )
        
        return {
            "topic": topic_name,
            "script_path": file_info["script_path"],
            "audio_path": audio_path,
            "uuid": file_info["file_uuid"],
            "length_validation": validation,
            "openai_time": podcast["openai_time"],
            "output_tokens": podcast["output_tokens"]
        }
    
    def generate_metadata_json(self, result: Dict[str, str], output_dir: str):
        """Generate a metadata JSON file for the podcast with all relevant information."""
        metadata = {
            "topic": result["topic"],
            "uuid": result["uuid"],
            "script_path": os.path.basename(result["script_path"]),
            "audio_path": os.path.basename(result["audio_path"]),
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "openai_time": result["openai_time"],
            "file_path": result.get("path", ""),
            "config": {
                "podcast_name": self.config["podcast_name"],
                "hosts": self.config["hosts"],
                "episode_length": self.config["episode_length"],
                "style": self.config["style"],
                "tts_model": self.config["audio"]["tts_model"]
            },
            "input_tokens": result.get("input_tokens", {}),
            "output_tokens": result.get("output_tokens", {}),
            "overall_time": result.get("overall_time", 0),
            "method": "openai_tts"
        }
        
        # Add length validation info if available
        if "length_validation" in result:
            metadata["length_validation"] = {
                "estimated_minutes": result["length_validation"]["estimated_minutes"],
                "target_minutes": result["length_validation"]["target_minutes"],
                "is_valid": result["length_validation"]["is_valid"],
                "message": result["length_validation"]["message"]
            }
        
        # Save metadata to JSON file
        metadata_path = os.path.join(output_dir, f"{os.path.basename(result['audio_path']).split('.')[0]}_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Generated metadata file: {metadata_path}")
        return metadata_path


class PiperPodcastGenerator(PodcastGenerator):
    """Podcast generator using Piper TTS"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Piper podcast generator."""
        super().__init__(config)
        
        # Set up OpenAI API key for content generation
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found for content generation. Set the OPENAI_API_KEY environment variable.")
        
        # Check available voice models
        self.available_models = self.check_voice_models()
    
    def check_voice_models(self) -> List[str]:
        """Check which voice models are available and return a list of model names."""
        logger.info(f"Checking for voice models in: {self.config['audio']['voices_dir']}")
        
        available_models = []
        
        if not os.path.exists(self.config['audio']['voices_dir']):
            logger.error(f"Voice directory does not exist: {self.config['audio']['voices_dir']}")
            return available_models
        
        models = [f for f in os.listdir(self.config['audio']['voices_dir']) if f.endswith('.onnx')]
        
        if not models:
            logger.error("No voice models (.onnx files) found!")
        else:
            logger.info(f"Found {len(models)} voice models:")
            for model in models:
                model_name = model[:-5]  # Remove .onnx extension
                available_models.append(model_name)
                logger.info(f"  - {model_name}")
        
        return available_models
    
    def select_available_voice(self, preferred_voice: str) -> str:
        """Select an available voice model, using fallbacks if necessary."""
        if preferred_voice in self.available_models:
            return preferred_voice
        
        # If Ryan voice isn't available but requested, warn and use available voice
        if preferred_voice == "en_US-ryan-medium" and "en_US-lessac-medium" in self.available_models:
            logger.warning(f"Ryan voice not available, using lessac voice instead")
            return "en_US-lessac-medium"
            
        # If Lessac voice isn't available but requested, warn and use available voice  
        if preferred_voice == "en_US-lessac-medium" and "en_US-ryan-medium" in self.available_models:
            logger.warning(f"Lessac voice not available, using ryan voice instead")
            return "en_US-ryan-medium"
        
        # If no specific fallbacks are available, use the first available model
        if self.available_models:
            logger.warning(f"Using first available voice model: '{self.available_models[0]}'")
            return self.available_models[0]
        
        # If no models available, raise error
        raise ValueError("No voice models available! Please check your Piper installation.")
    
    def preprocess_text_for_tts(self, text: str) -> str:
        """Preprocess text for better TTS results without SSML (since Piper might not properly support it)."""
        
        # Process speech markers with direct replacements instead of SSML
        processed_text = text
        
        # Replace pause markers with actual periods to force natural pauses
        processed_text = processed_text.replace("[pause-short]", ".")
        processed_text = processed_text.replace("[pause-medium]", ". .")
        processed_text = processed_text.replace("[pause-long]", ". . .")
        
        # Process emphasis - can't use SSML tags so just keep the text
        processed_text = re.sub(r'\[emphasis\](.*?)\[/emphasis\]', r'\1', processed_text)
        processed_text = re.sub(r'\[emphasis\](.*?)(?!\[/emphasis\])', r'\1', processed_text)
        
        # Process breath markers - add a period to force a pause
        processed_text = processed_text.replace("[breath]", ".")
        
        # Process thoughtful markers - add multiple periods for longer pause
        processed_text = processed_text.replace("[thoughtful]", ". . .")
        
        # Clean up any remaining markers that we don't explicitly handle
        processed_text = re.sub(r'\[.*?\]', '', processed_text)
        
        return processed_text
    
    def generate_speech_with_piper(self, text: str, output_file: str, voice_model: str, speaking_rate: float = 1.0) -> str:
        """Generate speech using Piper TTS with improved voice model handling and proper SSML support."""
        # Select an available voice model
        voice_model = self.select_available_voice(voice_model)
        
        # Preprocess text for Piper TTS
        processed_text = self.preprocess_text_for_tts(text)
        
        # Create a temporary file for the text
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding='utf-8') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(processed_text)
        
        # Build the full path to the model and verify it exists
        model_path = os.path.join(self.config["audio"]["voices_dir"], f"{voice_model}.onnx")
        
        if not os.path.exists(model_path):
            logger.error(f"Voice model file not found: {model_path}")
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            return None
        
        # Ensure the command uses proper quoting for Windows paths
        # Add --sentence-silence parameter to ensure pauses between sentences
        # cmd = [
        #     'cmd', '/c',  # Use Windows CMD
        #     f'type "{temp_file_path}" | '
        #     f'"{self.config["audio"]["piper_path"]}" '
        #     f'--model "{model_path}" '
        #     f'--output_file "{output_file}" '
        #     f'--sentence-silence 0.5'  # Increased sentence silence for better pauses
        # ]
#         cmd = [
#     'bash', '-c',
#     f'cat "{temp_file_path}" | '
#     f'"{self.config["audio"]["piper_path"]}" '
#     f'--model "{model_path}" '
#     f'--output_file "{output_file}" '
#     f'--sentence-silence 0.5'
# ]
        cmd = [
    'sudo', 'bash', '-c',
    f'cat "{temp_file_path}" | '
    f'"{self.config["audio"]["piper_path"]}" '
    f'--model "{model_path}" '
    f'--output_file "{output_file}" '
    f'--sentence-silence 0.5'
]

        logger.debug(f"Running Piper command: {' '.join(cmd)}")
        
        try:
            # Run the command
            logger.info(f"Generating audio with Piper using model '{voice_model}'...")
            logger.debug(f"Text sample: {text[:50]}...")
            
            # Run as a single string for Windows
            process = subprocess.run(cmd, shell=False, capture_output=True)
            
            # Check for errors
            if process.returncode != 0:
                logger.error(f"Piper error: {process.stderr.decode('utf-8', errors='ignore')}")
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                return None
            
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
            # Verify the output file was created
            if not os.path.exists(output_file):
                logger.error(f"Error: Output file '{output_file}' was not created!")
                return None
            
            # Apply speed adjustment if needed
            if speaking_rate != 1.0 and os.path.exists(output_file):
                logger.info(f"Adjusting speaking rate to {speaking_rate}...")
                # Load the audio
                audio = AudioSegment.from_file(output_file)
                
                # Adjust speed
                if speaking_rate > 1.0:
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate * speaking_rate)
                    }).set_frame_rate(audio.frame_rate)
                else:
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate / (2.0 - speaking_rate))
                    }).set_frame_rate(audio.frame_rate)
                
                # Save the modified audio
                audio.export(output_file, format="wav")
            
            return output_file
        
        except Exception as e:
            logger.error(f"Error generating speech with Piper TTS: {e}")
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            return None
    
    def generate_audio_from_script(self, script: str, output_dir: str, filename_base: str) -> str:
        """Generate a full podcast audio file from the script with improved voice selection."""
        
        logger.info("Starting audio generation from script...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a temporary directory for audio segments
        temp_audio_dir = tempfile.mkdtemp(prefix="temp_audio_segments_")
        
        # Parse the script into segments
        segments = self.parse_script_for_audio(script)
        
        # Final audio will be built from these segments
        final_audio = AudioSegment.silent(duration=500)  # Start with 0.5s silence
        
        # Add intro music if specified
        if self.config["audio"].get("intro_music") and os.path.exists(self.config["audio"]["intro_music"]):
            try:
                logger.info("Adding intro music...")
                intro_music = AudioSegment.from_file(self.config["audio"]["intro_music"])
                # Fade in and out, limit to 15 seconds
                intro_music = intro_music[:15000].fade_in(1000).fade_out(1000)
                final_audio += intro_music
                final_audio += AudioSegment.silent(duration=1000)  # 1s silence after intro
            except Exception as e:
                logger.warning(f"Couldn't add intro music: {e}")
        
        # Process each segment
        logger.info(f"Processing {len(segments)} script segments...")
        for i, segment in enumerate(segments):
            # Use a temporary file for each segment that will be deleted automatically
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                segment_file = temp_file.name
            
            if segment["type"] == "dialogue":
                # Get the speaker and text
                speaker = segment["speaker"]
                text = segment["text"]
                
                # Get voice configuration for this speaker with improved fallback
                voice_config = None
                
                # Try exact match first
                if speaker in self.config["audio"]["voices"]:
                    voice_config = self.config["audio"]["voices"][speaker]
                    logger.info(f"Using exact voice match for '{speaker}'")
                else:
                    # Try case-insensitive match
                    speaker_lower = speaker.lower()
                    for configured_speaker, config in self.config["audio"]["voices"].items():
                        if configured_speaker.lower() == speaker_lower:
                            voice_config = config
                            logger.info(f"Using case-insensitive voice match for '{speaker}' -> '{configured_speaker}'")
                            break
                
                # If still no match, use default
                if voice_config is None:
                    voice_config = self.config["audio"]["voices"]["default"]
                    logger.warning(f"Using default voice for '{speaker}' (no match found)")
                
                # Generate audio
                try:
                    voice_model = voice_config.get("model")
                    speaking_rate = voice_config.get("speaking_rate", 1.0)
                    
                    logger.info(f"Generating audio for '{speaker}' using model '{voice_model}'")
                    
                    success = self.generate_speech_with_piper(
                        text, 
                        segment_file, 
                        voice_model,
                        speaking_rate
                    )
                    
                    # Add the segment to the final audio
                    if success and os.path.exists(segment_file):
                        segment_audio = AudioSegment.from_file(segment_file)
                        final_audio += segment_audio
                        
                        # Add silence between segments if configured
                        if self.config["audio"]["add_silence_between_segments"]:
                            silence_ms = int(random.choice(self.config["audio"]["silence_duration"]) * 1000)
                            final_audio += AudioSegment.silent(duration=silence_ms)
                    else:
                        logger.warning(f"Output file not created for segment {i}, adding silent placeholder")
                        # Add a silent segment as a placeholder (3 seconds per 100 characters of text)
                        silence_duration = min(10000, max(2000, len(text) * 30))  # Between 2-10 seconds
                        final_audio += AudioSegment.silent(duration=silence_duration)
                    
                    # Delete the temporary segment file
                    try:
                        if os.path.exists(segment_file):
                            os.unlink(segment_file)
                    except Exception as e:
                        logger.warning(f"Could not delete temporary file {segment_file}: {e}")
                        
                except Exception as e:
                    logger.warning(f"Couldn't generate audio for segment {i}: {e}")
                    # Add a silent segment as a placeholder
                    silence_duration = min(10000, max(2000, len(text) * 30))  # Between 2-10 seconds
                    final_audio += AudioSegment.silent(duration=silence_duration)
                    
                    # Delete the temporary segment file
                    try:
                        if os.path.exists(segment_file):
                            os.unlink(segment_file)
                    except Exception as e2:
                        logger.warning(f"Could not delete temporary file {segment_file}: {e2}")
            
            elif segment["type"] == "sound_effect":
                # Add a sound effect placeholder (silence with a marker)
                final_audio += AudioSegment.silent(duration=500)
            
            elif segment["type"] == "transition":
                # Add a transition (longer silence)
                final_audio += AudioSegment.silent(duration=1500)
        
        # Add outro music if specified
        if self.config["audio"].get("outro_music") and os.path.exists(self.config["audio"]["outro_music"]):
            try:
                logger.info("Adding outro music...")
                outro_music = AudioSegment.from_file(self.config["audio"]["outro_music"])
                # Fade in and out, limit to 15 seconds
                outro_music = outro_music[:15000].fade_in(1000).fade_out(1000)
                final_audio += AudioSegment.silent(duration=1000)  # 1s silence before outro
                final_audio += outro_music
            except Exception as e:
                logger.warning(f"Couldn't add outro music: {e}")
        
        # Generate a UUID for the output file for uniqueness
        file_uuid = str(uuid.uuid4())
        
        # Export the final audio file with UUID in the name
        final_audio_path = os.path.join(output_dir, f"{filename_base}_{file_uuid}.mp3")
        logger.info(f"Exporting final audio to {final_audio_path}...")
        final_audio.export(final_audio_path, format="mp3", bitrate="192k")
        
        # Try to remove the temporary directory
        try:
            # Cleanup the temporary directory
            import shutil
            shutil.rmtree(temp_audio_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Could not remove temporary directory {temp_audio_dir}: {e}")
        
        logger.info(f"Audio generation complete!")
        return final_audio_path
    
    def create_podcast_with_audio(self, topic_name: str, content: str = None, output_dir: str = "./podcast_output"):
        """Create a complete podcast including audio with length validation."""
        
        # Generate all podcast content in a single API call
        logger.info(f"Generating complete podcast on: {topic_name} with single OpenAI API call")
        podcast = self.call_openai_all_in_one(topic_name, content)
        
        # Validate podcast length
        validation = self.validate_podcast_length(podcast["script"])
        
        # Save text content to files and get file info
        file_info = self.save_podcast_to_files(podcast, output_dir)
        
        # Generate audio using the safe topic and UUID
        audio_path = self.generate_audio_from_script(
            podcast["script"], 
            output_dir, 
            file_info["safe_topic"]
        )
        
        return {
            "topic": topic_name,
            "script_path": file_info["script_path"],
            "audio_path": audio_path,
            "uuid": file_info["file_uuid"],
            "length_validation": validation,
            "openai_time": podcast["openai_time"],
            "output_tokens": podcast["output_tokens"]
        }
    
    def generate_metadata_json(self, result: Dict[str, str], output_dir: str):
        """Generate a metadata JSON file for the podcast with all relevant information."""
        metadata = {
            "topic": result["topic"],
            "uuid": result["uuid"],
            "script_path": os.path.basename(result["script_path"]),
            "audio_path": os.path.basename(result["audio_path"]),
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "openai_time": result["openai_time"],
            "file_path": result.get("path", ""),
            "config": {
                "podcast_name": self.config["podcast_name"],
                "hosts": self.config["hosts"],
                "episode_length": self.config["episode_length"],
                "style": self.config["style"]
            },
            "input_tokens": result.get("input_tokens", {}),
            "output_tokens": result.get("output_tokens", {}),
            "overall_time": result.get("overall_time", 0),
            "method": "piper"
        }
        
        # Add length validation info if available
        if "length_validation" in result:
            metadata["length_validation"] = {
                "estimated_minutes": result["length_validation"]["estimated_minutes"],
                "target_minutes": result["length_validation"]["target_minutes"],
                "is_valid": result["length_validation"]["is_valid"],
                "message": result["length_validation"]["message"]
            }
        
        # Save metadata to JSON file
        metadata_path = os.path.join(output_dir, f"{os.path.basename(result['audio_path']).split('.')[0]}_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Generated metadata file: {metadata_path}")
        return metadata_path


# Define CSS for the AI District theme
AI_DISTRICT_CSS = """
:root {
    --primary-color: #A46655;
    --bg-color: #FFF2E6;
    --text-color: #6D4C41;
    --secondary-color: #E7BE9F;
    --accent-color: #D99B84;
}

body {
    background-color: var(--bg-color);
    color: var(--text-color);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

h1, h2, h3 {
    color: var(--primary-color);
}

.gradio-container {
    max-width: 1000px;
    margin: 0 auto;
}

.app-header {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 2rem;
}

.app-logo {
    height: 80px;
    margin-right: 15px;
}

.app-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: var(--primary-color);
}

.gradio-button.primary {
    background-color: var(--primary-color) !important;
}

.gradio-button.secondary {
    background-color: var(--secondary-color) !important;
    color: var(--text-color) !important;
}

.info-box {
    background-color: var(--secondary-color);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

.file-upload {
    border: 2px dashed var(--accent-color);
    padding: 1rem;
    border-radius: 8px;
}
"""

# Define SVG logo for AI District
AI_DISTRICT_LOGO = '''
<svg width="270" height="80" viewBox="0 0 270 80" xmlns="http://www.w3.org/2000/svg">
  <!-- Stylized geometric shape -->
  <polygon points="30,25 40,15 50,25 50,55 30,55" fill="#E7BE9F" />
  <polygon points="35,20 45,10 55,20 55,50 35,50" fill="#D99B84" />
  <!-- Text -->
  <text x="65" y="45" font-family="Arial" font-size="24" font-weight="bold" fill="#A46655">AI DISTRICT</text>
</svg>
'''

# Define default configuration for podcast generation
DEFAULT_CONFIG = {
    "episode_length": "25 minutes",
    "hosts": ["Ryan", "Lessac"],
    "podcast_name": "xIQ sales Xelerator, Deep Dive into Smart Account plan",
    "style": "conversational and informative",
    "audio": {
        # OpenAI TTS settings for OpenAI generator
        "tts_model": "gpt-4o-mini-tts",
        
        # Voice settings for OpenAI
        "voices": {
            "default": {
                "voice": "onyx"  # Default male voice
            },
            "Ryan": {
                "voice": "onyx"  # Male voice
            },
            "Lessac": {
                "voice": "nova"  # Female voice
            }
        },
        
        # Piper settings for Piper generator
        "piper_path": "C:\\piper\\piper.exe",
        "voices_dir": "C:\\piper\\voices",
        
        # Voice settings for Piper
        "voices": {
            "default": {
                "model": "en_US-ryan-medium",
                "speaking_rate": 1.05
            },
            "Ryan": {
                "model": "en_US-ryan-medium",
                "speaking_rate": 1.05
            },
            "Lessac": {
                "model": "en_US-lessac-medium",
                "speaking_rate": 1.05
            }
        },
        
        # Common settings
        "intro_music": None,
        "outro_music": None,
        "add_silence_between_segments": True,
        "silence_duration": [0.4, 0.2, 0.6]
    },
    
    # OpenAI settings for content generation
    "openai_model": "gpt-4o",
    "temperature": 0.7
}

# Function to create a config based on user selections
def create_config(tts_engine, api_key=None, piper_path=None, voices_dir=None):
    """Create a configuration dictionary based on user selections."""
    config = DEFAULT_CONFIG.copy()
    
    if tts_engine == "OpenAI TTS":
        # OpenAI specific configuration
        config["audio"]["voices"] = {
            "default": {"voice": "onyx"},
            "Ryan": {"voice": "onyx"},
            "Lessac": {"voice": "nova"}
        }
    else:  # piper
        # Piper specific configuration
        if piper_path:
            config["audio"]["piper_path"] = piper_path
        else:
            # Use Docker container default path
            config["audio"]["piper_path"] = "/app/piper/piper"
            
        if voices_dir:
            config["audio"]["voices_dir"] = voices_dir
        else:
            # Use Docker container default path
            config["audio"]["voices_dir"] = "/app/piper/voices"
            
        config["audio"]["voices"] = {
            "default": {"model": "en_US-ryan-medium", "speaking_rate": 1.05},
            "Ryan": {"model": "en_US-ryan-medium", "speaking_rate": 1.05},
            "Lessac": {"model": "en_US-lessac-medium", "speaking_rate": 1.05}
        }
    
    return config

# Function to handle podcast generation
def generate_podcast(pdf_file, tts_engine, api_key=None, piper_path=None, voices_dir=None, output_format="mp3"):
    """
    Main function to generate a podcast from a PDF file.
    
    Args:
        pdf_file: Uploaded PDF file
        tts_engine: "OpenAI TTS" or "Piper (Open Source)"
        api_key: OpenAI API key (required for openai)
        piper_path: Path to Piper executable (for piper)
        voices_dir: Path to voice models directory (for piper)
        output_format: Output audio format (mp3, wav, ogg)
        
    Returns:
        Tuple: (status, script_path, audio_path, metadata_path)
    """
    try:
        # Create temporary directory for processing
        output_dir = tempfile.mkdtemp(prefix="podcast_")
        
        # Save uploaded PDF to temporary file
        temp_pdf_path = os.path.join(output_dir, "input.pdf")
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_file)
        
        # Extract text from PDF
        content = extract_text_from_pdf(temp_pdf_path)
        
        # Calculate tokens for OpenAI
        input_tokens = calculate_gpt4o_cost(content, 'input')
        
        # Create configuration
        config = create_config(tts_engine, api_key, piper_path, voices_dir)
        
        # Initialize appropriate generator
        if tts_engine == "OpenAI TTS":
            if not api_key:
                return ("Error: OpenAI API key is required for OpenAI TTS", None, None, None)
            
            # For OpenAI, set the API key explicitly
            generator = OpenAIPodcastGenerator(config, api_key)
        else:  # piper
            # For Piper, set the OpenAI API key for content generation
            os.environ["OPENAI_API_KEY"] = api_key
            
            # If no paths provided, use Docker container defaults
            if not piper_path:
                config["audio"]["piper_path"] = "/app/piper/piper"
            if not voices_dir:
                config["audio"]["voices_dir"] = "/app/piper/voices"
                
            generator = PiperPodcastGenerator(config)
        
        # Generate podcast
        topic = "Smart account plan Analysis"
        
        t1 = time.time()
        result = generator.create_podcast_with_audio(topic, content, output_dir)
        result["path"] = temp_pdf_path
        result["input_tokens"] = input_tokens
        result["overall_time"] = time.time() - t1
        
        # Generate metadata
        metadata_path = generator.generate_metadata_json(result, output_dir)
        
        # Return file paths
        return (
            f"Successfully generated podcast in {result['overall_time']:.2f} seconds",
            result["script_path"],
            result["audio_path"],
            metadata_path
        )
    
    except Exception as e:
        logger.error(f"Error generating podcast: {e}")
        return (f"Error: {str(e)}", None, None, None)

# Gradio interface
def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Define custom HTML for the header with logo
    header_html = f"""
    <div class="app-header">
        {AI_DISTRICT_LOGO}
    </div>
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>AI Podcast Generator</h1>
        <p>Upload a PDF document and generate a professional podcast with AI voices</p>
    </div>
    """
    
    with gr.Blocks(css=AI_DISTRICT_CSS) as app:
        # Header
        gr.HTML(header_html)
        
        # File upload section
        with gr.Row():
            with gr.Column(scale=2):
                pdf_file = gr.File(label="Upload PDF Document", file_types=[".pdf"], type="binary")
                
                with gr.Row():
                    tts_engine = gr.Radio(
                        label="Text-to-Speech Engine", 
                        choices=["OpenAI TTS", "Piper (Open Source)"],
                        value="OpenAI TTS"
                    )
                    output_format = gr.Radio(
                        label="Output Format",
                        choices=["mp3", "wav", "ogg"],
                        value="mp3"
                    )
                
                # Dynamic inputs based on TTS selection
                with gr.Accordion("TTS Settings", open=True) as tts_settings:
                    openai_api_key = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="sk-...",
                        type="password"
                    )
                    
                    with gr.Group(visible=False) as piper_settings:
                        piper_path = gr.Textbox(
                            label="Piper Executable Path",
                            placeholder="/app/piper/piper",
                            value="/app/piper/piper"
                        )
                        voices_dir = gr.Textbox(
                            label="Voice Models Directory",
                            placeholder="/app/piper/voices",
                            value="/app/piper/voices"
                        )
                
                generate_btn = gr.Button("Generate Podcast", variant="primary")
            
            # Results section
            with gr.Column(scale=2):
                status = gr.Textbox(label="Status")
                script = gr.File(label="Podcast Script (Markdown)")
                audio = gr.Audio(label="Podcast Audio", type="filepath")
                metadata = gr.File(label="Podcast Metadata (JSON)")
        
        # Handle TTS engine selection
        def toggle_tts_settings(choice):
            if choice == "Piper (Open Source)":
                return gr.update(visible=True), gr.update(label="OpenAI API Key (for content generation only)")
            else:
                return gr.update(visible=False), gr.update(label="OpenAI API Key")
        
        tts_engine.change(
            toggle_tts_settings,
            inputs=[tts_engine],
            outputs=[piper_settings, openai_api_key]
        )
        
        # Handle podcast generation
        generate_btn.click(
            generate_podcast,
            inputs=[
                pdf_file,
                tts_engine,
                openai_api_key,
                piper_path,
                voices_dir,
                output_format
            ],
            outputs=[status, script, audio, metadata]
        )
    
    return app

# Launch the app when run directly
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch()



