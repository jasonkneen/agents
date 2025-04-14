import asyncio
import os
import logging
import edge_tts
from voice_core.tts_utils import markdown_to_text
import uuid

logger = logging.getLogger(__name__)

class EdgeTTSTTS:
    def __init__(
        self,
        voice_id: str = "en-US-AvaMultilingualNeural",
        output_format: str = "audio-16khz-128kbitrate-mono-mp3",
        rate: str = "+0%",
        volume: str = "+0%",  # Changed from +0dB to +0%
        chunk_dir: str = "temp_chunks"
    ):
        self.voice_id = voice_id
        self.output_format = output_format
        self.rate = rate
        self.volume = volume
        self.chunk_dir = chunk_dir
        self.initialized = False
        self.stt_service = None  # Will be set externally
        
        # Create chunks directory if it doesn't exist
        if not os.path.exists(chunk_dir):
            os.makedirs(chunk_dir)

    async def initialize(self):
        """Initialize the TTS service"""
        try:
            # Test voice availability by creating a small test communication
            test_communicate = edge_tts.Communicate(
                text="Test",
                voice=self.voice_id,
                rate=self.rate,
                volume=self.volume
            )
            # Verify we can create the stream
            async for _ in test_communicate.stream():
                break
            self.initialized = True
            logger.info(f"[EdgeTTSTTS] Initialized successfully with voice {self.voice_id}")
        except Exception as e:
            logger.error(f"[EdgeTTSTTS] Failed to initialize: {e}")
            raise

    async def process_text(self, text: str) -> str:
        """Process text to speech and return audio file path"""
        logger.info(f"[EdgeTTSTTS] Processing text with voice '{self.voice_id}'...")
        
        # Clean up old chunks
        self._cleanup_old_chunks()
        
        # Generate unique session ID for this synthesis
        session_id = str(uuid.uuid4())

        # Convert markdown to plain text if needed
        clean_text = markdown_to_text(text or "")
        if not clean_text.strip():
            logger.warning("[EdgeTTSTTS] Received empty text after Markdown conversion.")
            return ""

        try:
            # Create communicate object
            communicate = edge_tts.Communicate(
                text=clean_text,
                voice=self.voice_id,
                rate=self.rate,
                volume=self.volume
            )

            # Generate audio data
            audio_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])

            # Save as audio file
            if audio_data:
                audio_path = os.path.join(
                    self.chunk_dir,
                    f"{session_id}.mp3"
                )
                with open(audio_path, "wb") as f:
                    f.write(audio_data)
                    
                # Schedule STT reset after TTS processing
                asyncio.create_task(self._reset_stt())
                return audio_path

            logger.warning("[EdgeTTSTTS] No audio data generated")
            return ""

        except Exception as e:
            logger.error(f"[EdgeTTSTTS] Error processing text: {e}")
            return ""

    async def _reset_stt(self):
        """Reset STT service after TTS playback"""
        try:
            # Small delay to allow TTS to finish playing
            await asyncio.sleep(1)
            
            if self.stt_service and hasattr(self.stt_service, 'cleanup'):
                # Reset STT by cleaning up and reinitializing
                await self.stt_service.cleanup()
                await self.stt_service.initialize()
                logger.info("[EdgeTTSTTS] Successfully reset STT service after TTS playback")
            else:
                logger.warning("[EdgeTTSTTS] No STT service available to reset")
                
        except Exception as e:
            logger.error(f"[EdgeTTSTTS] Error resetting STT service: {e}")

    def _cleanup_old_chunks(self):
        """Clean up old audio chunks"""
        for chunk_path in os.listdir(self.chunk_dir):
            try:
                os.remove(os.path.join(self.chunk_dir, chunk_path))
            except Exception as e:
                logger.warning(f"[EdgeTTSTTS] Failed to remove chunk {chunk_path}: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        try:
            self._cleanup_old_chunks()
            self.initialized = False
            logger.info("[EdgeTTSTTS] Cleanup completed")
        except Exception as e:
            logger.error(f"[EdgeTTSTTS] Error during cleanup: {e}")
