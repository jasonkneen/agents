from __future__ import annotations

import os
import uuid
import logging
import asyncio
import time
from typing import Optional, Any
from datetime import datetime
import sys
from logger_config import setup_logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, AutoSubscribe
from voice_core.config.config import LucidiaConfig, LLMConfig
from voice_core.lo cal_stt_service import LocalSTTService
from voice_core.edge_tts_plugin import EdgeTTSTTS
from voice_core.llm_communication import LocalLLM
from voice_core.voice_pipeline_agent import LocalVoicePipelineAgent
from voice_core.llm_pipeline import LocalLLMPipeline
from async_timeout import timeout

load_dotenv()

LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')
LIVEKIT_URL = os.getenv('LIVEKIT_URL', 'ws://localhost:7880')

if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
    raise ValueError("Missing required environment variables: LIVEKIT_API_KEY and LIVEKIT_API_SECRET")

logger = setup_logging(level="INFO")

# --------------------------------------------------------------------------------
# Audio Utilities
# --------------------------------------------------------------------------------
class AudioFrame:
    """
    Wrapper for audio data that provides the frame interface expected by livekit.
    Assumes 16-bit mono audio at 24kHz by default.
    """

    def __init__(self, data: bytes):
        # Warn if data length is not aligned properly for 16-bit audio frames
        if len(data) % 2 != 0:
            logging.warning("Audio data length is not aligned with 16-bit frames")

        from livekit.rtc import AudioFrame as LiveKitAudioFrame
        # Get audio settings from environment
        sample_rate = int(os.getenv('LIVEKIT_SAMPLE_RATE', '48000'))
        num_channels = int(os.getenv('LIVEKIT_CHANNELS', '2'))
        
        self._lk_frame = LiveKitAudioFrame(
            data=data,
            samples_per_channel=len(data) // 2,
            sample_rate=sample_rate,
            num_channels=num_channels
        )
        # Store these for reference
        self.data = data
        self.samples_per_channel = len(data) // 2
        self.sample_rate = sample_rate
        self.num_channels = num_channels

    def __bytes__(self) -> bytes:
        return self.data

    def frame(self) -> Any:
        """
        Return the underlying LiveKit audio frame object.
        """
        return self._lk_frame


# --------------------------------------------------------------------------------
# Cleanup Helpers
# --------------------------------------------------------------------------------
async def cleanup_connection(assistant: Optional['CustomVoiceAssistant'], ctx: JobContext) -> None:
    """
    Gracefully cleanup the connection and resources with enhanced state management.
    """
    logger.info("Starting connection cleanup...")
    try:
        # Step 1: Cleanup assistant first to stop all services
        if assistant:
            try:
                async with async_timeout.timeout(10.0):
                    await assistant.cleanup()
                logger.info("Assistant cleanup completed")
            except asyncio.TimeoutError:
                logger.error("Timeout during assistant cleanup")
            except Exception as e:
                logger.error(f"Error during assistant cleanup: {e}")

        # Step 2: Handle room cleanup
        if ctx and hasattr(ctx, 'room') and ctx.room:
            try:
                # Log current state
                state = {
                    'ws_connected': bool(ctx.room._ws and ctx.room._ws.connected) if hasattr(ctx.room, '_ws') else False,
                    'participants': len(ctx.room._participants) if hasattr(ctx.room, '_participants') else 0,
                    'connection_state': ctx.room.connection_state if hasattr(ctx.room, 'connection_state') else None
                }
                logger.info(f"Room state before cleanup: {state}")

                # Close WebSocket first
                if hasattr(ctx.room, '_ws') and ctx.room._ws:
                    try:
                        await ctx.room._ws.close()
                        await asyncio.sleep(1)  # Brief wait for WebSocket closure
                    except Exception as e:
                        logger.error(f"Error closing WebSocket: {e}")

                # Force disconnect with timeout
                if ctx.room.connection_state != 0:  # 0 = Disconnected
                    try:
                        async with async_timeout.timeout(5.0):
                            await ctx.room.disconnect()
                            # Wait for full disconnection
                            while ctx.room.connection_state != 0:
                                await asyncio.sleep(0.1)
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for room disconnect")
                    except Exception as e:
                        logger.error(f"Error during room disconnect: {e}")

                # Clear room state
                if hasattr(ctx.room, '_participants'):
                    ctx.room._participants.clear()
                if hasattr(ctx.room, '_remote_tracks'):
                    ctx.room._remote_tracks.clear()
                
                # Final state check
                final_state = {
                    'ws_connected': bool(ctx.room._ws and ctx.room._ws.connected) if hasattr(ctx.room, '_ws') else False,
                    'participants': len(ctx.room._participants) if hasattr(ctx.room, '_participants') else 0,
                    'connection_state': ctx.room.connection_state if hasattr(ctx.room, 'connection_state') else None
                }
                logger.info(f"Room state after cleanup: {final_state}")

            except Exception as e:
                logger.error(f"Error during room cleanup: {e}")

    except Exception as e:
        logger.error(f"Error during connection cleanup: {e}")
    finally:
        # Extended wait for server cleanup
        await asyncio.sleep(2)
        logger.info("Connection cleanup completed")

async def force_room_cleanup(ctx: JobContext) -> None:
    """Force cleanup of room state with extended wait times"""
    try:
        logger.info("Starting forced room cleanup...")
        
        if hasattr(ctx, 'room') and ctx.room:
            try:
                # Log current state
                if hasattr(ctx.room, '_participants'):
                    participants = list(ctx.room._participants.values())
                    logger.info(f"Current participants in room: {[p.identity for p in participants]}")
                
                # Close WebSocket first
                if hasattr(ctx.room, '_ws') and ctx.room._ws:
                    logger.info("Closing existing WebSocket connection...")
                    try:
                        await ctx.room._ws.close()
                        await asyncio.sleep(2)  # Wait for WebSocket to close
                    except Exception as e:
                        logger.warning(f"Error closing WebSocket: {e}")
                
                # Force disconnect and wait
                logger.info("Forcing room disconnect...")
                await ctx.room.disconnect()
                await wait_for_disconnect(ctx, timeout=15)  # Extended wait
                
            except Exception as e:
                logger.warning(f"Error during graceful cleanup: {e}")
        
        # Clear mutable state
        if hasattr(ctx, 'room'):
            try:
                if hasattr(ctx.room, '_participants'):
                    ctx.room._participants.clear()
                if hasattr(ctx.room, '_ws'):
                    ctx.room._ws = None
                if hasattr(ctx.room, '_remote_tracks'):
                    ctx.room._remote_tracks.clear()
                
            except Exception as e:
                logger.warning(f"Error clearing room state: {e}")
        
        # Clear any cached token
        if hasattr(ctx, '_access_token'):
            delattr(ctx, '_access_token')
        
        # Extended wait for server cleanup
        await asyncio.sleep(5)
        logger.info("Room cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during room cleanup: {e}")
        raise


async def wait_for_disconnect(ctx, timeout: int = 15) -> bool:
    """
    Wait for room to fully disconnect with extended timeout.
    Returns True if disconnect confirmed, False if timeout.
    """
    logger.info(f"Waiting up to {timeout} seconds for room disconnect...")
    start_time = asyncio.get_event_loop().time()
    
    while asyncio.get_event_loop().time() - start_time < timeout:
        if not hasattr(ctx, 'room') or not ctx.room or ctx.room.connection_state == 0:
            logger.info("Room disconnect confirmed")
            return True
        await asyncio.sleep(1)
        
    logger.warning(f"Room disconnect timeout after {timeout} seconds")
    return False


async def verify_room_state(ctx: JobContext) -> bool:
    """
    Verify that the room is truly clean and ready for a new connection
    """
    try:
        if not hasattr(ctx, 'room'):
            return True
            
        # Check WebSocket state
        if hasattr(ctx.room, '_ws') and ctx.room._ws and ctx.room._ws.connected:
            logger.warning("Room still has connected WebSocket")
            return False
            
        # Check participant state
        if hasattr(ctx.room, '_participants') and ctx.room._participants:
            logger.warning("Room still has participants")
            return False
            
        # Check connection state
        if ctx.room.connection_state != 0:  # 0 = Disconnected
            logger.warning("Room not in disconnected state")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error verifying room state: {e}")
        return False

def prewarm(proc: JobContext) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_prewarm_async(proc))
    finally:
        loop.close()
        asyncio.set_event_loop(None)

async def _prewarm_async(proc: JobContext) -> None:
    logger.info("Prewarming resources...")
    try:
        config = LucidiaConfig()
        stt_service = LocalSTTService(config)
        await stt_service.initialize()
        logger.info("Successfully prewarmed STT model")
    except Exception as e:
        logger.error(f"Error prewarming resources: {e}")
        raise

async def cleanup_connection(assistant: Optional['CustomVoiceAssistant'], ctx: JobContext):
    try:
        if assistant:
            await assistant.cleanup()
        if ctx.room:
            await ctx.room.disconnect()
    except Exception as e:
        logger.error(f"Error during connection cleanup: {e}")

class CustomVoiceAssistant:
    """Custom voice assistant with Edge TTS streaming and enhanced LiveKit management."""
    def __init__(self):
        self.config = LucidiaConfig()
        self.llm_config = LLMConfig()
        self.room = None
        self.stt_service = None
        self.tts_service = None
        self.audio_source = None
        self.audio_track = None
        self.pipeline_agent = None
        self._shutdown = False
        self._cleanup_lock = asyncio.Lock()
        self._cleanup_event = asyncio.Event()
        self._track_tasks = {}
        logger.info(f"Voice assistant initialized with config: {self.config}")

    async def start(self, room: rtc.Room) -> None:
        """Start the voice assistant with enhanced LiveKit room integration."""
        try:
            logger.info("Starting voice assistant...")
            self.room = room
            self._shutdown = False

            # Initialize audio source and track
            self.audio_source = rtc.AudioSource(48000, 1)
            self.audio_track = rtc.LocalAudioTrack.create_audio_track("agent_voice", self.audio_source)
            
            # Wait for room connection to stabilize
            try:
                async with async_timeout.timeout(5.0):
                    while not self.room or self.room.connection_state != 2:  # 2 = Connected
                        await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for room connection")
                raise

            # Publish track with error handling
            try:
                await self.room.local_participant.publish_track(self.audio_track)
                logger.info("Audio track published successfully")
            except Exception as e:
                logger.error(f"Error publishing audio track: {e}")
                raise

            # Initialize services
            logger.info("Initializing services...")
            logger.info("Initializing STT service...")
            self.stt_service = LocalSTTService(self.config)
            await self.stt_service.initialize()
            logger.info("STT service initialized")
            
            logger.info("Initializing TTS service...")
            self.tts_service = EdgeTTSTTS()
            logger.info("TTS service initialized")
            
            # Initialize pipeline agent
            self.pipeline_agent = LocalVoicePipelineAgent(
                llm_pipeline=LocalLLMPipeline(config=self.llm_config),
                tts_service=self.tts_service,
                stt_service=self.stt_service,
                room=self.room,
                audio_source=self.audio_source
            )
            
            # Set transcript callback to pipeline agent
            self.stt_service.set_transcript_callback(self.pipeline_agent._handle_transcript)
            
            logger.info("Sending initial greeting")
            await self.pipeline_agent.send_greeting()
            
            logger.info("Voice assistant started successfully")

        except Exception as e:
            logger.error(f"Error starting voice assistant: {e}")
            await self.cleanup()
            raise

    async def cleanup(self):
        """Enhanced cleanup with proper resource management."""
        async with self._cleanup_lock:
            if self._cleanup_event.is_set():
                logger.info("Cleanup already in progress")
                return
            
            self._cleanup_event.set()
            try:
                logger.info("Starting voice assistant cleanup...")
                self._shutdown = True

                # Step 1: Stop TTS stream
                if self.tts_service:
                    try:
                        async with async_timeout.timeout(3.0):
                            await self.tts_service.cleanup()
                    except asyncio.TimeoutError:
                        logger.error("Timeout stopping TTS stream")
                    except Exception as e:
                        logger.error(f"Error stopping TTS stream: {e}")

                # Step 2: Stop audio track
                if self.audio_track:
                    try:
                        async with async_timeout.timeout(3.0):
                            await self.audio_track.stop()
                            # Wait for track to fully stop
                            while self.audio_track.state != "ended":
                                await asyncio.sleep(0.1)
                    except asyncio.TimeoutError:
                        logger.error("Timeout stopping audio track")
                    except Exception as e:
                        logger.error(f"Error stopping audio track: {e}")

                # Step 3: Cleanup track tasks
                for task_name, task in self._track_tasks.items():
                    if not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=2.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            logger.warning(f"Task {task_name} cancelled")
                self._track_tasks.clear()

                # Step 4: Clear room resources
                if self.room and hasattr(self.room, 'local_participant'):
                    try:
                        # Unpublish track if still published
                        if self.audio_track in self.room.local_participant.tracks.values():
                            await self.room.local_participant.unpublish_track(self.audio_track)
                    except Exception as e:
                        logger.error(f"Error unpublishing track: {e}")

                # Step 5: Clear references
                self.audio_source = None
                self.audio_track = None
                self.room = None

                logger.info("Voice assistant cleanup completed")

            except Exception as e:
                logger.error(f"Error during voice assistant cleanup: {e}")
                raise
            finally:
                self._shutdown = True
                self._cleanup_event.clear()

    async def speak(self, text: str) -> None:
        """Forward speak requests to pipeline agent"""
        if self.pipeline_agent:
            await self.pipeline_agent._handle_transcript(text)
        else:
            logger.error("Pipeline agent not initialized")

    def stop(self):
        """Stop the assistant and initiate cleanup."""
        if not self._shutdown:
            self._shutdown = True
            asyncio.create_task(self.cleanup())

async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint with enhanced connection lifecycle management."""
    start_time = time.time()
    logger.info("Starting voice assistant...")
    agent = None
    
    try:
        # Step 1: Verify and cleanup existing room state
        if hasattr(ctx, 'room') and ctx.room:
            state = {
                'ws_connected': bool(ctx.room._ws and ctx.room._ws.connected) if hasattr(ctx.room, '_ws') else False,
                'participants': len(ctx.room._participants) if hasattr(ctx.room, '_participants') else 0,
                'connection_state': ctx.room.connection_state if hasattr(ctx.room, 'connection_state') else None
            }
            if state['ws_connected'] or state['participants'] > 0 or (state['connection_state'] is not None and state['connection_state'] != 0):
                logger.warning(f"Room not clean before connect: {state}")
                await cleanup_connection(None, ctx)
                await asyncio.sleep(2)  # Wait for cleanup to settle
        
        # Step 2: Connect with retry logic
        max_retries = 3
        retry_delay = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Connection attempt {attempt + 1}/{max_retries}")
                
                # Connect with timeout
                async with async_timeout.timeout(10.0):
                    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
                    # Wait for connection to stabilize
                    while not ctx.room or ctx.room.connection_state != 2:  # 2 = Connected
                        await asyncio.sleep(0.1)
                
                logger.info("Connected to LiveKit room")
                break
                
            except asyncio.TimeoutError:
                last_error = "Connection timeout"
                logger.error(f"Connection attempt {attempt + 1} timed out")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to connect after {max_retries} attempts. Last error: {last_error}")
        
        # Step 3: Initialize and start voice assistant
        try:
            agent = CustomVoiceAssistant()
            await agent.start(ctx.room)
            
            elapsed = time.time() - start_time
            if elapsed > 2:
                logger.warning(f"Startup took {elapsed:.2f}s")
            else:
                logger.info(f"Startup completed in {elapsed:.2f}s")
            
            # Keep the connection alive
            while True:
                if not ctx.room or ctx.room.connection_state != 2:
                    raise RuntimeError("Room connection lost")
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in voice assistant: {e}")
            if agent:
                await agent.cleanup()
            raise
            
    except Exception as e:
        logger.error(f"Fatal error in entrypoint: {e}")
        if agent:
            await agent.cleanup()
        # Ensure room is cleaned up
        await cleanup_connection(agent, ctx)
        raise

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_name="lucid-agent-2",
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm
        )
    )