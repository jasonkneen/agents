import asyncio
import os
import math
import cv2

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero

# Video dimensions
WIDTH = 1280
HEIGHT = 720

# Load environment variables from .env file
load_dotenv()


# Preload resources to improve initialization performance
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# Main entry point for the voice assistant
async def entrypoint(ctx: JobContext):
    # Set up initial chat context
    initial_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content="You are a voice assistant. Pretend we're having a human conversation, no special formatting or headings, just natural speech.",
            )
        ]
    )

    # Configure the voice assistant with VAD, STT, LLM, and TTS
    assistant = VoiceAssistant(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        llm=openai.LLM(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        ),
        tts=openai.TTS(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            voice="nova",
        ),
        chat_ctx=initial_context,
    )

    camera = None
    camera_task = None
    video_track = None
    try:
        # Create video source and track for camera
        video_source = rtc.VideoSource(WIDTH, HEIGHT)
        video_track = rtc.LocalVideoTrack.create_video_track(
            "camera-feed", video_source
        )
        video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)

        # Try to find BRIO camera
        camera = None
        for i in [2, 1, 0]:  # Try common indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret:
                    # Get camera info
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"Found camera at index {i}: {width}x{height}")
                    camera = cap
                    break
                cap.release()

        if camera is None:
            raise RuntimeError("Could not find any camera")

        # Configure camera settings
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, 30)

        # Create camera capture task
        async def capture_camera():
            try:
                while True:
                    ret, frame = camera.read()
                    if ret:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Create frame data
                        frame_bytes = frame_rgb.tobytes()
                        # Create VideoFrame
                        frame_data = rtc.VideoFrame(
                            data=frame_bytes,
                            width=WIDTH,
                            height=HEIGHT,
                            format=2,  # FORMAT_RGB24
                        )
                        # Capture the frame
                        video_source.capture_frame(frame_data)
                    await asyncio.sleep(1 / 30)  # 30 fps
            except Exception as e:
                print(f"Camera capture error: {e}")
                raise

        # Start camera capture
        camera_task = asyncio.create_task(capture_camera())

        # Connect to the room
        await ctx.connect()

        # Start the voice assistant
        assistant.start(ctx.room)

        # Publish video track
        video_publication = await ctx.room.local_participant.publish_track(
            video_track, video_options
        )
        print(f"Published video track with SID: {video_publication.sid}")

        await asyncio.sleep(1)  # Allow resources to initialize
        await assistant.say(
            "Hi there, how are you doing today?", allow_interruptions=True
        )

        # Keep the connection alive
        while True:
            await asyncio.sleep(1)

    except Exception as e:
        print(f"Error in assistant: {e}")
        if video_track:
            await video_track.stop()
        raise e
    finally:
        if camera_task:
            camera_task.cancel()
            try:
                await camera_task
            except asyncio.CancelledError:
                pass
        if camera:
            camera.release()


# Application entry point
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_name="shimmer",
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
