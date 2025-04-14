import asyncio
import os

from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero

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
            model="gpt-4o-mini"
        ),
        tts=openai.TTS(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            voice="nova"
        ),
        chat_ctx=initial_context,
    )

    # Connect the assistant and start the interaction
    try:
        await ctx.connect()  # Establish connection with the room
        assistant.start(ctx.room)  # Start the voice assistant in the room
        await asyncio.sleep(1)  # Allow resources to initializel
        await assistant.say("Hi there, how are you doing today?", allow_interruptions=True)
    except Exception as e:
        print(f"Error starting the assistant: {e}")

# Application entry point
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_name="nova",
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )