import os
import chainlit as cl
import logging
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MessageRole

# Load environment variables
load_dotenv()

# Disable verbose connection logs
logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger.setLevel(logging.WARNING)

# ── Env vars ──────────────────────────────────────────────────────────────────
# Example: https://<your-project>.<region>.inference.ai.azure.com
AIPROJECT_ENDPOINT = os.getenv("AIPROJECT_ENDPOINT", "").rstrip("/")
AGENT_ID = os.getenv("AGENT_ID")

if not AIPROJECT_ENDPOINT:
    raise RuntimeError("AIPROJECT_ENDPOINT is not set in your environment (.env).")
if not AGENT_ID:
    raise RuntimeError("AGENT_ID is not set in your environment (.env).")

# ── Auth & client ─────────────────────────────────────────────────────────────
# DefaultAzureCredential will try Managed Identity, then env vars, then others.
credential = DefaultAzureCredential()
project_client = AIProjectClient(endpoint=AIPROJECT_ENDPOINT, credential=credential)

# ── Chainlit ──────────────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    # Create a thread for the agent
    if not cl.user_session.get("thread_id"):
        thread = project_client.agents.create_thread()
        cl.user_session.set("thread_id", thread.id)
        print(f"New Thread ID: {thread.id}")

@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")

    try:
        # Show thinking message to user
        msg = await cl.Message("thinking...", author="agent").send()

        project_client.agents.create_message(
            thread_id=thread_id,
            role="user",
            content=message.content,
        )

        # Run the agent to process the message in the thread
        run = project_client.agents.create_and_process_run(
            thread_id=thread_id,
            agent_id=AGENT_ID
        )
        print(f"Run finished with status: {run.status}")

        if run.status == "failed":
            # Surface service-side error details if present
            raise Exception(getattr(run, "last_error", "Run failed."))

        # Get all messages from the thread
        messages = project_client.agents.list_messages(thread_id)

        # Get the last message from the agent
        last_msg = messages.get_last_text_message_by_role(MessageRole.AGENT)
        if not last_msg:
            raise Exception("No response from the model.")

        msg.content = last_msg.text.value
        await msg.update()

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

if __name__ == "__main__":
    # Chainlit will automatically run the application
    pass
