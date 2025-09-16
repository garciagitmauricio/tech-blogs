import os
import chainlit as cl
import logging
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import MessageRole  # <-- moved here

# Load environment variables
load_dotenv()

# Disable verbose connection logs
logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger.setLevel(logging.WARNING)

# ── Env vars ──────────────────────────────────────────────────────────────────
# Example: https://<your-project>.<region>.inference.ai.azure.com
AIPROJECT_CONNECTION_STRING = os.getenv("AIPROJECT_CONNECTION_STRING", "").rstrip("/")
AGENT_ID = os.getenv("AGENT_ID")

if not AIPROJECT_CONNECTION_STRING:
    raise RuntimeError("AIPROJECT_CONNECTION_STRING is not set in your environment (.env).")
if not AGENT_ID:
    raise RuntimeError("AGENT_ID is not set in your environment (.env).")

# ── Auth & client ─────────────────────────────────────────────────────────────
# DefaultAzureCredential will try Managed Identity, then env vars, then others.
credential = DefaultAzureCredential()
project_client = AIProjectClient(endpoint=AIPROJECT_CONNECTION_STRING, credential=credential)

def _message_text(msg) -> str | None:
    """Extract text robustly from an agent message."""
    try:
        parts = getattr(msg, "content", None)
        if parts:
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "text":
                    return p["text"].get("value")
                pt = getattr(p, "text", None)
                if pt and hasattr(pt, "value"):
                    return pt.value
    except Exception:
        pass
    txt = getattr(msg, "text", None)
    return getattr(txt, "value", None) if txt else None

# ── Chainlit ──────────────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    # Create a thread for the agent (new API)
    if not cl.user_session.get("thread_id"):
        thread = project_client.agents.threads.create()
        cl.user_session.set("thread_id", thread.id)
        print(f"New Thread ID: {thread.id}")

@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")

    try:
        # Show thinking message to user
        msg = await cl.Message("thinking...", author="agent").send()

        # Add user message (new API)
        project_client.agents.messages.create(
            thread_id=thread_id,
            role="user",
            content=message.content,
        )

        # Run the agent to process the message in the thread (new API)
        run = project_client.agents.runs.create_and_process(
            thread_id=thread_id,
            agent_id=AGENT_ID
        )
        print(f"Run finished with status: {run.status}")

        if getattr(run, "status", None) == "failed":
            # Surface service-side error details if present
            raise Exception(getattr(run, "last_error", "Run failed."))

        # Get the last message from the agent (preferred direct call)
        try:
            last_msg = project_client.agents.messages.get_last_message_by_role(
                thread_id=thread_id, role=MessageRole.AGENT
            )
            text = _message_text(last_msg)
        except Exception:
            # Fallback: list all and pick last assistant/agent
            msgs = list(project_client.agents.messages.list(thread_id=thread_id))
            text = None
            for m in reversed(msgs):
                if getattr(m, "role", "").lower() in ("assistant", "agent"):
                    text = _message_text(m)
                    if text:
                        break

        if not text:
            raise Exception("No response from the model.")

        msg.content = text
        await msg.update()

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

if __name__ == "__main__":
    # Chainlit will automatically run the application
    pass

