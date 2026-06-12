"""
Restaurant Voice Receptionist — Multi-Agent System
===================================================
Each incoming call starts at the Greeter. Based on what the caller wants,
the Greeter hands off to a specialist agent:

  Greeter ──► Reservation  (book a table)
          ──► Takeaway ──► Checkout  (place & pay for an order)

Agents share a single UserData object so collected info (name, phone, order,
etc.) is available to every agent without asking the caller twice.
"""

import json
import logging
import os
import re
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated

import yaml
from dotenv import load_dotenv
from pydantic import Field

from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, ModelSettings, RunContext, cli
from livekit.agents.llm import function_tool
from livekit.plugins import openai as livekit_openai
from livekit.plugins import elevenlabs, sarvam, silero

logger = logging.getLogger("restaurant-receptionist")
logger.setLevel(logging.INFO)

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def save_call_data(call_type: str, userdata: "UserData") -> None:
    """Persist call data to a JSON file in the data/ folder.

    Each call gets its own file named: <type>_<YYYYMMDD_HHMMSS>.json
    call_type is either 'reservation' or 'order'.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{call_type}_{timestamp}.json"
    filepath = os.path.join(DATA_DIR, filename)

    record: dict = {"type": call_type, "saved_at": datetime.now().isoformat()}
    if userdata.customer_name:
        record["customer_name"] = userdata.customer_name
    if userdata.customer_phone:
        record["customer_phone"] = userdata.customer_phone
    if userdata.reservation_time:
        record["reservation_time"] = userdata.reservation_time
    if userdata.order:
        record["order"] = userdata.order
    if userdata.expense is not None:
        record["expense"] = userdata.expense

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    logger.info(f"call data saved → {filepath}")


# ElevenLabs voice IDs per agent — different voices signal handoff to the caller.
_VOICES = {
    "greeter":     "cgSgspJ2msm6clMCkdW9",  # Jessica — warm, friendly
    "reservation": "EXAVITQu4vr4xnSDxMaL",  # Sarah   — calm, clear
    "takeaway":    "pNInz6obpgDQGcFmaJgB",  # Adam    — friendly male
    "checkout":    "XB0fDUnXU5powFXDhCwa",  # Charlotte— professional
}


def _elevenlabs_tts(role: str) -> elevenlabs.TTS:
    return elevenlabs.TTS(
        voice_id=_VOICES[role],
        model="eleven_multilingual_v2",
    )

# Groq model used for all agents' LLM inference.
GROQ_MODEL = "llama-3.3-70b-versatile"

# Appended to every agent's instructions.
# Prevents llama3.2 from emitting markdown (Sarvam TTS rejects it) and
# IVR-style "press 1 / press 2" responses (this is a conversational voice agent).
PLAIN_TEXT_RULE = (
    "\n\nIMPORTANT — you are speaking on a live phone call:\n"
    "- Use plain natural speech only. No markdown, no asterisks, no bullet points.\n"
    "- NEVER say things like 'press 1', 'press 2', or 'select an option'. "
    "This is a conversation, not a phone menu.\n"
    "- Keep every response short — one or two sentences maximum.\n"
    "- Ask only one question at a time.\n"
    "- LANGUAGE: Detect the language the caller is speaking and always respond in that same language. "
    "If they speak Telugu, respond in Telugu. If Hindi, respond in Hindi. If English, respond in English."
)




def groq_llm(parallel_tool_calls: bool | None = None) -> livekit_openai.LLM:
    """Return an LLM client pointing at Groq's OpenAI-compatible API.

    Groq's LPU hardware gives sub-100ms time-to-first-token, which is the
    main latency win over running Ollama locally.
    max_completion_tokens=120 keeps responses short and snappy for voice.
    """
    kwargs = {}
    if parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = parallel_tool_calls
    return livekit_openai.LLM(
        model=GROQ_MODEL,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        max_completion_tokens=120,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Shared state — one instance per call, passed through every agent handoff
# ---------------------------------------------------------------------------

@dataclass
class UserData:
    """Holds everything collected from the caller across all agent interactions."""

    customer_name: str | None = None
    customer_phone: str | None = None

    reservation_time: str | None = None

    order: list[str] | None = None

    customer_credit_card: str | None = None
    customer_credit_card_expiry: str | None = None
    customer_credit_card_cvv: str | None = None

    expense: float | None = None
    checked_out: bool | None = None

    # Registry of all agent instances so any agent can look up a target by name.
    agents: dict[str, Agent] = field(default_factory=dict)
    # The agent that was active just before the current one (used to carry over chat history).
    prev_agent: Agent | None = None

    def summarize(self) -> str:
        """Serialize current state for injection into each agent's system prompt.

        Only includes fields that have actually been collected — omitting empty
        fields prevents the model from treating placeholder text like "unknown"
        as a real value. YAML is used because LLMs parse it more reliably than
        JSON inside a longer text block.
        """
        data = {}
        if self.customer_name:
            data["customer_name"] = self.customer_name
        if self.customer_phone:
            data["customer_phone"] = self.customer_phone
        if self.reservation_time:
            data["reservation_time"] = self.reservation_time
        if self.order:
            data["order"] = self.order
        if self.expense is not None:
            data["expense"] = self.expense
        if self.checked_out:
            data["checked_out"] = self.checked_out
        if not data:
            return "Nothing collected yet."
        return yaml.dump(data)


# Convenience type alias so every function signature stays concise.
RunContext_T = RunContext[UserData]


# ---------------------------------------------------------------------------
# Shared tools — available to multiple agents
# ---------------------------------------------------------------------------

@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    context.userdata.customer_name = name
    return f"The name is updated to {name}"


@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="The customer's phone number")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their phone number.
    Confirm the spelling with the user before calling the function."""
    context.userdata.customer_phone = phone
    return f"The phone number is updated to {phone}"


@function_tool()
async def to_greeter(context: RunContext_T) -> Agent:
    """Called when user asks any unrelated questions or requests
    any other services not in your job description."""
    curr_agent: BaseAgent = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context)


# ---------------------------------------------------------------------------
# Base agent — handles handoff logic and context continuity
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove markdown and leaked tool-call JSON before sending text to Sarvam TTS.

    llama3.2 sometimes emits the raw function-call JSON (e.g. {"name":"to_takeaway",...})
    inside the text stream instead of using the proper tool_calls API field.
    Sarvam also errors with 422 on markdown symbols, so both are stripped here.
    Returns an empty string if nothing speakable remains — the tts_node skips
    sending empty strings to avoid a Sarvam 422.
    """
    # Strip chat role prefixes that llama3.2 sometimes leaks into its output.
    text = re.sub(r"^(assistant|user|system)\s*:?\s*", "", text, flags=re.IGNORECASE)

    # Strip leaked tool-call JSON blobs — e.g. {"name":"to_takeaway","parameters":{}}
    # Apply repeatedly to collapse nested braces from the inside out.
    for _ in range(5):
        text, n = re.subn(r"\{[^{}]*\}", "", text)
        if n == 0:
            break

    # Strip markdown formatting
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)   # **bold**, *italic*
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)      # __bold__, _italic_
    text = re.sub(r"`{1,3}.*?`{1,3}", "", text)            # `code` / ```blocks```
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)     # ## headings
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)   # bullet points
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)   # numbered lists
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # [link](url)
    # Remove stray symbols that sit BETWEEN two word characters without adding a space
    # (e.g. p*i*z*z*a → pizza, p.l.a.i.n → plain). Without this, the replace below
    # would turn them into "p i z z a" which Sarvam reads letter by letter.
    text = re.sub(r"(?<=\w)[^\w\s,.'!?;:()\-](?=\w)", "", text)
    # Replace any remaining stray symbols with a single space
    text = re.sub(r"[^\w\s,.'!?;:()\-]", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


class BaseAgent(Agent):
    async def on_enter(self) -> None:
        """Called every time this agent becomes the active one.

        Two things happen here:
        1. The last 6 messages from the previous agent are copied into this
           agent's context so the conversation feels continuous to the caller.
        2. A fresh system message is injected with the latest UserData so the
           agent always has up-to-date info (name, order, etc.) without asking again.
        After setup, the agent immediately generates an opening reply.
        """
        agent_name = self.__class__.__name__
        logger.info(f"entering agent: {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # Carry over recent conversation history from the previous agent.
        if isinstance(userdata.prev_agent, Agent):
            truncated_chat_ctx = userdata.prev_agent.chat_ctx.copy(
                exclude_instructions=True,
                exclude_function_call=False,
                exclude_handoff=True,
                exclude_config_update=True,
            ).truncate(max_items=6)
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in truncated_chat_ctx.items if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # Inject current call state so the agent knows what has already been collected.
        # We do NOT repeat the role here — the agent's own instructions already define that.
        chat_ctx.add_message(
            role="system",
            content=f"Current call state:\n{userdata.summarize()}",
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Buffer full response, strip markdown, synthesise with ElevenLabs.

        Buffering the entire response before synthesis eliminates inter-sentence
        pauses. eleven_multilingual_v2 handles English, Hindi, and other
        languages without a separate TTS instance per language.
        """
        full_text = ""
        async for chunk in text:
            full_text += chunk

        cleaned = _strip_markdown(full_text.strip())
        if not cleaned:
            return

        async for audio_event in self._tts.synthesize(cleaned):
            yield audio_event.frame

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> tuple[Agent, str]:
        """Switch the active agent to `name` and record the current one as prev
        so the next agent can inherit conversation history via on_enter."""
        userdata = context.userdata
        next_agent = userdata.agents[name]
        userdata.prev_agent = context.session.current_agent
        return next_agent, f"Transferring to {name}."


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

class Greeter(BaseAgent):
    """Entry point for every call. Greets the caller and routes them to the
    right agent based on whether they want a reservation or a takeaway order."""

    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                "You are Luna, a warm receptionist at Spice Garden restaurant.\n"
                f"Menu: {menu}\n\n"
                "Greet the caller warmly by saying your name is Luna, then ask what they need.\n"
                "- If they want a table reservation → call to_reservation.\n"
                "- If they want to order food → call to_takeaway.\n"
                "- If they ask about the menu, tell them briefly what is available.\n"
                "- Never transfer without first understanding what the caller wants."
                + PLAIN_TEXT_RULE
            ),
            # parallel_tool_calls=False prevents the greeter from triggering
            # two transfers at once if the model gets confused.
            llm=groq_llm(parallel_tool_calls=False),
            tts=_elevenlabs_tts("greeter"),
        )
        self.menu = menu

    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when user wants to make or update a reservation.
        This function handles transitioning to the reservation agent
        who will collect the necessary details like reservation time,
        customer name and phone number."""
        return await self._transfer_to_agent("reservation", context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to place a takeaway order.
        This includes handling orders for pickup, delivery, or when the user wants to
        proceed to checkout with their existing order."""
        return await self._transfer_to_agent("takeaway", context)


class Reservation(BaseAgent):
    """Collects reservation details: time, name, and phone number.
    Confirms everything with the caller before finalising."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Ritu, the reservations agent at Spice Garden restaurant.\n\n"
                "Collect details in this exact order — ask ONE thing at a time:\n"
                "Step 1: Ask what DATE and TIME they want the reservation.\n"
                "Step 2: Ask for their FULL NAME. Then call update_name.\n"
                "Step 3: Ask for their PHONE NUMBER. Then call update_phone.\n"
                "Step 4: Read all three details back to confirm, then call confirm_reservation.\n\n"
                "Rules:\n"
                "- Never skip a step or ask for multiple things at once.\n"
                "- Never make up or assume any detail.\n"
                "- If the caller asks something outside reservations, call to_greeter."
                + PLAIN_TEXT_RULE
            ),
            tools=[update_name, update_phone, to_greeter],
            tts=_elevenlabs_tts("reservation"),
        )

    @function_tool()
    async def update_reservation_time(
        self,
        time: Annotated[str, Field(description="The reservation time")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their reservation time.
        Confirm the time with the user before calling the function."""
        context.userdata.reservation_time = time
        return f"The reservation time is updated to {time}"

    @function_tool()
    async def confirm_reservation(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the reservation.
        Validates that name, phone, and time are all collected before accepting."""
        userdata = context.userdata
        if not userdata.customer_name or not userdata.customer_phone:
            return "Please provide your name and phone number first."
        if not userdata.reservation_time:
            return "Please provide reservation time first."
        save_call_data("reservation", userdata)
        return await self._transfer_to_agent("greeter", context)


class Takeaway(BaseAgent):
    """Takes and manages the caller's food order.
    Hands off to Checkout once the order is confirmed."""

    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                "You are Rahul, the takeaway agent at Spice Garden restaurant.\n"
                f"Menu: {menu}\n\n"
                "Your job is to take the customer's food order. Follow these steps:\n"
                "Step 1: Ask what they would like to order from the menu.\n"
                "Step 2: Clarify any special requests or quantities.\n"
                "Step 3: Read the full order back to confirm it is correct.\n"
                "Step 4: Call update_order with all confirmed items.\n"
                "Step 5: Ask if they are ready to proceed to checkout. If yes, call to_checkout.\n\n"
                "Rules:\n"
                "- Only offer items that are on the menu above. Never invent items.\n"
                "- Never ask for name, phone, or payment — that is handled by other agents.\n"
                "- If the caller asks something unrelated, call to_greeter."
                + PLAIN_TEXT_RULE
            ),
            tools=[to_greeter],
            tts=_elevenlabs_tts("takeaway"),
        )

    @function_tool()
    async def update_order(
        self,
        items: Annotated[list[str], Field(description="The items of the full order")],
        context: RunContext_T,
    ) -> str:
        """Called when the user create or update their order."""
        context.userdata.order = items
        return f"The order is updated to {items}"

    @function_tool()
    async def to_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the order."""
        if not context.userdata.order:
            return "No takeaway order found. Please make an order first."
        return await self._transfer_to_agent("checkout", context)


class Checkout(BaseAgent):
    """Handles payment: confirms the bill amount then collects card details
    (number, expiry, CVV) step by step before completing the order."""

    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                "You are Kavya, the checkout agent at Spice Garden restaurant.\n\n"
                "Collect payment details in this exact order — ask ONE thing at a time:\n"
                "Step 1: Tell the caller the total amount for their order and ask them to confirm. "
                "Then call confirm_expense.\n"
                "Step 2: Ask for their FULL NAME. Then call update_name.\n"
                "Step 3: Ask for their PHONE NUMBER. Then call update_phone.\n"
                "Step 4: Ask for their CREDIT CARD NUMBER. Then their EXPIRY DATE. Then their CVV. "
                "Once you have all three, call update_credit_card.\n"
                "Step 5: Read back the order and total to confirm, then call confirm_checkout.\n\n"
                "Rules:\n"
                "- Never skip a step or ask for multiple things at once.\n"
                "- Never store or repeat card details back in full — only confirm the last 4 digits.\n"
                "- If the caller wants to change their order, call to_takeaway.\n"
                "- If the caller asks something unrelated, call to_greeter."
                + PLAIN_TEXT_RULE
            ),
            tools=[update_name, update_phone, to_greeter],
            tts=_elevenlabs_tts("checkout"),
        )

    @function_tool()
    async def confirm_expense(
        self,
        expense: Annotated[float, Field(description="The expense of the order")],
        context: RunContext_T,
    ) -> str:
        """Called when the user confirms the expense."""
        context.userdata.expense = expense
        return f"The expense is confirmed to be {expense}"

    @function_tool()
    async def update_credit_card(
        self,
        number: Annotated[str, Field(description="The credit card number")],
        expiry: Annotated[str, Field(description="The expiry date of the credit card")],
        cvv: Annotated[str, Field(description="The CVV of the credit card")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their credit card number, expiry date, and CVV.
        Confirm the spelling with the user before calling the function."""
        userdata = context.userdata
        userdata.customer_credit_card = number
        userdata.customer_credit_card_expiry = expiry
        userdata.customer_credit_card_cvv = cvv
        return f"The credit card number is updated to {number}"

    @function_tool()
    async def confirm_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the checkout.
        Validates expense and full card details before marking the order complete."""
        userdata = context.userdata
        if not userdata.expense:
            return "Please confirm the expense first."
        if (
            not userdata.customer_credit_card
            or not userdata.customer_credit_card_expiry
            or not userdata.customer_credit_card_cvv
        ):
            return "Please provide the credit card information first."
        userdata.checked_out = True
        save_call_data("order", userdata)
        return await to_greeter(context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to update their order."""
        return await self._transfer_to_agent("takeaway", context)


# ---------------------------------------------------------------------------
# Session entry point
# ---------------------------------------------------------------------------

_vad: silero.VAD | None = None

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Bootstraps a new call session.

    Creates all four agents upfront and stores them in UserData so any agent
    can look up and transfer to another by name. The session is wired with:
      - STT: Sarvam saarika:v2.5 (streaming, via WebSocket)
      - LLM: Groq llama-3.3-70b-versatile (cloud, OpenAI-compatible endpoint)
      - TTS: Sarvam bulbul:v3 (streaming, per-agent voice)
      - VAD: Silero (local voice activity detection)
    """
    global _vad
    if _vad is None:
        _vad = silero.VAD.load()
    menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"
    userdata = UserData()
    userdata.agents.update(
        {
            "greeter": Greeter(menu),
            "reservation": Reservation(),
            "takeaway": Takeaway(menu),
            "checkout": Checkout(menu),
        }
    )
    session = AgentSession[UserData](
        userdata=userdata,
        stt=sarvam.STT(language="unknown", model="saarika:v2.5"),
        llm=groq_llm(),
        tts=_elevenlabs_tts("greeter"),
        vad=_vad,
        # Caps the number of consecutive tool calls per turn to prevent loops.
        max_tool_steps=5,
    )

    await session.start(
        agent=userdata.agents["greeter"],
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
