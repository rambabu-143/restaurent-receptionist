---
title: Spice Garden Receptionist
emoji: 🍛
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 8080
pinned: true
---

# Restaurant Receptionist

A voice AI receptionist for **Spice Garden Restaurant** built on [LiveKit Agents](https://docs.livekit.io/agents/). Handles table reservations and takeaway orders over a live phone/web call.

## Features

- **Table reservations** — collects date, time, name, and phone number
- **Takeaway orders** — takes food orders from the menu, handles checkout
- **Indian-accented voice** — Sarvam AI STT and TTS tuned for Indian English
- **Local LLM** — runs on Ollama (no OpenAI dependency)
- **Multi-agent flow** — Greeter → Reservation or Takeaway → Checkout

## Stack

| Component | Technology |
|-----------|------------|
| Framework | LiveKit Agents |
| LLM | Ollama `llama3.2` (local) |
| STT | Sarvam AI `saarika:v2.5` |
| TTS | Sarvam AI `bulbul:v3` |
| VAD | Silero (local) |

## Setup

### 1. Install dependencies

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.ai/).

```bash
uv sync
ollama pull llama3.2
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
SARVAM_API_KEY=your_sarvam_api_key
```

Get your keys from:
- LiveKit Cloud: [cloud.livekit.io](https://cloud.livekit.io)
- Sarvam AI: [sarvam.ai](https://sarvam.ai)

### 3. Run

```bash
uv run python main.py dev
```

Open the LiveKit Agents console to connect a call.

## Agent Flow

```
Caller
  └── Greeter (Priya)
        ├── wants reservation → Reservation agent (Ritu)
        │                           collects: date/time, name, phone
        │                           saves to: data/reservation_*.json
        └── wants food order  → Takeaway agent (Rahul)
                                    collects: order items
                                    └── Checkout agent (Kavya)
                                            collects: name, phone, card details
                                            saves to: data/order_*.json
```

## Project Structure

```
main.py          # All agent logic
pyproject.toml   # Dependencies
.env.example     # Environment variable template
data/            # Saved call records (gitignored)
```
