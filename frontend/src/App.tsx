import { useState, useCallback } from 'react'
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useVoiceAssistant,
  BarVisualizer,
  VoiceAssistantControlBar,
} from '@livekit/components-react'
import '@livekit/components-styles'
import './App.css'

interface TokenData {
  token: string
  url: string
  room: string
}

function ActiveCall({ onEnd }: { onEnd: () => void }) {
  const { state, audioTrack } = useVoiceAssistant()

  const stateLabel: Record<string, string> = {
    disconnected: 'Connecting…',
    connecting: 'Connecting…',
    initializing: 'Starting agent…',
    listening: 'Listening…',
    thinking: 'Thinking…',
    speaking: 'Speaking…',
  }

  return (
    <div className="call-view">
      <div className="restaurant-name">Spice Garden</div>
      <p className="agent-state">{stateLabel[state] ?? state}</p>

      <div className="visualizer">
        <BarVisualizer
          state={state}
          trackRef={audioTrack}
          barCount={24}
          options={{ minHeight: 4 }}
        />
      </div>

      <VoiceAssistantControlBar />

      <button className="end-btn" onClick={onEnd}>
        End Call
      </button>
    </div>
  )
}

function App() {
  const [tokenData, setTokenData] = useState<TokenData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const startCall = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/token')
      if (!res.ok) throw new Error(`Token request failed: ${res.status}`)
      setTokenData(await res.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start call')
    } finally {
      setLoading(false)
    }
  }, [])

  const endCall = useCallback(() => setTokenData(null), [])

  if (tokenData) {
    return (
      <LiveKitRoom
        token={tokenData.token}
        serverUrl={tokenData.url}
        connect={true}
        audio={true}
        video={false}
        onDisconnected={endCall}
      >
        <RoomAudioRenderer />
        <ActiveCall onEnd={endCall} />
      </LiveKitRoom>
    )
  }

  return (
    <div className="home">
      <div className="logo">🍛</div>
      <h1>Spice Garden</h1>
      <p className="tagline">Talk to our AI receptionist to book a table or place a takeaway order.</p>

      {error && <p className="error">{error}</p>}

      <button className="call-btn" onClick={startCall} disabled={loading}>
        {loading ? 'Connecting…' : '📞 Call Now'}
      </button>

      <p className="hint">Make sure your microphone is allowed.</p>
    </div>
  )
}

export default App
