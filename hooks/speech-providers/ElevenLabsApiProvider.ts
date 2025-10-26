import { ISpeechProvider, SpeechProviderCallbacks } from "./ISpeechProvider";

// This is a placeholder implementation to demonstrate how you would integrate
// a third-party speech-to-text API like ElevenLabs.
// This approach requires a backend service to securely handle your API key.

export class ElevenLabsApiProvider implements ISpeechProvider {
  private callbacks: SpeechProviderCallbacks;
  private listening = false;
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private mediaStream: MediaStream | null = null;
  private websocket: WebSocket | null = null;

  constructor(callbacks: SpeechProviderCallbacks) {
    this.callbacks = callbacks;
  }

  public async start(): Promise<void> {
    if (this.listening) return;

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
      
      this.processor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        // The audio is captured as 32-bit float. We need to convert it to 16-bit PCM
        // for most speech-to-text APIs.
        const pcmData = this.floatTo16BitPCM(inputData);
        this.sendAudioToBackend(pcmData);
      };

      source.connect(this.processor);
      this.processor.connect(this.audioContext.destination);

      this.connectToBackend();

      this.listening = true;
      this.callbacks.onStart();

    } catch (error) {
      console.error("Error starting audio capture:", error);
      this.callbacks.onError("Microphone access was denied or an error occurred.");
    }
  }

  public stop(): void {
    if (!this.listening) return;

    // 1. Disconnect from WebSocket
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }

    // 2. Stop audio processing
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    
    // 3. Stop microphone tracks
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    
    // 4. Close audio context
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    
    this.listening = false;
    this.callbacks.onStop();
  }

  public isListening(): boolean {
    return this.listening;
  }

  private connectToBackend() {
    // ========================================================================
    // TODO: IMPLEMENT YOUR BACKEND CONNECTION HERE
    // ========================================================================
    // This is where you would establish a WebSocket connection to your secure
    // backend service.
    // e.g., this.websocket = new WebSocket('wss://your-backend.com/speech');

    // MOCKUP: Simulating a WebSocket connection for demonstration.
    // Replace this with your actual WebSocket implementation.
    console.log("Attempting to connect to backend WebSocket...");
    this.websocket = {
      send: (data: ArrayBuffer) => {
        // console.log(`Sending ${data.byteLength} bytes of audio data...`);
      },
      close: () => {
         console.log("WebSocket connection closed.");
      },
      // You would add onmessage, onopen, onerror, onclose handlers here
    } as any;
    
    // Handle incoming messages from the backend (the transcripts)
    // this.websocket.onmessage = (event) => {
    //   const { transcript, isFinal } = JSON.parse(event.data);
    //   this.callbacks.onResult(transcript, isFinal);
    // };
  }

  private sendAudioToBackend(audioData: Int16Array) {
    if (this.websocket) {
      // ========================================================================
      // TODO: SEND AUDIO DATA TO YOUR BACKEND
      // ========================================================================
      // Your backend would then forward this data to the ElevenLabs API.
      this.websocket.send(audioData.buffer as ArrayBuffer);
    }
  }
  
  private floatTo16BitPCM(input: Float32Array): Int16Array {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
        const s = Math.max(-1, Math.min(1, input[i]));
        output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return output;
  }
}
