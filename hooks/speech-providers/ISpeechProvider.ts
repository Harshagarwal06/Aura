/**
 * Callbacks that a speech provider will use to communicate with the main hook.
 */
export interface SpeechProviderCallbacks {
  onStart: () => void;
  onStop: () => void;
  onResult: (transcript: string, isFinal: boolean) => void;
  onError: (error: string) => void;
}

/**
 * Defines the contract for a speech recognition provider.
 * Any service (browser's SpeechRecognition, ElevenLabs, etc.) must implement this.
 */
export interface ISpeechProvider {
  /**
   * Starts the speech recognition process.
   */
  start(): void;
  
  /**
   * Stops the speech recognition process.
   */
  stop(): void;

  /**
   * Returns true if the provider is currently listening for audio.
   */
  isListening(): boolean;
}
