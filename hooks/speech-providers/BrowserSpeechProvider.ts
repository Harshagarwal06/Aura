import { ISpeechProvider, SpeechProviderCallbacks } from "./ISpeechProvider";

interface SpeechRecognition {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: () => void;
  onresult: (event: any) => void;
  onerror: (event: any) => void;
  onend: () => void;
  start: () => void;
  stop: () => void;
  abort: () => void;
}

export class BrowserSpeechProvider implements ISpeechProvider {
  private recognition: SpeechRecognition | null = null;
  private callbacks: SpeechProviderCallbacks;
  private listening = false;

  constructor(callbacks: SpeechProviderCallbacks) {
    this.callbacks = callbacks;
    const SpeechRecognitionAPI = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognitionAPI) {
      this.recognition = new SpeechRecognitionAPI();
      this.configureRecognition();
    } else {
      this.callbacks.onError("Speech Recognition API is not supported in this browser.");
    }
  }

  private configureRecognition() {
    if (!this.recognition) return;

    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.lang = 'en-US';

    this.recognition.onstart = () => {
      this.listening = true;
      this.callbacks.onStart();
    };

    this.recognition.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      if (finalTranscript) {
        this.callbacks.onResult(finalTranscript, true);
      }
      if (interimTranscript) {
        this.callbacks.onResult(interimTranscript, false);
      }
    };

    this.recognition.onerror = (event) => {
      this.callbacks.onError(event.error);
    };

    this.recognition.onend = () => {
      this.listening = false;
      this.callbacks.onStop();
      // Some browsers stop recognition after a period of silence.
      // This could be re-enabled for auto-restart if needed.
    };
  }

  public start(): void {
    if (this.recognition && !this.listening) {
      try {
        this.recognition.start();
      } catch (e) {
        console.error("Error starting speech recognition:", e);
        this.callbacks.onError("Could not start speech recognition.");
      }
    }
  }

  public stop(): void {
    if (this.recognition && this.listening) {
      this.recognition.stop();
    }
  }

  public isListening(): boolean {
    return this.listening;
  }
}
