import { useRef, useCallback } from 'react';

// A "delight" layer to make the app feel more like a personal coach.
export const useAuraVoice = () => {
  const hasSpokenRef = useRef(false);

  const speak = useCallback((text: string, { onEnd, force = false }: { onEnd?: () => void; force?: boolean } = {}) => {
    if ('speechSynthesis' in window && (!hasSpokenRef.current || force)) {
      window.speechSynthesis.cancel(); // Stop any previous speech
      const utterance = new SpeechSynthesisUtterance(text);
      
      // Find a premium, calm voice. "Aura" is a good keyword.
      const voices = speechSynthesis.getVoices();
      utterance.voice = voices.find(v => v.name.includes("Aura") || v.name.includes("Google US English") || v.name.includes("Samantha")) || voices[0];
      
      utterance.pitch = 1.1;
      utterance.rate = 1;
      utterance.volume = 0.8;
      
      if (onEnd) {
          utterance.onend = onEnd;
      }

      window.speechSynthesis.speak(utterance);

      if (!force) {
        hasSpokenRef.current = true;
      }
    }
  }, []);
  
  // Reset the spoken flag, e.g., for a new page or session.
  const resetSpokenFlag = useCallback(() => {
    hasSpokenRef.current = false;
  }, []);

  return { speak, resetSpokenFlag };
};
