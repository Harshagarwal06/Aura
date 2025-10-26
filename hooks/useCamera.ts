import { useState, useEffect } from 'react';

export const useCamera = () => {
  const [mediaStream, setMediaStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;

    const getMedia = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true,
        });
        setMediaStream(stream);
      } catch (err) {
        if (err instanceof Error) {
            if (err.name === 'NotAllowedError') {
                setError('Camera and microphone access was denied. Please allow access in your browser settings.');
            } else {
                 setError(`Error accessing media devices: ${err.message}`);
            }
        } else {
            setError('An unknown error occurred while accessing media devices.');
        }
      }
    };

    getMedia();

    // Cleanup function to stop all tracks when the component unmounts
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return { mediaStream, error };
};
