import React, { useEffect, RefObject } from 'react';
import { useCamera } from '../hooks/useCamera';
import { VideoCameraSlashIcon } from '@heroicons/react/24/solid';

interface VideoFeedProps {
  videoRef: RefObject<HTMLVideoElement>;
}

const VideoFeed: React.FC<VideoFeedProps> = ({ videoRef }) => {
  const { mediaStream, error } = useCamera();

  useEffect(() => {
    if (videoRef.current && mediaStream) {
      videoRef.current.srcObject = mediaStream;
    }
  }, [mediaStream, videoRef]);

  if (error) {
    return (
      <div role="alert" className="w-full h-full flex flex-col items-center justify-center text-center text-rose-400 p-4">
        <VideoCameraSlashIcon className="h-12 w-12 mb-2" />
        <p className="font-semibold">Access Denied</p>
        <p className="text-sm text-slate-400 max-w-xs">{error}</p>
      </div>
    );
  }

  if (!mediaStream) {
     return (
        <div className="w-full h-full flex flex-col items-center justify-center text-center text-slate-500">
            <svg className="animate-spin h-8 w-8 text-indigo-400 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p className="font-semibold">Activating Camera...</p>
            <p className="text-sm">Please allow camera and microphone access.</p>
        </div>
     );
  }

  return (
    <video
      ref={videoRef}
      autoPlay
      playsInline
      muted
      className="w-full h-full object-cover rounded-xl"
    />
  );
};

export default VideoFeed;
