import { useState, useRef, useCallback, RefObject } from 'react';

// Simplified analysis constants
const CENTER_THRESHOLD = 0.1; // 10% deviation from center allowed
const MOVEMENT_THRESHOLD = 2; // Pixel difference threshold for movement detection
const GOOD_POSTURE_SCORE_INCREMENT = 0.1;
const BAD_POSTURE_SCORE_DECREMENT = 0.2;

export const usePostureAnalysis = (videoRef: RefObject<HTMLVideoElement>) => {
  const [postureFeedback, setPostureFeedback] = useState('Not tracking');
  const [postureScore, setPostureScore] = useState(85); // Start with a decent score
  
  const analysisFrameRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastCenterRef = useRef<{ x: number, y: number } | null>(null);

  const analyzeFrame = useCallback(() => {
    if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) {
      analysisFrameRef.current = requestAnimationFrame(analyzeFrame);
      return;
    }
    
    if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas');
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    if (!ctx) {
        analysisFrameRef.current = requestAnimationFrame(analyzeFrame);
        return;
    }
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    if (canvas.width === 0 || canvas.height === 0) {
        analysisFrameRef.current = requestAnimationFrame(analyzeFrame);
        return;
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // This is a very simplified simulation of posture analysis.
    // A real implementation would use a model like PoseNet.
    // Here, we find the "center of mass" of non-background pixels.
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    let totalX = 0, totalY = 0, count = 0;

    // Sample a subset of pixels for performance
    for (let i = 0; i < data.length; i += 4 * 20) { // Sample every 20th pixel
        // A simple skin-tone detection heuristic (very rough)
        const r = data[i];
        const g = data[i+1];
        const b = data[i+2];
        if (r > 95 && g > 40 && b > 20 && r > g && r > b && Math.abs(r - g) > 15) {
            const x = (i / 4) % canvas.width;
            const y = Math.floor((i / 4) / canvas.width);
            totalX += x;
            totalY += y;
            count++;
        }
    }

    if (count > 100) { // Only analyze if we have enough points
        const centerX = totalX / count;
        const normalizedCenterX = centerX / canvas.width;
        let feedback = "Good";
        let isGoodPosture = true;

        if (normalizedCenterX < 0.5 - CENTER_THRESHOLD) {
            feedback = "Leaning Left";
            isGoodPosture = false;
        } else if (normalizedCenterX > 0.5 + CENTER_THRESHOLD) {
            feedback = "Leaning Right";
            isGoodPosture = false;
        }

        if (lastCenterRef.current) {
            const movement = Math.abs(centerX - lastCenterRef.current.x);
            if (movement > MOVEMENT_THRESHOLD * (canvas.width/100) ) { // Movement relative to width
                feedback = "Swaying";
                isGoodPosture = false;
            }
        }
        
        setPostureFeedback(feedback);
        lastCenterRef.current = { x: centerX, y: totalY/count };
        
        // Update score
        setPostureScore(prev => Math.min(100, Math.max(0, prev + (isGoodPosture ? GOOD_POSTURE_SCORE_INCREMENT : -BAD_POSTURE_SCORE_DECREMENT))));
    }


    analysisFrameRef.current = requestAnimationFrame(analyzeFrame);
  }, [videoRef]);

  const startPostureAnalysis = useCallback(() => {
    if (analysisFrameRef.current) {
        cancelAnimationFrame(analysisFrameRef.current);
    }
    setPostureFeedback('Good');
    analysisFrameRef.current = requestAnimationFrame(analyzeFrame);
  }, [analyzeFrame]);

  const stopPostureAnalysis = useCallback(() => {
    if (analysisFrameRef.current) {
      cancelAnimationFrame(analysisFrameRef.current);
      analysisFrameRef.current = null;
    }
    setPostureFeedback('Not tracking');
  }, []);

  const getPostureReport = useCallback(() => {
      return {
          postureScore: Math.round(postureScore)
      }
  }, [postureScore]);

  return { postureFeedback, startPostureAnalysis, stopPostureAnalysis, getPostureReport };
};
