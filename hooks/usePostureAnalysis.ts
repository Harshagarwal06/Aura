import { useState, useRef, useCallback, RefObject, useEffect } from 'react';
import { BodyLanguageSessionAnalyzer, createFeedbackMessage, LandmarkLike, DEFAULT_VOICE_METRICS } from '../analysis/bodyLanguage';
import { getPoseLandmarker, getFaceLandmarker } from '../analysis/visionLoader';

const MIN_READY_STATE = HTMLMediaElement.HAVE_CURRENT_DATA;

export const usePostureAnalysis = (videoRef: RefObject<HTMLVideoElement>) => {
  const [postureFeedback, setPostureFeedback] = useState('Not tracking');
  const analyzerRef = useRef<BodyLanguageSessionAnalyzer>(new BodyLanguageSessionAnalyzer());
  const animationFrameRef = useRef<number | null>(null);
  const startTimestampRef = useRef<number | null>(null);
  const lastReportRef = useRef<ReturnType<BodyLanguageSessionAnalyzer['buildReport']> | null>(null);
  const poseLandmarkerRef = useRef<Awaited<ReturnType<typeof getPoseLandmarker>> | null>(null);
  const faceLandmarkerRef = useRef<Awaited<ReturnType<typeof getFaceLandmarker>> | null>(null);
  const loadingModelsRef = useRef(false);

  const ensureModelsLoaded = useCallback(async () => {
    if (poseLandmarkerRef.current && faceLandmarkerRef.current) {
      return;
    }
    if (loadingModelsRef.current) {
      return;
    }
    loadingModelsRef.current = true;
    try {
      const [pose, face] = await Promise.all([getPoseLandmarker(), getFaceLandmarker()]);
      poseLandmarkerRef.current = pose;
      faceLandmarkerRef.current = face;
    } catch (error) {
      console.error('Failed to load vision models', error);
      setPostureFeedback('Camera analysis unavailable. Please refresh and try again.');
    } finally {
      loadingModelsRef.current = false;
    }
  }, []);

  const analyzeFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || !poseLandmarkerRef.current || !faceLandmarkerRef.current) {
      animationFrameRef.current = requestAnimationFrame(analyzeFrame);
      return;
    }

    if (video.readyState < MIN_READY_STATE || video.videoWidth === 0 || video.videoHeight === 0) {
      animationFrameRef.current = requestAnimationFrame(analyzeFrame);
      return;
    }

    const now = performance.now();
    if (startTimestampRef.current === null) {
      startTimestampRef.current = now;
    }
    const timestampSec = (now - startTimestampRef.current) / 1000;

    try {
      const poseResult = poseLandmarkerRef.current.detectForVideo(video, now);
      const faceResult = faceLandmarkerRef.current.detectForVideo(video, now);
      const poseLandmarks = (poseResult?.landmarks?.[0] as LandmarkLike[] | undefined) ?? undefined;
      const faceLandmarks = (faceResult?.faceLandmarks?.[0] as LandmarkLike[] | undefined) ?? undefined;
      const summary = analyzerRef.current.processFrame(
        poseLandmarks,
        faceLandmarks,
        video.videoWidth,
        video.videoHeight,
        timestampSec,
      );
      setPostureFeedback(createFeedbackMessage(summary));
    } catch (error) {
      console.error('Live posture analysis failed', error);
    }

    animationFrameRef.current = requestAnimationFrame(analyzeFrame);
  }, [videoRef]);

  const startPostureAnalysis = useCallback(async () => {
    await ensureModelsLoaded();
    analyzerRef.current.reset();
    lastReportRef.current = null;
    startTimestampRef.current = null;
    setPostureFeedback('Tracking posture...');
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    animationFrameRef.current = requestAnimationFrame(analyzeFrame);
  }, [analyzeFrame, ensureModelsLoaded]);

  const stopPostureAnalysis = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    setPostureFeedback('Not tracking');
  }, []);

  const getPostureReport = useCallback(() => {
    if (!lastReportRef.current) {
      lastReportRef.current = analyzerRef.current.buildReport(DEFAULT_VOICE_METRICS);
    }
    return {
      postureScore: lastReportRef.current.skills.posture.score,
    };
  }, []);

  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return { postureFeedback, startPostureAnalysis, stopPostureAnalysis, getPostureReport };
};
