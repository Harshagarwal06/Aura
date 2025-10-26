import { getPoseLandmarker, getFaceLandmarker } from './visionLoader';
import {
  BodyLanguageSessionAnalyzer,
  LandmarkLike,
  VideoAnalysisReport,
  VoiceMetrics,
  DEFAULT_VOICE_METRICS,
} from './bodyLanguage';
import { analyzeAudioTrack } from './audio';

export interface VideoAnalysisOptions {
  onProgress?: (progress: number) => void;
  includeAudio?: boolean;
}

export interface VideoAnalysisOutcome {
  report: VideoAnalysisReport;
  voice: VoiceMetrics;
}

const MIN_FRAME_GAP_SEC = 1 / 15; // Sample at most ~15 FPS

function extractLandmarksGroup(result: unknown, key: string): LandmarkLike[] | undefined {
  if (!result || typeof result !== 'object') {
    return undefined;
  }

  const record = result as Record<string, unknown>;
  const candidates = record[key];
  if (Array.isArray(candidates) && candidates.length > 0) {
    return candidates[0] as LandmarkLike[];
  }
  return undefined;
}

function getPoseLandmarks(result: unknown): LandmarkLike[] | undefined {
  return (
    extractLandmarksGroup(result, 'landmarks') ??
    extractLandmarksGroup(result, 'poseLandmarks')
  );
}

function getFaceLandmarks(result: unknown): LandmarkLike[] | undefined {
  return (
    extractLandmarksGroup(result, 'faceLandmarks') ??
    extractLandmarksGroup(result, 'landmarks')
  );
}

export async function analyzeVideoFile(file: File, options: VideoAnalysisOptions = {}): Promise<VideoAnalysisOutcome> {
  if (typeof document === 'undefined') {
    throw new Error('Video analysis requires a browser environment.');
  }

  const poseLandmarker = await getPoseLandmarker();
  const faceLandmarker = await getFaceLandmarker();

  const analyzer = new BodyLanguageSessionAnalyzer();
  analyzer.reset();

  const video = document.createElement('video');
  video.src = URL.createObjectURL(file);
  video.muted = true;
  video.playsInline = true;
  video.crossOrigin = 'anonymous';
  video.preload = 'auto';

  await new Promise<void>((resolve, reject) => {
    const handleLoaded = () => {
      video.removeEventListener('loadedmetadata', handleLoaded);
      video.removeEventListener('error', handleError);
      resolve();
    };
    const handleError = () => {
      video.removeEventListener('loadedmetadata', handleLoaded);
      video.removeEventListener('error', handleError);
      reject(new Error('Failed to load video metadata.'));
    };
    video.addEventListener('loadedmetadata', handleLoaded);
    video.addEventListener('error', handleError);
  });

  const duration = video.duration || 0;
  if (!duration || !video.videoWidth || !video.videoHeight) {
    URL.revokeObjectURL(video.src);
    throw new Error('Video metadata unavailable—cannot analyze file.');
  }

  const audioPromise = options.includeAudio === false ? Promise.resolve({ ...DEFAULT_VOICE_METRICS }) : analyzeAudioTrack(file);

  const supportsRVFC = typeof (video as HTMLVideoElement & { requestVideoFrameCallback?: unknown }).requestVideoFrameCallback === 'function';

  await new Promise<void>((resolve) => {
    const width = video.videoWidth;
    const height = video.videoHeight;

    let lastTimestamp = -Infinity;
    let finished = false;

    const process = (timestampSec: number, now: number) => {
      if (timestampSec < lastTimestamp + MIN_FRAME_GAP_SEC) {
        return;
      }
      lastTimestamp = timestampSec;
      try {
        const poseResult = poseLandmarker.detectForVideo(video, now);
        const faceResult = faceLandmarker.detectForVideo(video, now);
        const poseLandmarks = getPoseLandmarks(poseResult);
        const faceLandmarks = getFaceLandmarks(faceResult);
        analyzer.processFrame(poseLandmarks, faceLandmarks, width, height, Math.min(timestampSec, duration));
      } catch (error) {
        console.error('Frame analysis error', error);
      }
    };

    const finalize = () => {
      if (finished) {
        return;
      }
      finished = true;
      options.onProgress?.(1);
      resolve();
    };

    if (supportsRVFC && video.requestVideoFrameCallback) {
      const step = (now: number, metadata: VideoFrameCallbackMetadata) => {
        const timestampSec = metadata.mediaTime ?? video.currentTime;
        process(timestampSec, now);
        options.onProgress?.(Math.min(1, timestampSec / duration));
        if (timestampSec >= duration || video.ended) {
          finalize();
          return;
        }
        video.requestVideoFrameCallback(step);
      };
      video.requestVideoFrameCallback(step);
      void video.play().catch(() => {
        video.currentTime = 0;
        video.requestVideoFrameCallback(step);
      });
    } else {
      // Fallback for browsers without requestVideoFrameCallback
      const stepSeconds = MIN_FRAME_GAP_SEC;
      const seekAndProcess = () => {
        process(video.currentTime, performance.now());
        options.onProgress?.(Math.min(1, video.currentTime / duration));
        if (video.currentTime >= duration) {
          finalize();
          return;
        }
        video.currentTime = Math.min(duration, video.currentTime + stepSeconds);
      };
      video.pause();
      video.currentTime = 0;
      video.addEventListener('seeked', seekAndProcess);
      seekAndProcess();
    }

    video.addEventListener('ended', finalize, { once: true });
  });

  const voice = await audioPromise.catch(() => ({ ...DEFAULT_VOICE_METRICS }));
  const report = analyzer.buildReport(voice);

  URL.revokeObjectURL(video.src);

  return { report, voice };
}
