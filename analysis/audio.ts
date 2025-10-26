import { DEFAULT_VOICE_METRICS, VoiceMetrics } from './bodyLanguage';

const QUIET_THRESHOLD = 0.015;
const WINDOW_SECONDS = 0.5;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export async function analyzeAudioTrack(file: File): Promise<VoiceMetrics> {
  if (typeof window === 'undefined' || typeof window.AudioContext === 'undefined') {
    return { ...DEFAULT_VOICE_METRICS };
  }

  const AudioCtor = (window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext) as
    | typeof AudioContext
    | undefined;
  if (!AudioCtor) {
    return { ...DEFAULT_VOICE_METRICS };
  }

  const audioContext = new AudioCtor();
  try {
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const { length, numberOfChannels, sampleRate } = audioBuffer;
    const duration = audioBuffer.duration || (length / sampleRate);
    if (!length || !duration) {
      return { ...DEFAULT_VOICE_METRICS };
    }

    const merged = new Float32Array(length);
    for (let channel = 0; channel < numberOfChannels; channel += 1) {
      const channelData = audioBuffer.getChannelData(channel);
      for (let i = 0; i < length; i += 1) {
        merged[i] += channelData[i] / numberOfChannels;
      }
    }

    let sumSquares = 0;
    for (let i = 0; i < merged.length; i += 1) {
      const sample = merged[i];
      sumSquares += sample * sample;
    }
    const rms = Math.sqrt(sumSquares / merged.length);

    const windowSize = Math.max(1, Math.floor(sampleRate * WINDOW_SECONDS));
    let speakingWindows = 0;
    let quietStart = true;
    let firstWindowChecked = false;
    for (let offset = 0; offset < merged.length; offset += windowSize) {
      let windowSum = 0;
      const end = Math.min(merged.length, offset + windowSize);
      for (let i = offset; i < end; i += 1) {
        windowSum += Math.abs(merged[i]);
      }
      const meanAbs = windowSum / (end - offset);
      if (meanAbs > QUIET_THRESHOLD) {
        speakingWindows += 1;
        if (!firstWindowChecked) {
          quietStart = false;
        }
      }
      if (!firstWindowChecked) {
        firstWindowChecked = true;
        if (meanAbs > QUIET_THRESHOLD) {
          quietStart = false;
        }
      }
    }

    const speakingTime = speakingWindows * WINDOW_SECONDS;
    const speakingRatio = duration > 0 ? clamp(speakingTime / duration, 0, 1) : 0;
    const energyLevel = clamp(rms / 0.12, 0, 1);

    const paceWpm = Math.round(clamp(140 + (speakingRatio - 0.5) * 120, 90, 210));
    const voiceScoreBase = 55 + energyLevel * 25 + speakingRatio * 20 - (quietStart ? 5 : 0);
    const voiceScore = Math.round(clamp(voiceScoreBase, 0, 100));

    let paceFeedback: string;
    if (paceWpm > 180) {
      paceFeedback = 'Trim the pace slightly so your ideas have time to land.';
    } else if (paceWpm < 120) {
      paceFeedback = 'Pick up the pace to keep the audience leaning in.';
    } else {
      paceFeedback = 'Nice conversational pacing—easy to follow.';
    }

    let energyFeedback: string;
    if (energyLevel > 0.75) {
      energyFeedback = 'Powerful projection that fills the room.';
    } else if (energyLevel < 0.35) {
      energyFeedback = 'Boost your volume to sound more confident.';
    } else {
      energyFeedback = 'Steady volume that feels natural.';
    }

    const comment = `${energyFeedback} ${paceFeedback}`.trim();

    return {
      voiceScore,
      paceWpm,
      fillerPerMin: 0,
      comment,
      quietStart,
      energy: energyLevel,
      speakingRatio,
    };
  } catch (error) {
    console.error('Audio analysis failed', error);
    return { ...DEFAULT_VOICE_METRICS };
  } finally {
    try {
      await audioContext.close();
    } catch (error) {
      // Ignore close errors
    }
  }
}
