import { useState, useRef, useCallback, useEffect } from 'react';
import { AnalysisReport, ChartDataPoint } from '../types';
import { ISpeechProvider } from './speech-providers/ISpeechProvider';
import { BrowserSpeechProvider } from './speech-providers/BrowserSpeechProvider';
// To switch to ElevenLabs, you would import and instantiate this provider instead.
// import { ElevenLabsApiProvider } from './speech-providers/ElevenLabsApiProvider';

const FILLER_WORD_REGEX = new RegExp(
  '\\b(' +
  'u(m|h)+|a(h|m)+|er+|hmm+|' +
  'like|basically|actually|literally|seriously|right|well|okay|' +
  'you know|sort of|kind of|i mean|i guess|i suppose|at the end of the day' +
  ')\\b', 'gi'
);

const generateSpeechReport = (
  finalTranscript: string, 
  fillerWords: { t: number, word: string }[], 
  startTime: number | null, 
  paceTimeline: ChartDataPoint[],
  topicKeywords: string[]
): Omit<AnalysisReport, 'postureScore'> => {
    const fullText = finalTranscript.trim();
    const words = fullText.split(/\s+/).filter(Boolean);
    const totalWords = words.length;
    const elapsedTimeMinutes = ((performance.now() - (startTime ?? 0)) / 1000) / 60;
    const avgWPM = elapsedTimeMinutes > 0 ? Math.round(totalWords / elapsedTimeMinutes) : 0;

    const phraseCounts = new Map<string, number>();
    if (words.length >= 3) {
        for (let i = 0; i <= words.length - 3; i++) {
            const phrase = words.slice(i, i + 3).join(' ');
            phraseCounts.set(phrase, (phraseCounts.get(phrase) || 0) + 1);
        }
    }
    const repeatedPhrases = Array.from(phraseCounts.entries())
        .filter(([, count]) => count >= 3)
        .map(([phrase, count]) => ({ phrase, count }))
        .sort((a, b) => b.count - a.count);

    const lowerCaseText = fullText.toLowerCase();
    const foundKeywords = topicKeywords.filter(kw => lowerCaseText.includes(kw.toLowerCase()));
    const missingKeywords = topicKeywords.filter(kw => !lowerCaseText.includes(kw.toLowerCase()));
    const topicScore = topicKeywords.length > 0 ? Math.round((foundKeywords.length / topicKeywords.length) * 100) : 100;

    return {
        transcript: [{ t: 0, text: fullText }],
        fullTranscript: fullText,
        fillerWordCount: fillerWords.length,
        fillerWords: fillerWords,
        avgWPM: avgWPM,
        paceTimeline: paceTimeline,
        repeatedPhrases: repeatedPhrases,
        topicKeywords: {
            expected: topicKeywords,
            found: foundKeywords,
            missing: missingKeywords,
            topicScore: topicScore
        }
    };
};

export const useSpeechAnalysis = () => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [liveFillerCount, setLiveFillerCount] = useState(0);
  const [liveWPM, setLiveWPM] = useState(0);

  const providerRef = useRef<ISpeechProvider | null>(null);
  
  const finalTranscriptRef = useRef('');
  const lastWordCountRef = useRef(0);
  const startTimeRef = useRef<number | null>(null);
  const paceIntervalRef = useRef<number | null>(null);
  const fillerWordsRef = useRef<{ t: number, word: string }[]>([]);
  const paceTimelineRef = useRef<ChartDataPoint[]>([]);
  
  const processTranscript = useCallback((result: string, isFinal: boolean) => {
    if (isFinal) {
      finalTranscriptRef.current += result + ' ';
      setTranscript(finalTranscriptRef.current);
      
      const matches = result.toLowerCase().match(FILLER_WORD_REGEX);
      if (matches) {
        const timestamp = performance.now() - (startTimeRef.current ?? 0);
        for (const match of matches) {
          fillerWordsRef.current.push({ t: timestamp / 1000, word: match });
        }
        setLiveFillerCount(prev => prev + matches.length);
      }
    } else {
      setTranscript(finalTranscriptRef.current + result);
    }
  }, []);

  useEffect(() => {
    // Initialize the speech provider on component mount.
    // This establishes the connection to the speech recognition service.
    const provider = new BrowserSpeechProvider({
      onStart: () => setIsListening(true),
      onStop: () => setIsListening(false),
      onResult: processTranscript,
      onError: (error) => console.error(`Speech Provider Error: ${error}`),
    });
    providerRef.current = provider;

    // To use a different provider (like ElevenLabs), you would swap the line above:
    // const provider = new ElevenLabsApiProvider({ ... });

    return () => {
      // Cleanup provider resources on unmount.
      provider.stop();
    };
  }, [processTranscript]);
  
  const startListening = useCallback((topicKeywords: string[]) => {
    if (isListening) return;

    finalTranscriptRef.current = '';
    lastWordCountRef.current = 0;
    fillerWordsRef.current = [];
    paceTimelineRef.current = [];
    setLiveFillerCount(0);
    setLiveWPM(0);
    setTranscript('');
    startTimeRef.current = performance.now();

    providerRef.current?.start();
    
    if (paceIntervalRef.current) {
        clearInterval(paceIntervalRef.current);
    }
    const PACE_INTERVAL_MS = 5000;
    paceIntervalRef.current = window.setInterval(() => {
        if (!startTimeRef.current) return;
        const currentWordCount = finalTranscriptRef.current.trim().split(/\s+/).filter(Boolean).length;
        const wordsInInterval = currentWordCount - lastWordCountRef.current;
        lastWordCountRef.current = currentWordCount;
        const currentWPM = Math.round(wordsInInterval * (60000 / PACE_INTERVAL_MS));
        
        if (isFinite(currentWPM) && currentWPM >= 0) {
            setLiveWPM(currentWPM);
            const totalElapsedTimeSeconds = Math.floor((performance.now() - startTimeRef.current) / 1000);
            const timeLabel = `${totalElapsedTimeSeconds}s`;
            paceTimelineRef.current.push({ time: timeLabel, wpm: currentWPM });
        }
    }, PACE_INTERVAL_MS);

  }, [isListening]);

  const stopListening = useCallback((topicKeywords: string[]): Omit<AnalysisReport, 'postureScore'> => {
    providerRef.current?.stop();
    
    if (paceIntervalRef.current) {
        clearInterval(paceIntervalRef.current);
        paceIntervalRef.current = null;
    }

    return generateSpeechReport(
      finalTranscriptRef.current,
      fillerWordsRef.current,
      startTimeRef.current,
      paceTimelineRef.current,
      topicKeywords
    );
  }, []);

  return {
    isListening,
    transcript,
    liveFillerCount,
    liveWPM,
    startListening,
    stopListening
  };
};
