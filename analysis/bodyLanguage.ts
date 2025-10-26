const POSTURE_TILT_BAD_DEG = 20.0;
const FACING_DEPTH_DIFF_BAD = 0.2;
const FIDGET_MOTION_CALM = 0.02;
const FIDGET_MOTION_FIDGET = 0.06;
const GESTURE_MIN_HEIGHT_FRAC = 0.6;
const SMILE_MOUTH_RATIO_SMILE = 0.35;
const OPENING_WINDOW_SEC = 5.0;
const MIN_LANDMARK_VISIBILITY = 0.5;

export interface LandmarkLike {
  x: number;
  y: number;
  z: number;
  visibility?: number;
  presence?: number;
}

interface XYPoint {
  x: number;
  y: number;
}

const DEFAULT_VOICE_COMMENT = 'Good projection. Moderate filler words.';

export interface VoiceMetrics {
  voiceScore: number;
  paceWpm: number;
  fillerPerMin: number;
  comment: string;
  quietStart: boolean;
  energy?: number;
  speakingRatio?: number;
}

const DEFAULT_VOICE_METRICS: VoiceMetrics = {
  voiceScore: 70,
  paceWpm: 170,
  fillerPerMin: 8,
  comment: DEFAULT_VOICE_COMMENT,
  quietStart: false,
  energy: undefined,
  speakingRatio: undefined,
};

function dist2d(a: XYPoint, b: XYPoint): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function getXY(landmarks: LandmarkLike[], idx: number, w: number, h: number): XYPoint {
  const lm = landmarks[idx];
  return { x: lm.x * w, y: lm.y * h };
}

function isVisible(landmarks: LandmarkLike[], idx: number, threshold = MIN_LANDMARK_VISIBILITY): boolean {
  const lm = landmarks[idx];
  const visibility = lm.visibility ?? lm.presence ?? 0;
  return visibility > threshold;
}

function torsoUprightAngleDeg(lms: LandmarkLike[], w: number, h: number): number | null {
  if (![11, 12, 23, 24].every((i) => isVisible(lms, i))) {
    return null;
  }
  const lShoulder = getXY(lms, 11, w, h);
  const rShoulder = getXY(lms, 12, w, h);
  const lHip = getXY(lms, 23, w, h);
  const rHip = getXY(lms, 24, w, h);
  const midShoulder = { x: (lShoulder.x + rShoulder.x) / 2, y: (lShoulder.y + rShoulder.y) / 2 };
  const midHip = { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };
  const dx = midShoulder.x - midHip.x;
  const dy = midShoulder.y - midHip.y;
  const angleFromVertical = Math.atan2(Math.abs(dx), Math.abs(dy) + 1e-6) * (180 / Math.PI);
  return angleFromVertical;
}

function facingDepthDiff(lms: LandmarkLike[]): number | null {
  if (![11, 12].every((i) => isVisible(lms, i))) {
    return null;
  }
  const lZ = lms[11].z;
  const rZ = lms[12].z;
  return Math.abs(lZ - rZ);
}

function bodyScale(lms: LandmarkLike[], w: number, h: number): number | null {
  if (![11, 12].every((i) => isVisible(lms, i))) {
    return null;
  }
  const ls = getXY(lms, 11, w, h);
  const rs = getXY(lms, 12, w, h);
  return dist2d(ls, rs) + 1e-6;
}

interface HandsInfo {
  lWr: XYPoint;
  rWr: XYPoint;
  lHeightFrac: number;
  rHeightFrac: number;
  wristsDist: number;
  torsoHeight: number;
}

function handsInfo(lms: LandmarkLike[], w: number, h: number): HandsInfo | null {
  if (![11, 12, 15, 16, 23, 24].every((i) => isVisible(lms, i))) {
    return null;
  }

  const lWr = getXY(lms, 15, w, h);
  const rWr = getXY(lms, 16, w, h);
  const lSh = getXY(lms, 11, w, h);
  const rSh = getXY(lms, 12, w, h);
  const midChest = { x: (lSh.x + rSh.x) / 2, y: (lSh.y + rSh.y) / 2 };
  const lHp = getXY(lms, 23, w, h);
  const rHp = getXY(lms, 24, w, h);
  const midHip = { x: (lHp.x + rHp.x) / 2, y: (lHp.y + rHp.y) / 2 };
  const torsoHeight = Math.abs(midHip.y - midChest.y) + 1e-6;

  const lHeightFrac = 1 - (lWr.y - midChest.y) / torsoHeight;
  const rHeightFrac = 1 - (rWr.y - midChest.y) / torsoHeight;
  const wristsDist = dist2d(lWr, rWr);

  return { lWr, rWr, lHeightFrac, rHeightFrac, wristsDist, torsoHeight };
}

function smileProxyFromFace(faceLandmarks: LandmarkLike[], w: number, h: number): { smiling: boolean; ratio: number } {
  const xy = (idx: number): XYPoint => ({ x: faceLandmarks[idx].x * w, y: faceLandmarks[idx].y * h });
  const leftCorner = xy(78);
  const rightCorner = xy(308);
  const topLip = xy(13);
  const bottomLip = xy(14);
  const mouthWidth = dist2d(leftCorner, rightCorner) + 1e-6;
  const mouthOpen = dist2d(topLip, bottomLip);
  const ratio = mouthOpen / mouthWidth;
  return { smiling: ratio > SMILE_MOUTH_RATIO_SMILE, ratio };
}

function getNosePosition(lms: LandmarkLike[], w: number, h: number): XYPoint | null {
  if (!isVisible(lms, 0)) {
    return null;
  }
  return getXY(lms, 0, w, h);
}

class FidgetTracker {
  private leftHist: XYPoint[] = [];
  private rightHist: XYPoint[] = [];
  private noseHist: XYPoint[] = [];
  private readonly maxHist: number;

  constructor(maxHist = 15) {
    this.maxHist = maxHist;
  }

  updateMotion(leftXY: XYPoint, rightXY: XYPoint, scale: number, noseXY?: XYPoint | null): number {
    this.pushWithLimit(this.leftHist, leftXY);
    this.pushWithLimit(this.rightHist, rightXY);
    if (noseXY) {
      this.pushWithLimit(this.noseHist, noseXY);
    }

    const avgStep = (hist: XYPoint[]): number => {
      if (hist.length < 2) {
        return 0;
      }
      let total = 0;
      for (let i = 1; i < hist.length; i += 1) {
        total += dist2d(hist[i], hist[i - 1]);
      }
      return (total / (hist.length - 1)) / (scale + 1e-6);
    };

    let cameraMotion = 0;
    if (this.noseHist.length >= 2) {
      cameraMotion = avgStep(this.noseHist);
    }

    const leftJit = Math.max(0, avgStep(this.leftHist) - cameraMotion);
    const rightJit = Math.max(0, avgStep(this.rightHist) - cameraMotion);
    return 0.5 * (leftJit + rightJit);
  }

  private pushWithLimit(arr: XYPoint[], value: XYPoint) {
    arr.push(value);
    if (arr.length > this.maxHist) {
      arr.shift();
    }
  }
}

export interface FrameSummary {
  timestamp: number;
  postureOk: number;
  facingOk: number;
  calmOk: number;
  gestureOk: number;
  smileOk: number;
  handsRaised: number;
  landmarksVisible: number;
  fidgetDetected: number;
}

export interface TimelineHighlight {
  start: number;
  end: number;
  tag: 'Slouching' | 'Looking away' | 'Hand fidgeting' | 'Great presence';
  advice: string;
}

export interface SkillDetail {
  score: number;
  percentGood: number;
  label: string;
  whyItMatters: string;
  coachComment: string;
  moments: TimelineHighlight[];
  windowSeconds?: [number, number];
  paceWpm?: number;
  fillerPerMin?: number;
}

export interface TimeSeriesData {
  frameCount: number;
  durationSec: number;
  fps: number;
  t: number[];
  posture: number[];
  facing: number[];
  gesture: number[];
  calm: number[];
  smile: number[];
  torsoAngle: number[];
  depthDiff: number[];
  jitter: number[];
  mouthRatio: number[];
  handsRaised: number[];
  visible: number[];
  fidget: number[];
  format: {
    description: string;
    note: string;
    arrays: string[];
  };
}

export interface TraitSummary {
  name: string;
  score: number;
  summary: string;
}

export interface VideoAnalysisReport {
  presenceScore: number;
  bestTrait: TraitSummary;
  needsImprovement: TraitSummary;
  skills: {
    posture: SkillDetail;
    eyeContact: SkillDetail;
    gestures: SkillDetail;
    calmHands: SkillDetail;
    smile: SkillDetail;
    openingPresence: SkillDetail;
    voice: SkillDetail;
  };
  highlights: TimelineHighlight[];
  timeSeries: TimeSeriesData;
  voice: VoiceMetrics;
}

function pctGood(values: Array<boolean | null>): number {
  const valid = values.filter((v): v is boolean => v !== null);
  if (!valid.length) {
    return 0;
  }
  const good = valid.filter(Boolean).length;
  return good / valid.length;
}

function compressTimestamps(tsList: number[], maxGap = 0.75): Array<{ start: number; end: number }> {
  if (!tsList.length) {
    return [];
  }
  const sorted = [...tsList].sort((a, b) => a - b);
  const groups: Array<{ start: number; end: number }> = [];
  let start = sorted[0];
  let prev = sorted[0];
  for (let i = 1; i < sorted.length; i += 1) {
    const t = sorted[i];
    if (t - prev <= maxGap) {
      prev = t;
    } else {
      groups.push({ start, end: prev });
      start = t;
      prev = t;
    }
  }
  groups.push({ start, end: prev });
  return groups;
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

export class BodyLanguageSessionAnalyzer {
  private fidgetTracker = new FidgetTracker();

  private timestamps: number[] = [];
  private postureSeries: number[] = [];
  private facingSeries: number[] = [];
  private gestureSeries: number[] = [];
  private calmSeries: number[] = [];
  private smileSeries: number[] = [];
  private torsoAngles: number[] = [];
  private depthDiffs: number[] = [];
  private jitterSeries: number[] = [];
  private mouthRatios: number[] = [];
  private handsRaisedSeries: number[] = [];
  private visibleSeries: number[] = [];
  private fidgetSeries: number[] = [];

  private framesPostureOk: Array<boolean | null> = [];
  private framesFacingOk: Array<boolean | null> = [];
  private framesGestureOk: Array<boolean | null> = [];
  private framesCalmOk: Array<boolean | null> = [];
  private framesSmileOk: Array<boolean | null> = [];

  private openingPostureOk: boolean[] = [];
  private openingFacingOk: boolean[] = [];

  private timelineNotes: Record<'slouch' | 'looking_away' | 'fidget' | 'great_presence', number[]> = {
    slouch: [],
    looking_away: [],
    fidget: [],
    great_presence: [],
  };

  private lastFrameSummary: FrameSummary | null = null;

  reset() {
    this.fidgetTracker = new FidgetTracker();
    this.timestamps = [];
    this.postureSeries = [];
    this.facingSeries = [];
    this.gestureSeries = [];
    this.calmSeries = [];
    this.smileSeries = [];
    this.torsoAngles = [];
    this.depthDiffs = [];
    this.jitterSeries = [];
    this.mouthRatios = [];
    this.handsRaisedSeries = [];
    this.visibleSeries = [];
    this.fidgetSeries = [];

    this.framesPostureOk = [];
    this.framesFacingOk = [];
    this.framesGestureOk = [];
    this.framesCalmOk = [];
    this.framesSmileOk = [];

    this.openingPostureOk = [];
    this.openingFacingOk = [];

    this.timelineNotes = {
      slouch: [],
      looking_away: [],
      fidget: [],
      great_presence: [],
    };

    this.lastFrameSummary = null;
  }

  processFrame(
    poseLandmarks: LandmarkLike[] | undefined,
    faceLandmarks: LandmarkLike[] | undefined,
    width: number,
    height: number,
    timestampSec: number,
  ): FrameSummary {
    const postureOkDefault = -1;
    let postureOk = postureOkDefault;
    let facingOk = postureOkDefault;
    let gestureOk = postureOkDefault;
    let calmOk = postureOkDefault;
    let smileOk = postureOkDefault;
    let torsoAngle = -1;
    let depthDiff = -1;
    let jitter = -1;
    let mouthRatio = -1;
    let handsRaised = 0;
    let visible = 0;
    let fidgetDetected = 0;

    if (poseLandmarks && poseLandmarks.length) {
      visible = 1;
      const torsoAngleVal = torsoUprightAngleDeg(poseLandmarks, width, height);
      if (torsoAngleVal !== null) {
        torsoAngle = round(torsoAngleVal, 1);
        postureOk = torsoAngleVal < POSTURE_TILT_BAD_DEG ? 1 : 0;
      }

      const depthDiffVal = facingDepthDiff(poseLandmarks);
      if (depthDiffVal !== null) {
        depthDiff = round(depthDiffVal, 3);
        facingOk = depthDiffVal < FACING_DEPTH_DIFF_BAD ? 1 : 0;
      }

      const hands = handsInfo(poseLandmarks, width, height);
      const scale = bodyScale(poseLandmarks, width, height);
      const nosePos = getNosePosition(poseLandmarks, width, height);

      if (hands && scale !== null) {
        const highEnough = hands.lHeightFrac > GESTURE_MIN_HEIGHT_FRAC || hands.rHeightFrac > GESTURE_MIN_HEIGHT_FRAC;
        handsRaised = highEnough ? 1 : 0;
        const jitterVal = this.fidgetTracker.updateMotion(hands.lWr, hands.rWr, scale, nosePos);
        jitter = round(jitterVal, 4);

        if (jitterVal < FIDGET_MOTION_CALM) {
          calmOk = 1;
        } else if (jitterVal > FIDGET_MOTION_FIDGET) {
          calmOk = 0;
        } else {
          const midpoint = (FIDGET_MOTION_CALM + FIDGET_MOTION_FIDGET) / 2;
          calmOk = jitterVal < midpoint ? 1 : 0;
        }

        if (hands.wristsDist < 0.2 * scale && calmOk === 0) {
          this.timelineNotes.fidget.push(timestampSec);
          fidgetDetected = 1;
        }

        gestureOk = highEnough && calmOk === 1 ? 1 : 0;
      }

      if (postureOk === 0) {
        this.timelineNotes.slouch.push(timestampSec);
      }
      if (facingOk === 0) {
        this.timelineNotes.looking_away.push(timestampSec);
      }
      if (postureOk === 1 && facingOk === 1 && gestureOk === 1 && calmOk === 1) {
        this.timelineNotes.great_presence.push(timestampSec);
      }

      if (timestampSec <= OPENING_WINDOW_SEC) {
        if (postureOk !== postureOkDefault) {
          this.openingPostureOk.push(postureOk === 1);
        }
        if (facingOk !== postureOkDefault) {
          this.openingFacingOk.push(facingOk === 1);
        }
      }
    }

    if (faceLandmarks && faceLandmarks.length) {
      const { smiling, ratio } = smileProxyFromFace(faceLandmarks, width, height);
      smileOk = smiling ? 1 : 0;
      mouthRatio = round(ratio, 3);
    }

    this.timestamps.push(round(timestampSec, 2));
    this.postureSeries.push(postureOk);
    this.facingSeries.push(facingOk);
    this.gestureSeries.push(gestureOk);
    this.calmSeries.push(calmOk);
    this.smileSeries.push(smileOk);
    this.torsoAngles.push(torsoAngle);
    this.depthDiffs.push(depthDiff);
    this.jitterSeries.push(jitter);
    this.mouthRatios.push(mouthRatio);
    this.handsRaisedSeries.push(handsRaised);
    this.visibleSeries.push(visible);
    this.fidgetSeries.push(fidgetDetected);

    this.framesPostureOk.push(postureOk === postureOkDefault ? null : postureOk === 1);
    this.framesFacingOk.push(facingOk === postureOkDefault ? null : facingOk === 1);
    this.framesGestureOk.push(gestureOk === postureOkDefault ? null : gestureOk === 1);
    this.framesCalmOk.push(calmOk === postureOkDefault ? null : calmOk === 1);
    this.framesSmileOk.push(smileOk === postureOkDefault ? null : smileOk === 1);

    this.lastFrameSummary = {
      timestamp: timestampSec,
      postureOk,
      facingOk,
      calmOk,
      gestureOk,
      smileOk,
      handsRaised,
      landmarksVisible: visible,
      fidgetDetected,
    };

    return this.lastFrameSummary;
  }

  buildReport(voiceMetrics?: VoiceMetrics): VideoAnalysisReport {
    const voiceData = voiceMetrics ?? DEFAULT_VOICE_METRICS;

    const posturePercent = pctGood(this.framesPostureOk);
    const eyePercent = pctGood(this.framesFacingOk);
    const gesturePercent = pctGood(this.framesGestureOk);
    const calmPercent = pctGood(this.framesCalmOk);
    const smilePercent = pctGood(this.framesSmileOk);

    const openingPostureOk = this.openingPostureOk.length ? this.openingPostureOk : [false];
    const openingFacingOk = this.openingFacingOk.length ? this.openingFacingOk : [false];

    const openPosturePct = openingPostureOk.filter(Boolean).length / openingPostureOk.length;
    const openFacingPct = openingFacingOk.filter(Boolean).length / openingFacingOk.length;
    const openVoicePct = voiceData.quietStart ? 0 : 1;

    const openingPresenceScore = (0.4 * openPosturePct + 0.4 * openFacingPct + 0.2 * openVoicePct) * 100;

    const postureScore = posturePercent * 100;
    const eyeScore = eyePercent * 100;
    const gestureScore = gesturePercent * 100;
    const calmScore = calmPercent * 100;
    const smileScore = smilePercent * 100;
    const voiceScore = Number.isFinite(voiceData.voiceScore) ? voiceData.voiceScore : DEFAULT_VOICE_METRICS.voiceScore;

    const presenceScore =
      0.18 * postureScore +
      0.25 * eyeScore +
      0.25 * voiceScore +
      0.12 * gestureScore +
      0.05 * calmScore +
      0.05 * smileScore +
      0.1 * openingPresenceScore;

    const traitMap: Record<string, number> = {
      'Posture & Space': postureScore,
      'Eye Contact': eyeScore,
      'Gestures / Open Hands': gestureScore,
      'Calm Hands': calmScore,
      'Smile / Warmth': smileScore,
      'Opening Presence': openingPresenceScore,
      Voice: voiceScore,
    };

    const entries = Object.entries(traitMap);
    const bestTraitName = entries.reduce((best, current) => (current[1] > traitMap[best] ? current[0] : best), entries[0][0]);
    const worstTraitName = entries.reduce((worst, current) => (current[1] < traitMap[worst] ? current[0] : worst), entries[0][0]);

    const highlights: TimelineHighlight[] = [];
    const highlightTemplates: Array<{
      list: number[];
      tag: TimelineHighlight['tag'];
      advice: string;
    }> = [
      { list: this.timelineNotes.slouch, tag: 'Slouching', advice: 'Lift chest and roll shoulders back.' },
      { list: this.timelineNotes.looking_away, tag: 'Looking away', advice: 'Face the audience while delivering key lines.' },
      { list: this.timelineNotes.fidget, tag: 'Hand fidgeting', advice: 'Hands twisting together. Keep them apart and still.' },
      { list: this.timelineNotes.great_presence, tag: 'Great presence', advice: 'Open stance, facing forward, calm hands. Use this in your intro.' },
    ];

    highlightTemplates.forEach(({ list, tag, advice }) => {
      compressTimestamps(list).forEach(({ start, end }) => {
        highlights.push({ start: round(start, 2), end: round(end, 2), tag, advice });
      });
    });

    const skills = {
      posture: {
        score: Math.round(postureScore),
        percentGood: round(posturePercent, 2),
        label: 'Stand tall and take up space',
        whyItMatters: 'Strong posture = confidence signal.',
        coachComment: 'Keep shoulders open. Avoid leaning forward.',
        moments: highlights.filter((h) => h.tag === 'Slouching' || h.tag === 'Great presence'),
      },
      eyeContact: {
        score: Math.round(eyeScore),
        percentGood: round(eyePercent, 2),
        label: 'Maintain eye contact',
        whyItMatters: 'Facing the audience makes you sound certain.',
        coachComment: 'Square your chest forward when you speak.',
        moments: highlights.filter((h) => h.tag === 'Looking away' || h.tag === 'Great presence'),
      },
      gestures: {
        score: Math.round(gestureScore),
        percentGood: round(gesturePercent, 2),
        label: 'Talk with your hands / Use open gestures',
        whyItMatters: 'Open hands make you look trustworthy and engaged.',
        coachComment: 'Keep hands visible around chest height.',
        moments: highlights.filter((h) => h.tag === 'Great presence'),
      },
      calmHands: {
        score: Math.round(calmScore),
        percentGood: round(calmPercent, 2),
        label: 'Eliminate nervous gestures',
        whyItMatters: 'Fidgeting reads as stress.',
        coachComment: 'Avoid rubbing/twisting fingers.',
        moments: highlights.filter((h) => h.tag === 'Hand fidgeting'),
      },
      smile: {
        score: Math.round(smileScore),
        percentGood: round(smilePercent, 2),
        label: 'Smile',
        whyItMatters: 'Warmth builds trust.',
        coachComment: 'Bring that friendly energy right from the start.',
        moments: [],
      },
      openingPresence: {
        score: Math.round(openingPresenceScore),
        percentGood: round((openPosturePct + openFacingPct + openVoicePct) / 3, 2),
        label: 'Prime the pump',
        whyItMatters: 'Your first 5 seconds decide if people lean in.',
        coachComment: 'Start already facing forward and projecting.',
        moments: [],
        windowSeconds: [0, OPENING_WINDOW_SEC] as [number, number],
      },
      voice: {
        score: Math.round(voiceScore),
        percentGood: 0,
        label: 'Get that voice down',
        whyItMatters: 'Clear confident delivery makes people trust you.',
        coachComment: voiceData.comment || DEFAULT_VOICE_COMMENT,
        moments: [],
        paceWpm: voiceData.paceWpm,
        fillerPerMin: voiceData.fillerPerMin,
      },
    } satisfies VideoAnalysisReport['skills'];

    return {
      presenceScore: Math.round(presenceScore),
      bestTrait: {
        name: bestTraitName,
        score: Math.round(traitMap[bestTraitName]),
        summary: '',
      },
      needsImprovement: {
        name: worstTraitName,
        score: Math.round(traitMap[worstTraitName]),
        summary: '',
      },
      skills,
      highlights,
      timeSeries: {
        frameCount: this.timestamps.length,
        durationSec: this.timestamps.length ? round(this.timestamps[this.timestamps.length - 1], 2) : 0,
        fps:
          this.timestamps.length && this.timestamps[this.timestamps.length - 1] > 0
            ? round(this.timestamps.length / this.timestamps[this.timestamps.length - 1], 1)
            : 0,
        t: this.timestamps,
        posture: this.postureSeries,
        facing: this.facingSeries,
        gesture: this.gestureSeries,
        calm: this.calmSeries,
        smile: this.smileSeries,
        torsoAngle: this.torsoAngles,
        depthDiff: this.depthDiffs,
        jitter: this.jitterSeries,
        mouthRatio: this.mouthRatios,
        handsRaised: this.handsRaisedSeries,
        visible: this.visibleSeries,
        fidget: this.fidgetSeries,
        format: {
          description: 'Columnar time-series format for efficiency',
          note: 'Use -1 for missing/None values in numeric arrays',
          arrays: [
            't',
            'posture',
            'facing',
            'gesture',
            'calm',
            'smile',
            'torsoAngle',
            'depthDiff',
            'jitter',
            'mouthRatio',
            'handsRaised',
            'visible',
            'fidget',
          ],
        },
      },
      voice: voiceData,
    };
  }

  getLastFrame(): FrameSummary | null {
    return this.lastFrameSummary;
  }
}

export function createFeedbackMessage(frame: FrameSummary | null): string {
  if (!frame) {
    return 'Not tracking';
  }
  if (frame.landmarksVisible === 0) {
    return 'Step into view so Aura can see you.';
  }
  if (frame.postureOk === 0) {
    return 'Roll your shoulders back and lift your chest.';
  }
  if (frame.facingOk === 0) {
    return 'Face the camera to reinforce eye contact.';
  }
  if (frame.calmOk === 0 || frame.fidgetDetected === 1) {
    return 'Relax your hands—avoid rubbing or twisting.';
  }
  if (frame.gestureOk === 0) {
    return frame.handsRaised ? 'Keep gestures steady at chest height.' : 'Lift your hands to chest height when you gesture.';
  }
  if (frame.smileOk === 0) {
    return 'Add a quick smile to warm up the room.';
  }
  return 'Great presence—keep this energy!';
}

export { DEFAULT_VOICE_METRICS };
