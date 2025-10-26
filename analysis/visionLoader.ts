const WASM_ASSET_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';

interface PoseLandmarkerInstance {
  setOptions?(options: Record<string, unknown>): void;
  detectForVideo(video: HTMLVideoElement, timestamp: number): {
    landmarks?: Array<Array<{ x: number; y: number; z: number; visibility?: number; presence?: number }>>;
  } | null;
}

interface FaceLandmarkerInstance {
  setOptions?(options: Record<string, unknown>): void;
  detectForVideo(video: HTMLVideoElement, timestamp: number): {
    faceLandmarks?: Array<Array<{ x: number; y: number; z: number; visibility?: number; presence?: number }>>;
  } | null;
}

interface VisionModule {
  FilesetResolver: {
    forVisionTasks(baseUrl: string): Promise<unknown>;
  };
  PoseLandmarker: {
    createFromOptions(fileset: unknown, options: Record<string, unknown>): Promise<PoseLandmarkerInstance>;
  };
  FaceLandmarker: {
    createFromOptions(fileset: unknown, options: Record<string, unknown>): Promise<FaceLandmarkerInstance>;
  };
}

let visionModulePromise: Promise<VisionModule> | null = null;
let filesetResolverPromise: Promise<unknown> | null = null;
let poseLandmarkerPromise: Promise<PoseLandmarkerInstance> | null = null;
let faceLandmarkerPromise: Promise<FaceLandmarkerInstance> | null = null;

async function loadVisionModule(): Promise<VisionModule> {
  if (!visionModulePromise) {
    visionModulePromise = import(/* @vite-ignore */ 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14').then(
      (mod) => mod as unknown as VisionModule,
    );
  }
  return visionModulePromise;
}

async function getFilesetResolver(): Promise<unknown> {
  if (!filesetResolverPromise) {
    const vision = await loadVisionModule();
    filesetResolverPromise = vision.FilesetResolver.forVisionTasks(WASM_ASSET_BASE);
  }
  return filesetResolverPromise;
}

async function createPoseLandmarkerInstance(): Promise<PoseLandmarkerInstance> {
  const vision = await loadVisionModule();
  const fileset = await getFilesetResolver();
  return vision.PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: `${WASM_ASSET_BASE}/pose_landmarker_full.task`,
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    outputSegmentationMasks: false,
  });
}

async function createFaceLandmarkerInstance(): Promise<FaceLandmarkerInstance> {
  const vision = await loadVisionModule();
  const fileset = await getFilesetResolver();
  return vision.FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: `${WASM_ASSET_BASE}/face_landmarker.task`,
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
}

export async function getPoseLandmarker(): Promise<PoseLandmarkerInstance> {
  if (!poseLandmarkerPromise) {
    poseLandmarkerPromise = createPoseLandmarkerInstance();
  }
  return poseLandmarkerPromise;
}

export async function getFaceLandmarker(): Promise<FaceLandmarkerInstance> {
  if (!faceLandmarkerPromise) {
    faceLandmarkerPromise = createFaceLandmarkerInstance();
  }
  return faceLandmarkerPromise;
}

export type PoseResult = ReturnType<PoseLandmarkerInstance['detectForVideo']>;
export type FaceResult = ReturnType<FaceLandmarkerInstance['detectForVideo']>;

export function resetVisionModels() {
  visionModulePromise = null;
  filesetResolverPromise = null;
  poseLandmarkerPromise = null;
  faceLandmarkerPromise = null;
}
