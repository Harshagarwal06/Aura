import cv2
import mediapipe as mp
import math
import json
import time
from collections import deque
import threading
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as wav_write
from pydub import AudioSegment
import os

### ---------------------------
### Tunable thresholds
### ---------------------------

POSTURE_TILT_BAD_DEG = 20.0
FACING_DEPTH_DIFF_BAD = 0.20
FIDGET_MOTION_CALM = 0.02
FIDGET_MOTION_FIDGET = 0.06
GESTURE_MIN_HEIGHT_FRAC = 0.6
SMILE_MOUTH_RATIO_SMILE = 0.35
OPENING_WINDOW_SEC = 5.0
MIN_LANDMARK_VISIBILITY = 0.5

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

### ---------------------------
### Geometry helpers
### ---------------------------

def dist2d(a, b):
    return math.dist(a, b)

def get_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)

def is_visible(landmarks, idx, threshold=MIN_LANDMARK_VISIBILITY):
    """Check if landmark has sufficient visibility"""
    return landmarks[idx].visibility > threshold

def torso_upright_angle_deg(lms, w, h):
    """Calculate torso angle with visibility checks"""
    if not all(is_visible(lms, i) for i in [11, 12, 23, 24]):
        return None

    l_sh = get_xy(lms, 11, w, h)
    r_sh = get_xy(lms, 12, w, h)
    l_hp = get_xy(lms, 23, w, h)
    r_hp = get_xy(lms, 24, w, h)
    mid_sh = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    mid_hp = ((l_hp[0] + r_hp[0]) / 2, (l_hp[1] + r_hp[1]) / 2)
    dx = mid_sh[0] - mid_hp[0]
    dy = mid_sh[1] - mid_hp[1]
    angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
    return angle_from_vertical

def facing_depth_diff(lms):
    """Calculate facing direction with visibility checks"""
    if not all(is_visible(lms, i) for i in [11, 12]):
        return None
    l_sh_z = lms[11].z
    r_sh_z = lms[12].z
    return abs(l_sh_z - r_sh_z)

def body_scale(lms, w, h):
    """Calculate body scale (shoulder width) with visibility checks"""
    if not all(is_visible(lms, i) for i in [11, 12]):
        return None
    ls = get_xy(lms, 11, w, h)
    rs = get_xy(lms, 12, w, h)
    return dist2d(ls, rs) + 1e-6

def hands_info(lms, w, h):
    """Get hand positions and metrics with visibility checks"""
    if not all(is_visible(lms, i) for i in [11, 12, 15, 16, 23, 24]):
        return None

    l_wr = get_xy(lms, 15, w, h)
    r_wr = get_xy(lms, 16, w, h)

    l_sh = get_xy(lms, 11, w, h)
    r_sh = get_xy(lms, 12, w, h)
    mid_chest = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)

    l_hp = get_xy(lms, 23, w, h)
    r_hp = get_xy(lms, 24, w, h)
    mid_hip = ((l_hp[0] + r_hp[0]) / 2, (l_hp[1] + r_hp[1]) / 2)

    torso_height = abs(mid_hip[1] - mid_chest[1]) + 1e-6

    l_height_frac = 1.0 - ((l_wr[1] - mid_chest[1]) / torso_height)
    r_height_frac = 1.0 - ((r_wr[1] - mid_chest[1]) / torso_height)

    wrists_dist = dist2d(l_wr, r_wr)

    return {
        "l_wr": l_wr,
        "r_wr": r_wr,
        "l_height_frac": l_height_frac,
        "r_height_frac": r_height_frac,
        "wrists_dist": wrists_dist,
        "torso_height": torso_height,
    }

def smile_proxy_from_face(face_landmarks, w, h):
    """Detect smiling with improved landmarks"""
    try:
        def xy(idx):
            lm = face_landmarks[idx]
            return (lm.x * w, lm.y * h)

        left_corner = xy(78)   # Left mouth corner
        right_corner = xy(308) # Right mouth corner
        top_lip = xy(13)       # Upper lip top
        bot_lip = xy(14)       # Lower lip bottom

        mouth_w = dist2d(left_corner, right_corner) + 1e-6
        mouth_open = dist2d(top_lip, bot_lip)

        ratio = mouth_open / mouth_w
        smiling = ratio > SMILE_MOUTH_RATIO_SMILE
        return smiling, ratio
    except Exception:
        return False, 0.0

def get_nose_position(lms, w, h):
    """Get nose position for camera motion compensation"""
    if not is_visible(lms, 0):  # Nose is landmark 0
        return None
    return get_xy(lms, 0, w, h)

### ---------------------------
### Fidget tracker with motion compensation
### ---------------------------

class FidgetTracker:
    def __init__(self, max_hist=15):
        self.left_hist = deque(maxlen=max_hist)
        self.right_hist = deque(maxlen=max_hist)
        self.nose_hist = deque(maxlen=max_hist)
        self.last_jitter = 0.0
        self._has_history = False

    def update_motion(self, left_xy, right_xy, scale, nose_xy=None):
        """Calculate hand jitter with camera motion compensation"""
        self.left_hist.append(left_xy)
        self.right_hist.append(right_xy)
        if nose_xy is not None:
            self.nose_hist.append(nose_xy)

        def avg_step(hist):
            if len(hist) < 2:
                return 0.0
            total = 0.0
            for i in range(1, len(hist)):
                total += dist2d(hist[i], hist[i - 1])
            return (total / (len(hist) - 1)) / (scale + 1e-6)

        camera_motion = 0.0
        if len(self.nose_hist) >= 2:
            camera_motion = avg_step(self.nose_hist)

        left_jit = max(0.0, avg_step(self.left_hist) - camera_motion)
        right_jit = max(0.0, avg_step(self.right_hist) - camera_motion)

        jitter_val = 0.5 * (left_jit + right_jit)
        self.last_jitter = jitter_val
        self._has_history = True
        return jitter_val

    def last_or_default(self):
        return self.last_jitter if self._has_history else None

### ---------------------------
### Frame-by-frame analyzer with time-series output
### ---------------------------

def analyze_session(frames, timestamps_sec, voice_data=None):
    with mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        fidget_tracker = FidgetTracker()

        frame_timestamps = []
        frame_posture_ok = []
        frame_facing_ok = []
        frame_gesture_ok = []
        frame_calm_ok = []
        frame_smile_ok = []
        frame_torso_angles = []
        frame_depth_diffs = []
        frame_jitter_scores = []
        frame_mouth_ratios = []
        frame_hands_raised = []
        frame_landmarks_visible = []
        frame_fidget_detected = []

        frames_posture_ok = []
        frames_facing_ok = []
        frames_gesture_ok = []
        frames_calm_ok = []
        frames_smile_ok = []

        timeline_notes = {
            "slouch": [],
            "looking_away": [],
            "fidget": [],
            "great_presence": [],
        }

        opening_posture_ok = []
        opening_facing_ok = []

        no_pose_frames = 0
        no_face_frames = 0
        pose_warned = False
        face_warned = False

        for frame_idx, (frame, t_sec) in enumerate(zip(frames, timestamps_sec)):
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            posture_ok = -1
            facing_ok = -1
            gesture_ok = -1
            calm_ok = -1
            smile_ok = -1
            torso_angle = -1
            depth_diff = -1
            jitter = -1
            mouth_ratio = -1
            hands_raised = 0
            landmarks_visible = 0
            fidget_detected = 0

            pose_res = pose.process(rgb) if pose else None
            if pose_res and pose_res.pose_landmarks:
                no_pose_frames = 0
                lms = pose_res.pose_landmarks.landmark
                landmarks_visible = 1

                torso_angle_val = torso_upright_angle_deg(lms, w, h)
                if torso_angle_val is not None:
                    torso_angle = round(torso_angle_val, 1)
                    posture_ok = 1 if torso_angle_val < POSTURE_TILT_BAD_DEG else 0

                depth_diff_val = facing_depth_diff(lms)
                if depth_diff_val is not None:
                    depth_diff = round(depth_diff_val, 3)
                    facing_ok = 1 if depth_diff_val < FACING_DEPTH_DIFF_BAD else 0

                hi = hands_info(lms, w, h)
                scale = body_scale(lms, w, h)
                nose_pos = get_nose_position(lms, w, h)

                if hi is not None and scale is not None:
                    high_enough = (
                        hi["l_height_frac"] > GESTURE_MIN_HEIGHT_FRAC
                        or hi["r_height_frac"] > GESTURE_MIN_HEIGHT_FRAC
                    )
                    hands_raised = 1 if high_enough else 0

                    jitter_val = fidget_tracker.update_motion(
                        hi["l_wr"], hi["r_wr"], scale, nose_pos
                    )
                    jitter = round(jitter_val, 4)

                    if jitter_val < FIDGET_MOTION_CALM:
                        calm_ok = 1
                    elif jitter_val > FIDGET_MOTION_FIDGET:
                        calm_ok = 0
                    else:
                        calm_ok = 1 if jitter_val < (FIDGET_MOTION_CALM + FIDGET_MOTION_FIDGET) / 2 else 0

                    if hi["wrists_dist"] < (0.2 * scale) and calm_ok == 0:
                        timeline_notes["fidget"].append(t_sec)
                        fidget_detected = 1

                    gesture_ok = 1 if (high_enough and calm_ok == 1) else 0

                else:
                    last_jitter = fidget_tracker.last_or_default()
                    if last_jitter is not None:
                        jitter = round(last_jitter, 4)
                        if jitter < FIDGET_MOTION_CALM:
                            calm_ok = 1
                        elif jitter > FIDGET_MOTION_FIDGET:
                            calm_ok = 0
                        else:
                            calm_ok = 1 if jitter < (FIDGET_MOTION_CALM + FIDGET_MOTION_FIDGET) / 2 else 0
            else:
                no_pose_frames += 1
                if not pose_warned and no_pose_frames >= 30:
                    print("WARN: no pose landmarks for 30 frames (check lighting / camera angle)")
                    pose_warned = True

            face_res = face_mesh.process(rgb) if face_mesh else None
            if face_res and face_res.multi_face_landmarks:
                no_face_frames = 0
                f_lms = face_res.multi_face_landmarks[0].landmark
                smiling, ratio = smile_proxy_from_face(f_lms, w, h)
                smile_ok = 1 if smiling else 0
                mouth_ratio = round(ratio, 3)
            else:
                no_face_frames += 1
                if not face_warned and no_face_frames >= 30:
                    print("WARN: no face landmarks for 30 frames (check lighting / camera angle)")
                    face_warned = True

            if posture_ok == 0:
                timeline_notes["slouch"].append(t_sec)
            if facing_ok == 0:
                timeline_notes["looking_away"].append(t_sec)
            if (
                posture_ok == 1
                and facing_ok == 1
                and gesture_ok == 1
                and calm_ok == 1
            ):
                timeline_notes["great_presence"].append(t_sec)

            if t_sec <= OPENING_WINDOW_SEC:
                if posture_ok != -1:
                    opening_posture_ok.append(posture_ok == 1)
                if facing_ok != -1:
                    opening_facing_ok.append(facing_ok == 1)

            frame_timestamps.append(round(t_sec, 2))
            frame_posture_ok.append(posture_ok)
            frame_facing_ok.append(facing_ok)
            frame_gesture_ok.append(gesture_ok)
            frame_calm_ok.append(calm_ok)
            frame_smile_ok.append(smile_ok)
            frame_torso_angles.append(torso_angle)
            frame_depth_diffs.append(depth_diff)
            frame_jitter_scores.append(jitter)
            frame_mouth_ratios.append(mouth_ratio)
            frame_hands_raised.append(hands_raised)
            frame_landmarks_visible.append(landmarks_visible)
            frame_fidget_detected.append(fidget_detected)

            frames_posture_ok.append(None if posture_ok == -1 else (posture_ok == 1))
            frames_facing_ok.append(None if facing_ok == -1 else (facing_ok == 1))
            frames_gesture_ok.append(None if gesture_ok == -1 else (gesture_ok == 1))
            frames_calm_ok.append(None if calm_ok == -1 else (calm_ok == 1))
            frames_smile_ok.append(None if smile_ok == -1 else (smile_ok == 1))

        def pct_good(arr):
            valid = [x for x in arr if x is not None]
            if not valid:
                return 0.0
            return sum(1 for x in valid if x) / len(valid)

        posture_percent = pct_good(frames_posture_ok)
        eye_percent = pct_good(frames_facing_ok)
        gesture_percent = pct_good(frames_gesture_ok)
        calm_percent = pct_good(frames_calm_ok)
        smile_percent = pct_good(frames_smile_ok)

        if voice_data is None:
            voice_data = {
                "voice_score": 70,
                "pace_wpm": 170,
                "filler_per_min": 8,
                "comment": "Good projection. Moderate filler words.",
                "quiet_start": False,
            }

        voice_score = float(voice_data.get("voice_score", 70.0))
        quiet_start = voice_data.get("quiet_start", False)

        if not opening_posture_ok:
            opening_posture_ok = [False]
        if not opening_facing_ok:
            opening_facing_ok = [False]

        open_posture_pct = sum(1 for x in opening_posture_ok if x) / len(opening_posture_ok)
        open_facing_pct = sum(1 for x in opening_facing_ok if x) / len(opening_facing_ok)
        open_voice_pct = 0.0 if quiet_start else 1.0

        opening_presence_score = (
            0.4 * open_posture_pct
            + 0.4 * open_facing_pct
            + 0.2 * open_voice_pct
        ) * 100.0

        posture_score = posture_percent * 100.0
        eye_score = eye_percent * 100.0
        gesture_score = gesture_percent * 100.0
        calm_score = calm_percent * 100.0
        smile_score = smile_percent * 100.0

        presence_score = (
            0.18 * posture_score
            + 0.25 * eye_score
            + 0.25 * voice_score
            + 0.12 * gesture_score
            + 0.05 * calm_score
            + 0.05 * smile_score
            + 0.10 * opening_presence_score
        )

        trait_map = {
            "Posture & Space": posture_score,
            "Eye Contact": eye_score,
            "Gestures / Open Hands": gesture_score,
            "Calm Hands": calm_score,
            "Smile / Warmth": smile_score,
            "Opening Presence": opening_presence_score,
            "Voice": voice_score,
        }

        highlights = []
        any_visible = any(frame_landmarks_visible)
        if any_visible:
            def compress_timestamps(ts_list, max_gap=0.75):
                if not ts_list:
                    return []
                ts_list = sorted(ts_list)
                groups = []
                start = ts_list[0]
                prev = ts_list[0]
                for t in ts_list[1:]:
                    if t - prev <= max_gap:
                        prev = t
                    else:
                        groups.append((start, prev))
                        start = t
                        prev = t
                groups.append((start, prev))
                return groups

            for (lst, tag, advice) in [
                (timeline_notes["slouch"], "Slouching", "Lift chest and roll shoulders back."),
                (timeline_notes["looking_away"], "Looking away", "Face the audience while delivering key lines."),
                (timeline_notes["fidget"], "Hand fidgeting", "Hands twisting together. Keep them apart and still."),
                (timeline_notes["great_presence"], "Great presence", "Open stance, facing forward, calm hands. Use this in your intro."),
            ]:
                for (s, e) in compress_timestamps(lst):
                    highlights.append(
                        {
                            "start": round(s, 2),
                            "end": round(e, 2),
                            "tag": tag,
                            "advice": advice,
                        }
                    )

            best_trait_name = max(trait_map, key=lambda k: trait_map[k])
            worst_trait_name = min(trait_map, key=lambda k: trait_map[k])
        else:
            posture_score = 0.0
            eye_score = 0.0
            gesture_score = 0.0
            calm_score = 0.0
            smile_score = 0.0
            opening_presence_score = 0.0
            voice_score = 0.0
            presence_score = 0.0
            trait_map = {
                "Posture & Space": posture_score,
                "Eye Contact": eye_score,
                "Gestures / Open Hands": gesture_score,
                "Calm Hands": calm_score,
                "Smile / Warmth": smile_score,
                "Opening Presence": opening_presence_score,
                "Voice": voice_score,
            }
            best_trait_name = "No subject detected"
            worst_trait_name = "No subject detected"

        result = {
            "presence_score": round(presence_score),
            "best_trait": {
                "name": best_trait_name,
                "score": round(trait_map[best_trait_name]) if any_visible else 0,
                "summary": "",
            },
            "needs_improvement": {
                "name": worst_trait_name,
                "score": round(trait_map[worst_trait_name]) if any_visible else 0,
                "summary": "",
            },
            "time_series": {
                "frame_count": len(frame_timestamps),
                "duration_sec": round(timestamps_sec[-1] if timestamps_sec else 0, 2),
                "fps": round(len(frame_timestamps) / timestamps_sec[-1], 1)
                if timestamps_sec and timestamps_sec[-1] > 0
                else 0,
                "t": frame_timestamps,
                "posture": frame_posture_ok,
                "facing": frame_facing_ok,
                "gesture": frame_gesture_ok,
                "calm": frame_calm_ok,
                "smile": frame_smile_ok,
                "torso_angle": frame_torso_angles,
                "depth_diff": frame_depth_diffs,
                "jitter": frame_jitter_scores,
                "mouth_ratio": frame_mouth_ratios,
                "hands_raised": frame_hands_raised,
                "visible": frame_landmarks_visible,
                "fidget": frame_fidget_detected,
                "_format": {
                    "description": "Columnar time-series format for efficiency",
                    "note": "Use -1 for missing/None values in numeric arrays",
                    "example": "posture[i] == 1 (good), 0 (bad), -1 (no data)",
                    "arrays": [
                        "t",
                        "posture",
                        "facing",
                        "gesture",
                        "calm",
                        "smile",
                        "torso_angle",
                        "depth_diff",
                        "jitter",
                        "mouth_ratio",
                        "hands_raised",
                        "visible",
                        "fidget",
                    ],
                },
            },
            "details": {
                "skills": {
                    "posture": {
                        "score": round(posture_score),
                        "percent_good": round(posture_percent, 2),
                        "label": "Stand tall and take up space",
                        "why_it_matters": "Strong posture = confidence signal.",
                        "coach_comment": "Keep shoulders open. Avoid leaning forward.",
                        "moments": [
                            h
                            for h in highlights
                            if h["tag"] in ["Slouching", "Great presence"]
                        ],
                    },
                    "eye_contact": {
                        "score": round(eye_score),
                        "percent_good": round(eye_percent, 2),
                        "label": "Maintain eye contact",
                        "why_it_matters": "Facing the audience makes you sound certain.",
                        "coach_comment": "Square your chest forward when you speak.",
                        "moments": [
                            h
                            for h in highlights
                            if h["tag"] in ["Looking away", "Great presence"]
                        ],
                    },
                    "gestures": {
                        "score": round(gesture_score),
                        "percent_good": round(gesture_percent, 2),
                        "label": "Talk with your hands / Use open gestures",
                        "why_it_matters": "Open hands make you look trustworthy and engaged.",
                        "coach_comment": "Keep hands visible around chest height.",
                        "moments": [
                            h for h in highlights if h["tag"] in ["Great presence"]
                        ],
                    },
                    "calm_hands": {
                        "score": round(calm_score),
                        "percent_good": round(calm_percent, 2),
                        "label": "Eliminate nervous gestures",
                        "why_it_matters": "Fidgeting reads as stress.",
                        "coach_comment": "Avoid rubbing/twisting fingers.",
                        "moments": [
                            h for h in highlights if h["tag"] in ["Hand fidgeting"]
                        ],
                    },
                    "smile": {
                        "score": round(smile_score),
                        "percent_good": round(smile_percent, 2),
                        "label": "Smile",
                        "why_it_matters": "Warmth builds trust.",
                        "coach_comment": "Bring that friendly energy right from the start.",
                        "moments": [],
                    },
                    "opening_presence": {
                        "score": round(opening_presence_score),
                        "percent_good": round(
                            (open_posture_pct + open_facing_pct + open_voice_pct) / 3.0,
                            2,
                        ),
                        "label": "Prime the pump",
                        "why_it_matters": "Your first 5 seconds decide if people lean in.",
                        "coach_comment": "Start already facing forward and projecting.",
                        "window_seconds": [0, OPENING_WINDOW_SEC],
                    },
                    "voice": {
                        "score": round(voice_score),
                        "label": "Get that voice down",
                        "why_it_matters": "Clear confident delivery makes people trust you.",
                        "pace_wpm": voice_data.get("pace_wpm", None),
                        "filler_per_min": voice_data.get("filler_per_min", None),
                        "coach_comment": voice_data.get("comment", ""),
                    },
                },
                "highlights": highlights,
            },
        }

        return result

### ---------------------------
### Audio recorder
### ---------------------------

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.frames = []
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio callback status: {status}")
        if self.recording:
            self.frames.append(indata.copy())

    def start(self):
        """Returns start timestamp for sync"""
        self.frames = []
        self.recording = True

        print("\nAvailable audio devices:")
        print(sd.query_devices())

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._callback,
                dtype=np.float32,
            )
            self.stream.start()
            print(f"✓ Audio recording started: {self.sample_rate}Hz, {self.channels} channel(s)")
        except Exception as e:
            print(f"✗ Audio recording failed: {e}")
            self.stream = None

        start_time = time.time()
        return start_time

    def stop(self, filename="audio.mp3"):
        """Stop recording and save as MP3"""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

        if len(self.frames) == 0:
            print("⚠ No audio data recorded!")
            return None

        audio_data = np.concatenate(self.frames, axis=0)
        print(
            f"✓ Captured {len(audio_data)} audio samples ({len(audio_data)/self.sample_rate:.2f}s)"
        )

        temp_wav = "temp_audio.wav"

        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)

        audio_int16 = np.int16(audio_data * 32767)
        wav_write(temp_wav, self.sample_rate, audio_int16)

        try:
            print("Converting to MP3...")
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(
                filename,
                format="mp3",
                bitrate="128k",
                parameters=["-ar", str(self.sample_rate)],
            )

            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                os.remove(temp_wav)
                print(f"✓ Audio saved as {filename} ({os.path.getsize(filename)} bytes)")
                return filename
            else:
                print("✗ MP3 file is empty or not created")
                print(f"  Keeping WAV file as {temp_wav}")
                return temp_wav

        except Exception as e:
            print(f"✗ MP3 conversion failed: {e}")
            print(
                "  This usually means ffmpeg is not installed or not in PATH"
            )
            print("  Keeping WAV file as temp_audio.wav")
            return temp_wav

### ---------------------------
### Live AV capture with synchronized timestamps
### ---------------------------

def record_av_session(max_duration_sec=60):
    """
    Records webcam frames + timestamps AND mic audio with synchronized clock
    """
    audio_rec = AudioRecorder(sample_rate=16000, channels=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frames = []
    timestamps = []

    print("Starting in 3 seconds...")
    time.sleep(3)

    start_time = time.time()
    audio_start = audio_rec.start()

    print("Recording A/V... Press 'q' to stop.")
    consecutive_initial_failures = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            if not frames:
                consecutive_initial_failures += 1
                if consecutive_initial_failures >= 30:
                    cap.release()
                    cv2.destroyAllWindows()
                    audio_rec.stop()
                    raise RuntimeError("Webcam is returning no frames")
            else:
                print("WARN: dropped frame")
            continue

        consecutive_initial_failures = 0
        frame = cv2.flip(frame, 1)
        now = time.time()
        t_sec = now - start_time

        preview = frame.copy()
        cv2.putText(
            preview,
            f"REC {t_sec:.1f}s - press q to stop",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Recording", preview)

        frames.append(frame)
        timestamps.append(t_sec)

        if (now - start_time) >= max_duration_sec:
            print("Auto-stop: duration limit reached.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    duration_sec = time.time() - start_time
    audio_path = audio_rec.stop(filename="audio.mp3")

    if audio_path is None:
        print("⚠ Recording completed but no audio was captured!")
        print("  Check microphone permissions and device availability")

    print(f"Recorded {len(frames)} frames over {duration_sec:.2f}s")
    return frames, timestamps, audio_path, duration_sec

### ---------------------------
### Main
### ---------------------------

if __name__ == "__main__":
    frames, timestamps, audio_file, dur = record_av_session()

    voice_metrics = {
        "voice_score": 70,
        "pace_wpm": 170,
        "filler_per_min": 8,
        "comment": "Good projection. Moderate filler words.",
        "quiet_start": False,
    }

    print("Analyzing body language frame-by-frame...")
    report = analyze_session(frames, timestamps, voice_data=voice_metrics)

    report["audio_capture"] = {
        "audio_path": audio_file,
        "duration_sec": round(dur, 2),
        "sample_rate_hz": 16000,
    }

    print("\n" + "=" * 60)
    print(f"PRESENCE SCORE: {report['presence_score']}/100")
    print(
        f"Analyzed {report['time_series']['frame_count']} frames @ {report['time_series']['fps']} fps"
    )
    print(f"Duration: {report['time_series']['duration_sec']}s")
    print(
        f"Best trait: {report['best_trait']['name']} ({report['best_trait']['score']})"
    )
    print(
        f"Needs work: {report['needs_improvement']['name']} ({report['needs_improvement']['score']})"
    )
    print("=" * 60)

    print("\nSample frame data (first 3 frames):")
    for i in range(min(3, len(report['time_series']['t']))):
        print(f"  Frame {i} @ {report['time_series']['t'][i]}s:")
        print(
            f"    Posture: {report['time_series']['posture'][i]} (angle: {report['time_series']['torso_angle'][i]}°)"
        )
        print(
            f"    Facing: {report['time_series']['facing'][i]} (depth: {report['time_series']['depth_diff'][i]})"
        )
        print(
            f"    Calm: {report['time_series']['calm'][i]} (jitter: {report['time_series']['jitter'][i]})"
        )
        print(
            f"    Smile: {report['time_series']['smile'][i]} (ratio: {report['time_series']['mouth_ratio'][i]})"
        )

    json_str = json.dumps(report, indent=2)
    with open("last_session.json", "w") as f:
        f.write(json_str)

    json_size_kb = len(json_str) / 1024

    if audio_file:
        print(f"\nSaved last_session.json ({json_size_kb:.1f} KB) and {audio_file}")
    else:
        print(f"\nSaved last_session.json ({json_size_kb:.1f} KB) - no audio recorded")

    print("Time-series format: Columnar arrays (use time_series.posture[i], etc.)")
    if report['time_series']['frame_count']:
        print(
            f"Memory efficient: ~{json_size_kb / report['time_series']['frame_count']:.2f} KB per frame"
        )
    else:
        print("Memory efficient: n/a (no frames captured)")
