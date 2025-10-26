<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/18bYCHVmoqoOx-DeS_0v9c5ZmsVLqDEUI

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

## Body-language analysis prototype (Python)

The `scripts/presence_analyzer.py` module mirrors the body-language prototype you
shared. It records audio and webcam footage, runs a Mediapipe-based analysis,
and writes a JSON report alongside the captured audio.

### Python setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r scripts/requirements.txt
   ```
   *`pydub`/`ffmpeg` are optional—without them the script saves a WAV file instead of an MP3.*
3. Ensure `ffmpeg` is installed and available on your `PATH` if you want MP3
   export (e.g., `brew install ffmpeg` on macOS or `apt-get install ffmpeg` on
   Linux).

### Running the analyzer

The analyzer needs access to your webcam and microphone:

```bash
python scripts/presence_analyzer.py
```

Useful flags:

* `--skip-audio` – record video only (helpful on devices without a microphone).
* `--video path/to/file.mp4` – skip live capture and analyze an existing video.
* `--voice-json voice.json` – include precomputed voice metrics in the report.
  *Try the bundled* `scripts/sample_voice_metrics.json` *for a quick demo.*
* `--output report.json` / `--audio-output audio.mp3` – choose output paths.

The script starts a short countdown, records until you press `q`, and then runs
the analysis. Results are written to `last_session.json` by default (or to the
path supplied with `--output`). When audio capture succeeds you also get
`audio.mp3`; otherwise a fallback WAV file is kept.
