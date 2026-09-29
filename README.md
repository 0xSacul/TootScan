# 💨 TootScan

> **Automatically detect and extract farts and burps from your video collection.**
>
> Powered by YAMNet, FFmpeg, and questionable life choices.

TootScan scans your local video clips, listens very carefully for humanity's greatest acoustic achievements, detects **farts** and **burps**, and automatically extracts each occurrence into its own video clip.

Yes, this is real.

Yes, it works.

And yes, machine learning was absolutely necessary.

> **Even ChatGPT said: _"for once, we put IA in a really important scientific mission."_**

---

## Why?

You know the situation.

You have hundreds — possibly thousands — of Medal clips sitting on a drive.

Somewhere inside those clips are moments of historical importance:

- a perfectly timed fart during an intense fight;
- a microphone-destroying burp;
- someone casually committing acoustic terrorism mid-conversation;
- that one sound you remember happening three years ago but have absolutely no idea which clip contains it.

Manually reviewing 700 hours of gameplay would be unreasonable.

So naturally, we use machine learning.

---

## What does TootScan do?

TootScan:

- 🔎 recursively scans your video folders;
- 🎧 extracts audio using **FFmpeg**;
- 🧠 analyzes it locally using **YAMNet**;
- 💨 detects `Fart`;
- 🤢 detects `Burping, eructation`;
- 🕐 identifies every occurrence, including multiple events in the same video;
- ✂️ extracts a short video around each detected event;
- 📊 keeps the confidence score and timestamp;
- 💾 caches analyzed files so you don't process your entire library again;
- 🎙️ can analyze a specific audio track if your recordings separate microphone/game audio;
- 🔒 runs entirely locally.

No API.

No cloud uploads.

No subscription.

Your farts remain private.

---

## How it works

The extremely sophisticated research pipeline looks something like this:

```text
Video library
     │
     ▼
┌──────────────┐
│    FFmpeg    │
│ Extract audio│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    YAMNet    │
│ 521 classes  │
└──────┬───────┘
       │
       ├──── Fart
       │
       └──── Burping, eructation
              │
              ▼
      Detection timestamps
              │
              ▼
┌────────────────────────┐
│ Merge nearby detections│
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│ Extract glorious moment│
└────────────────────────┘
```

YAMNet analyzes the audio in small overlapping windows and produces confidence scores for hundreds of AudioSet sound classes.

TootScan only cares about the two classes that matter.

Science has priorities.

---

# Requirements

## Python

Python **3.11 is recommended**.

Newer Python versions may work, but TensorFlow enjoys making version compatibility unnecessarily exciting.

Check your version:

```powershell
py -0p
```

If Python 3.11 isn't installed:

```powershell
winget install Python.Python.3.11
```

---

## FFmpeg

TootScan uses FFmpeg for audio extraction and video cutting.

Install it on Windows:

```powershell
winget install Gyan.FFmpeg
```

Restart your terminal afterward and verify:

```powershell
ffmpeg -version
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/tootscan.git
cd tootscan
```

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

### PowerShell users

If PowerShell refuses to activate the environment because scripts are disabled:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You can also skip activating the virtual environment entirely and use:

```powershell
.\.venv\Scripts\python.exe
```

because sometimes fighting PowerShell is less productive than simply going around it.

---

# Usage

Basic usage:

```powershell
python .\tootscan.py "E:\Medal\Clips" `
  --output "C:\TootScan\output"
```

TootScan will recursively search the source directory for supported video files.

Example:

```text
Scanning clips...

[18/427] GTAV_2026_08_14.mp4
  analyzed -> 2 event(s)

  ✓ FART @ 42.72s
    confidence: 83.4%

  ✓ BURP @ 113.28s
    confidence: 71.8%

[19/427] GTAV_2026_08_15.mp4
  no gaseous activity detected
```

A sentence nobody expected computers to print in 2026.

---

# Output

Example output directory:

```text
output/
├── farts/
│   ├── GTAV_2026_08_14__fart_001__42.72s__0.83.mp4
│   ├── GTAV_2026_08_14__fart_002__198.24s__0.78.mp4
│   └── ...
│
├── burps/
│   ├── GTAV_2026_07_20__burp_001__18.24s__0.91.mp4
│   └── ...
│
├── detections.json
└── .gas_detector_cache.json
```

Each detection is exported separately.

By default, TootScan keeps a few seconds before and after the detected event so you don't just get a 400 ms clip of unexplained gastrointestinal violence.

---

# Dry run

Want to scan everything without creating video clips?

Use:

```powershell
python .\tootscan.py "E:\Medal\Clips" `
  --output ".\output" `
  --dry-run
```

This is useful for checking detection quality before unleashing the extractor on a massive archive.

---

# Confidence thresholds

Machine learning is not magic.

Sometimes a chair squeak believes deeply in its heart that it is a fart.

You can adjust the detection thresholds.

Example:

```powershell
python .\tootscan.py "E:\Medal\Clips" `
  --output ".\output" `
  --fart-threshold 0.50 `
  --burp-threshold 0.50
```

Higher threshold:

```text
fewer detections
more confidence
less furniture classified as digestive activity
```

Lower threshold:

```text
more detections
more false positives
greater chance of discovering a hidden masterpiece
```

The default threshold is intentionally somewhat permissive.

---

# Multiple audio tracks

Some Medal recordings contain multiple audio streams.

For example:

```text
Stream 0 → Game + Discord
Stream 1 → Microphone
```

If the interesting biological events happened on your microphone track, you can analyze only that stream:

```powershell
python .\tootscan.py "E:\Medal\Clips" `
  --output ".\output" `
  --audio-stream 1
```

This can dramatically reduce false positives from:

- gunshots;
- cars;
- explosions;
- game sound effects;
- Discord;
- suspicious noises generated by other humans.

---

# Cache

Analyzing hundreds or thousands of clips can take a while.

TootScan therefore keeps a local cache:

```text
.gas_detector_cache.json
```

Already processed and unchanged files are skipped on future runs.

So if you scan 2,000 clips today and add another 15 tomorrow, TootScan doesn't develop amnesia and start listening to the first 2,000 again.

---

# Detection accuracy

TootScan currently uses **YAMNet**, a general-purpose audio classification model trained on AudioSet.

Conveniently, AudioSet includes the classes:

```text
Fart
Burping, eructation
```

Apparently someone at Google also understood what machine learning should be used for.

Because YAMNet is a general audio model, false positives are possible.

Potential enemies include:

```text
chair squeaks
microphone clipping
mouth noises
low-frequency game sounds
vehicle noises
explosions
someone breathing aggressively
the mysterious noise your microphone makes every Tuesday
```

If accuracy isn't good enough, experiment with the confidence thresholds.

---

# Privacy

Everything happens locally.

TootScan does **not**:

- upload your videos;
- upload your audio;
- send detections to an API;
- report your gastrointestinal statistics to Google;
- build an advertising profile based on your fart frequency.

YAMNet runs on your machine.

Your gas is your business.

---

# Performance

The expensive part is audio inference, not video decoding.

TootScan extracts low-resolution audio suitable for YAMNet instead of processing video frames.

Your GPU is therefore not required.

A reasonably modern CPU should work perfectly fine.

Which means you do **not** need an RTX 5090 to locate a fart.

We live in wonderful times.

---

# Roadmap

Potential future improvements, depending on how far this joke gets out of control:

- [ ] HTML detection report
- [ ] built-in preview player
- [ ] `Keep` / `Reject` buttons
- [ ] automatic threshold calibration
- [ ] custom classifier trained from accepted/rejected detections
- [ ] statistics
- [ ] detection timeline
- [ ] parallel processing
- [ ] GPU acceleration
- [ ] Docker support
- [ ] proper CLI package
- [ ] support additional scientifically significant noises
- [ ] leaderboard for longest burp
- [ ] enterprise edition with SSO and SOC 2 compliance

The last two may not happen.

No promises.

---

# Example

```powershell
python .\tootscan.py "E:\Medal\Clips" `
  --output ".\gas" `
  --fart-threshold 0.30 `
  --burp-threshold 0.30
```

Possible output:

```text
══════════════════════════════════════
          TOOTSCAN COMPLETE
══════════════════════════════════════

Videos analyzed:       1,842
Farts detected:           37
Burps detected:           21
Total gas events:         58

Scientific progress: immeasurable
══════════════════════════════════════
```

---

# FAQ

### Is this a joke?

Yes.

Also no.

It genuinely works.

---

### Why not use an LLM?

Because asking a 200-billion-parameter language model whether someone farted would be an impressive waste of electricity.

YAMNet is actually designed for audio classification.

Use the right tool for the job.

Even when the job is stupid.

---

### Does it work with videos that aren't from Medal?

Yes.

Despite the original script being built for a Medal clip collection, there is nothing fundamentally Medal-specific about the detection pipeline.

If FFmpeg can read it, TootScan can probably analyze it.

---

### Can it detect multiple farts in the same video?

Of course.

We would never ship such a critical system without multi-fart support.

---

### Can it distinguish between people?

No.

TootScan identifies the event, not the perpetrator.

Attribution remains a human responsibility.

---

### Is the confidence score proof that someone farted?

No.

Please do not use TootScan results in court.

---

# Contributing

Pull requests are welcome.

Especially if they:

- improve detection accuracy;
- improve performance;
- make installation easier;
- reduce false positives;
- unnecessarily professionalize the project.

Bug reports are also welcome.

Please include useful information such as:

```text
OS
Python version
FFmpeg version
TootScan arguments
Relevant error output
```

A detailed description of the fart itself is optional.

---

# Acknowledgements

Built using:

- **TensorFlow**
- **TensorFlow Hub**
- **YAMNet**
- **FFmpeg**
- **AudioSet**

And several decisions that seemed increasingly reasonable as development continued.

---

# Disclaimer

TootScan is provided as-is.

No guarantees are made regarding:

- detection accuracy;
- fart authenticity;
- burp attribution;
- relationship damage resulting from archived evidence;
- existential questions caused by discovering exactly how many hours of your life contain detectable flatulence.

Use responsibly.

---

## License

Choose whatever license fits your project.

For a simple permissive open-source release, **MIT** is probably the easiest choice.

---

<div align="center">

### 💨 TootScan

**Finding the moments that truly matter.**

`YAMNet + FFmpeg + bad ideas`

</div>
