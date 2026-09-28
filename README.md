# UVLP Software

Desktop control software for a **maskless UV photolithography wafer stepper** (pattern projection + XY/Z stage). The current build runs in **simulation mode** — stage motion and exposure are modeled in software so you can develop the UI and job flow before wiring hardware.

## Requirements

- Python 3.10+
- Windows recommended (multi-monitor DMD output uses OpenCV fullscreen)

```bash
pip install -r requirements.txt
```

## Run

```bash
python desktop/main.py
```

Automated UI smoke test (saves PNGs under `desktop/screenshots/`):

```bash
python desktop/main.py --capture-smoke
```

## What you can do

- Load an exposure pattern image (`pattern.png` loads by default if present)
- Jog / move / home a simulated wafer stage
- Preview grid sites on a circular wafer with edge exclusion
- Run a step-and-expose grid job with progress and cancel
- Save / load settings to `desktop/config.json`
- Fullscreen the pattern onto a selected monitor (DMD / SLM stand-in)

## Layout

```
desktop/
  main.py          # application entry point
  pattern.png      # default test pattern
  calib.jpg        # calibration image
  check.png        # small check pattern
  wallpaper.jpg    # alternate test image
```

## Notes

- No stage or DMD drivers are connected yet; treat Run Grid / Expose as a process simulator.
- Configuration is written next to the app as `desktop/config.json` (gitignored).
