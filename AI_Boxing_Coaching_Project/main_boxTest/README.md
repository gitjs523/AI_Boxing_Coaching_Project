# MoveNet Original + Team3

Separate runnable version of the user-provided boxing.py with Team 3 functions.

## Features
- CLAHE enhancement on LAB brightness channel, toggle with `C` (default OFF for A/B comparison)
- Virtual mitt target, toggle with `G`; alternating left/right; score on motion-gated entry, not static overlap
- `R` saves report JSON and CSV; `Q` quits and saves automatically to `reports/`
- JAB COUNT and original elbow-angle-based detection retained
- Accuracy grade is **UNDEFINED** until the team agrees on S~C scoring rules
- No HSV detection (original MoveNet control variant)

## Run
```powershell
python -m pip install -r requirements.txt
python boxing.py
```

MoveNet ONNX file auto-download is attempted if absent; an internet connection is required for first run. Ensure your ONNX model is a **MoveNet 17-keypoint (y,x,score)** model with compatible 192x192 input. The old fallback URL to UltraFace was intentionally removed because it is not a pose model.

## Integration
Common modules: `preprocessing.py`, `mitt_game.py`, `report.py`. They can be integrated with other teammates' versions by importing these modules. Do not overwrite teammates' `boxing.py` directly; compare and merge changes.

## Test notes
1. Start program: camera opens and MoveNet pose appears.
2. Press C and compare brightness/pose response in low-light conditions.
3. Press G: move the requested hand into target from outside, hit counter should increment once; next target changes side.
4. Hold hand stationary inside target: no repeated hits.
5. Press R: JSON and CSV created; grade remains UNDEFINED.
6. Press Q: camera closes and a report is saved.

**Limitations:** The virtual mitt uses 2D screen proximity and velocity only; no depth/contact detection. CLAHE may increase noise. `guard_warning_frames` counts video frames, not individual mistakes. The S~C accuracy formula is intentionally not invented. HSV thresholds may require on-site calibration.
