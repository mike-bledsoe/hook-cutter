# Third-party components

hook-cutter's own code and documentation are MIT licensed (`LICENSE`). This file lists what it
bundles, what it installs, and what it expects you to provide.

## Bundled with the skill

### Fonts (SIL Open Font License 1.1)

Shipped unmodified (not subset, not renamed) with their licence files in `fonts/`. Integrity
hashes are in `fonts/manifest.json`.

| File | Family | Copyright | Licence file |
|---|---|---|---|
| `fonts/ArchivoBlack-Regular.ttf` | Archivo Black | Copyright 2017 The Archivo Black Project Authors (https://github.com/Omnibus-Type/ArchivoBlack) | `fonts/ArchivoBlack-OFL.txt` |
| `fonts/Anton-Regular.ttf` | Anton | Copyright 2020 The Anton Project Authors (https://github.com/googlefonts/AntonFont.git) | `fonts/Anton-OFL.txt` |
| `fonts/Montserrat[wght].ttf` | Montserrat (variable) | Copyright 2024 The Montserrat.Git Project Authors (https://github.com/JulietaUla/Montserrat.git) | `fonts/Montserrat-OFL.txt` |

The OFL permits bundling and redistribution with software. The fonts may not be sold on their
own, and modified versions must not use the original font names.

### Face detection model (MIT)

| File | Model | Copyright | Licence file |
|---|---|---|---|
| `models/face_detection_yunet_2023mar.onnx` | YuNet (2023mar), from the OpenCV Zoo | Copyright (c) 2020 Shiqi Yu | `models/YUNET-LICENSE.txt` |

## Installed by `doctor --install`

Installed into the skill's own venv (in your user data folder) from PyPI. Not redistributed with
the skill.

| Package | Role | Licence |
|---|---|---|
| faster-whisper | speech recognition (default backend) | MIT |
| CTranslate2 | inference engine for faster-whisper | MIT |
| mlx-whisper (`--with mlx`) | speech recognition on Apple Silicon | MIT |
| OpenCV (`opencv-python-headless`) | video analysis, YuNet inference | Apache-2.0 |
| Pillow | text and image rendering | MIT-CMU (HPND) |
| NumPy | numerics | BSD-3-Clause |
| fontTools | glyph coverage checks, variable font instances | MIT |
| yt-dlp (`--with url`) | URL downloads | Unlicense |

Notable transitive dependencies: PyAV (BSD-3-Clause), onnxruntime (MIT), huggingface_hub and
tokenizers (Apache-2.0), tqdm (MPL-2.0 and MIT), certifi (MPL-2.0). The `url` extra also brings
mutagen (GPL-2.0-or-later), brotli, pycryptodomex and websockets. The `mlx` extra also brings mlx
(MIT), PyTorch (BSD-3-Clause), numba (BSD-2-Clause), SciPy (BSD-3-Clause) and tiktoken (MIT).
Check `pip list` inside the venv for the exact set and versions.

GPL components: PyAV and OpenCV wheels bundle FFmpeg builds that include GPL codecs (x264,
x265); the url extra pulls mutagen (GPL-2.0-or-later). These are installed into your venv from
PyPI, not redistributed.

### Speech model weights (downloaded on first use or with `doctor --model`)

The OpenAI Whisper model weights are released under the MIT licence. hook-cutter downloads
converted checkpoints from Hugging Face (CTranslate2 conversions for faster-whisper, MLX
conversions for mlx-whisper); these conversions are published by third parties, generally under
the same MIT terms. Check the model card of the exact model you choose. Weights are stored in
your user data folder, never in the skill folder.

## Provided by you

| Tool | Role | Licence |
|---|---|---|
| FFmpeg and ffprobe (6.0 or newer, with libx264 and AAC) | decoding, cutting, encoding | LGPL-2.1 or GPL-2.0 or later, depending on how your build was configured. Installed by you (for example with Homebrew or apt); hook-cutter calls the binaries and never redistributes them |
| uv (optional) | provisions a compatible Python for the venv | Apache-2.0 or MIT |
| deno or node (optional) | JavaScript runtime yt-dlp uses for YouTube | MIT |
| whisper.cpp `whisper-cli` (optional) | alternative speech recognition backend | MIT |

Output files you render are yours. Rights in the footage you cut belong to whoever owns that
footage; see `references/fidelity.md`.
