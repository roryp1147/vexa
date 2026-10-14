# Vexa

A cyberpunk android-bust music player for Windows whose visor is the visualizer. She floats on your desktop with no window, and her visor, eyes and neon veins react to your music.

![Vexa](https://roryp1147.github.io/img/vexa-spectrum.webp)

## Download

**[Download Vexa for Windows (v1.0.0)](https://github.com/roryp1147/vexa/releases/latest/download/Vexa-1.0.0-win64.zip)** · [Project page](https://roryp1147.github.io/projects/vexa.html)

Windows 10/11 (64-bit). No Python required: extract the zip and run `Vexa.exe`. The app isn't code-signed, so SmartScreen may warn you; click **More info → Run anyway**.

## Run from source

Requires Windows 10/11 and Python 3.14.

```bat
git clone https://github.com/roryp1147/vexa.git
cd vexa
py -3.14 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py
```

`run_vexa.bat` starts it without a console window. Useful options: `--no-boot`, `--on-top`, `--source auto|local|loopback|synth|none`. Setting `VEXA_MUTE=1` silences everything, and `VEXA_NO_SETTINGS=1` ignores saved settings (they normally live in the registry under `HKCU\Software\Vexa`).

## Build the exe (PyInstaller)

```bat
.venv\Scripts\pip install pyinstaller
.venv\Scripts\pyinstaller --noconfirm --clean --onedir --windowed --name Vexa ^
  --icon assets\vexa.ico --add-data "assets;assets" --paths . ^
  --collect-submodules player --collect-submodules winrt main.py
```

The app ends up in `dist\Vexa\`. Zip that folder to share it. You can delete `_internal\PySide6\opengl32sw.dll` to save about 20 MB.

## Developer tools

- `tools/build_layers.py` rebuilds the art layers from the original concept image (not included), and `tools/build_sounds.py` regenerates the UI sounds. These need `pip install -r requirements-tools.txt`.

## License

The **code** is MIT licensed, (c) 2026 Rory. The **art and sound assets** in `assets/` are free for personal use only and may not be redistributed or resold. See [LICENSE](LICENSE) for details. Not affiliated with any company. 
