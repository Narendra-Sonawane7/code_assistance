# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Meeting Assistant.
Build with: pyinstaller build.spec

FIX (this version):
  The old spec only listed hiddenimports for PyQt5/speech_recognition/
  pytesseract/mss/keyboard/tavily/PIL. Nothing for soundcard, openai,
  numpy, or pywin32's COM modules — all required by the System Audio
  (Ctrl+Shift+S) feature. That worked on the dev machine only because
  those packages were already fully present/cached there.

  On a different PC, two things can happen silently:
    1. `import soundcard` fails because its compiled CFFI backend wasn't
       bundled (PyInstaller's static import scan doesn't reliably catch
       this — it needs explicit collection).
    2. soundcard imports fine, loopback "works", but the Groq Whisper
       call (via openai -> httpx -> certifi's CA bundle) fails — e.g. an
       SSL cert path issue inside the frozen exe — and that exception was
       previously caught and silently discarded in audio_listener.py.

  Fix: explicitly collect_all() the packages whose native/data files
  PyInstaller tends to miss, and temporarily run with console=True so any
  remaining failure is visible instead of disappearing.
"""

from PyInstaller.utils.hooks import collect_all

datas = [('config.json', '.')]
binaries = []
hiddenimports = [
    'PyQt5',
    'PyQt5.QtWidgets',
    'PyQt5.QtGui',
    'PyQt5.QtCore',
    'speech_recognition',
    'pytesseract',
    'mss',
    'mss.windows',
    'keyboard',
    'tavily',
    'PIL',
    'numpy',
    'pythoncom',
    'pywintypes',
    'win32com',
    'win32com.client',
    'win32timezone',
]

# Packages whose compiled extensions / bundled data files PyInstaller's
# static analysis commonly misses — collect everything for these explicitly.
for pkg in ('soundcard', 'openai', 'certifi', 'httpx', 'httpcore', 'anyio'):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        # If a package isn't installed in this build env, skip it rather
        # than failing the whole build.
        pass

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MeetingAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    # TEMP: True so you can SEE any remaining error on the broken PC.
    # Flip back to False once Ctrl+Shift+S is confirmed working there.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,              # Add .ico path here if you want a custom icon
)
