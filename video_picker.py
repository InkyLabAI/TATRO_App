"""Native desktop file picker for a locally hosted app (Windows, WSL, Linux)."""
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def is_wsl():
    return bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))


def _run(command):
    kwargs = {"capture_output": True, "text": True, "encoding": "utf-8"}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    result = subprocess.run(command, **kwargs)
    if result.returncode:
        raise RuntimeError("Could not open the desktop file picker. "
                           "On Linux, install python3-tk and use a desktop session. "
                           "On WSL, enable Windows interoperability and make powershell.exe available. "
                           "You can also paste the full MP4 path and press Enter.")
    return result.stdout.strip()


def choose_video(initial_path=""):
    """Return an original local path, or None when the user cancels.

    Dialogs run in a child process so GUI event loops never run inside a Dash
    worker thread. WSL uses the Windows dialog, then translates its result.
    """
    initial = Path(initial_path or Path.cwd()).expanduser()
    directory = initial if initial.is_dir() else initial.parent
    wsl = is_wsl()
    if os.name == "nt" or wsl:
        powershell = shutil.which("powershell.exe")
        if not powershell and wsl:
            candidate = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
            powershell = str(candidate) if candidate.is_file() else None
        if not powershell:
            raise RuntimeError("Windows PowerShell is unavailable. Paste the full MP4 path and press Enter.")
        initial_directory = _run(["wslpath", "-w", str(directory.resolve())]) if wsl else str(directory.resolve())
        encoded_directory = base64.b64encode(initial_directory.encode("utf-8")).decode("ascii")
        script = """
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.Opacity = 0
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Select your MP4 video'
$dialog.Filter = 'MP4 videos (*.mp4)|*.mp4'
$dialog.Multiselect = $false
$dialog.CheckFileExists = $true
$dialog.InitialDirectory = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__DIRECTORY__'))
try {
    $owner.Show()
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
        [Console]::Write([Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($dialog.FileName)))
    }
} finally {
    $dialog.Dispose()
    $owner.Dispose()
}
""".replace("__DIRECTORY__", encoded_directory)
        encoded_script = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        output = _run([powershell, "-NoProfile", "-STA", "-NonInteractive", "-WindowStyle", "Hidden",
                       "-EncodedCommand", encoded_script])
        if not output:
            return None
        path = base64.b64decode(output, validate=True).decode("utf-8")
        return Path(_run(["wslpath", "-u", path]) if wsl else path)
    script = """
import json, sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
try:
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(parent=root, title='Select your MP4 video',
        initialdir=sys.argv[1], filetypes=[('MP4 videos', '*.mp4')])
    print(json.dumps(path))
finally:
    root.destroy()
"""
    path = json.loads(_run([sys.executable, "-c", script, str(directory.resolve())]))
    return Path(path) if path else None
