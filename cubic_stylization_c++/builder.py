# Builds the C++ server (src/) into bin/ with CMake.
#
# Used by the add-on preferences' "Build Server" button, and runnable from a
# terminal with any Python 3:  python builder.py
#
# Needs CMake >= 3.18 and a C++17 compiler (Visual Studio 2019+ on Windows,
# Xcode command line tools on macOS, gcc/clang on Linux). Eigen is used from
# an installed package when CMake finds one, else downloaded on first build.
# No bpy imports here.

import os
import shutil
import subprocess
import sys

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ADDON_DIR, "src")
BUILD_DIR = os.path.join(ADDON_DIR, "build")
BIN_DIR = os.path.join(ADDON_DIR, "bin")


def _windows_vs_cmake():
    """CMake bundled with Visual Studio (not on PATH outside a dev prompt)."""
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        return None
    try:
        out = subprocess.run(
            [vswhere, "-latest", "-products", "*", "-property", "installationPath"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            creationflags=_NO_WINDOW).stdout
    except OSError:
        return None
    for root in out.splitlines():
        exe = os.path.join(root.strip(), "Common7", "IDE", "CommonExtensions",
                           "Microsoft", "CMake", "CMake", "bin", "cmake.exe")
        if os.path.isfile(exe):
            return exe
    return None


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def find_cmake():
    exe = shutil.which("cmake")
    if exe:
        return exe
    if sys.platform.startswith("win"):
        return _windows_vs_cmake()
    for cand in ("/opt/homebrew/bin/cmake", "/usr/local/bin/cmake",
                 "/Applications/CMake.app/Contents/bin/cmake"):
        if os.path.isfile(cand):
            return cand
    return None


def build_commands(cmake):
    return [
        [cmake, "-S", SRC_DIR, "-B", BUILD_DIR, "-DCMAKE_BUILD_TYPE=Release"],
        [cmake, "--build", BUILD_DIR, "--config", "Release", "--parallel"],
    ]


def build(log=print):
    """Configure + build. Returns (ok, message); full output goes to log()."""
    cmake = find_cmake()
    if cmake is None:
        return False, ("CMake not found — install CMake (and a C++ compiler) "
                       "or build manually, see README")
    for cmd in build_commands(cmake):
        log("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  creationflags=_NO_WINDOW)
        except OSError as exc:
            return False, f"could not run CMake: {exc}"
        log(proc.stdout)
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").strip().splitlines()[-3:])
            return False, "build failed — see the system console. " + tail
    return True, "server built into " + BIN_DIR


def package(zip_path=None):
    """Zip the add-on for Blender's Install dialog: Python files, bin/ and
    the C++ source (so Build Server works), without build/ or tests/."""
    import zipfile
    name = os.path.basename(ADDON_DIR)
    zip_path = zip_path or os.path.join(os.path.dirname(ADDON_DIR), name + ".zip")
    skip = {"build", "tests", "__pycache__"}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ADDON_DIR):
            dirs[:] = sorted(d for d in dirs if d not in skip and not d.startswith("."))
            for f in sorted(files):
                if f.endswith(".pyc") or f.startswith("."):
                    continue
                full = os.path.join(root, f)
                zf.write(full, os.path.join(name, os.path.relpath(full, ADDON_DIR)))
    return zip_path


if __name__ == "__main__":
    # python builder.py          build the server into bin/
    # python builder.py --zip    build, then package the add-on zip
    ok, msg = build()
    print(msg)
    if ok and "--zip" in sys.argv[1:]:
        print("packaged", package())
    sys.exit(0 if ok else 1)
