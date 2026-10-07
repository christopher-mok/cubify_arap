# Client for the C++ solver process (cubify_server, built from src/).
#
# Blender starts a single long-lived server on first use and talks to it over
# stdin/stdout pipes (wire format documented in src/protocol.h). Every mesh
# gets a server-side session holding its prefactorized system, wrapped here
# by RemoteStylizer, which mirrors the Python add-on's CubicStylizer API:
# .pins, .lam, .solve(), .run() — plus .close() to free the session.
#
# No bpy imports: this module can be used (and tested) outside Blender.

import os
import struct
import subprocess
import sys
import threading

import numpy as np

PROTOCOL_VERSION = 4

# TargetShape codes (src/target_shape.h)
TARGETS = ("CUBE", "OCTAHEDRON", "PYRAMID", "HEX_COLUMN", "ROUNDED_CUBE")

(OP_HELLO, OP_CREATE, OP_SOLVE, OP_SET_LAMBDA, OP_SET_THREADS, OP_DESTROY, OP_SHUTDOWN,
 OP_FIX_THIN_WALLS) = range(8)
RESP_OK, RESP_ERROR, RESP_PROGRESS = range(3)

FLAG_PIN_POS = 1 << 0
FLAG_V_INIT = 1 << 1
FLAG_PROGRESS = 1 << 2
FLAG_WARM_LAST = 1 << 4

_HEADER = struct.Struct("<IIQ")

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))

# A locally built `cubify_server` wins over the per-platform binary bundled
# in the all-in-one zip (cubify_server-macos is a universal2 build).
if sys.platform.startswith("win"):
    _EXE_NAMES = ("cubify_server.exe",)
elif sys.platform == "darwin":
    _EXE_NAMES = ("cubify_server", "cubify_server-macos")
else:
    _EXE_NAMES = ("cubify_server", "cubify_server-linux")
EXE_NAME = _EXE_NAMES[0]


class ServerError(RuntimeError):
    pass


def _first_existing(directory):
    for name in _EXE_NAMES:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def default_server_path():
    bindir = os.path.join(ADDON_DIR, "bin")
    return _first_existing(bindir) or os.path.join(bindir, EXE_NAME)


def resolve_server_path(override=""):
    """The executable to launch: the preferences override, else bin/."""
    if override:
        path = os.path.abspath(os.path.expanduser(override))
        if os.path.isdir(path):
            path = _first_existing(path) or os.path.join(path, EXE_NAME)
        return path
    return default_server_path()


# ================== server process

class Server:
    """One cubify_server process. Requests are strictly sequential; a lock
    makes the object safe to share between threads."""

    def __init__(self, path):
        if not os.path.isfile(path):
            raise ServerError(f"C++ server not found at {path} — build it from "
                              "the add-on preferences (Build Server)")
        if not sys.platform.startswith("win") and not os.access(path, os.X_OK):
            # Blender installs add-ons with python's zipfile, which drops
            # the unix exec bit — restore it
            try:
                os.chmod(path, 0o755)
            except OSError:
                pass
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen(
                [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                creationflags=flags)
        except OSError as exc:
            raise ServerError(f"could not start the C++ server ({exc})") from exc
        self.path = path
        self._lock = threading.Lock()
        try:
            payload = self.request(OP_HELLO)
        except ServerError:
            self.kill()
            raise
        self.protocol, self.hardware_threads = struct.unpack_from("<ii", payload)
        self.version = bytes(payload[8:]).decode("utf-8", "replace")
        if self.protocol != PROTOCOL_VERSION:
            self.kill()
            raise ServerError(f"server protocol {self.protocol} does not match "
                              f"the add-on's ({PROTOCOL_VERSION}) — rebuild it")

    def alive(self):
        return self.proc.poll() is None

    # ---- framing

    def _write(self, op, parts):
        size = sum(memoryview(p).nbytes for p in parts)
        out = self.proc.stdin
        out.write(_HEADER.pack(op, 0, size))
        for p in parts:
            out.write(memoryview(p).cast("B"))
        out.flush()

    def _read_exact(self, nbytes):
        buf = bytearray(nbytes)
        view = memoryview(buf)
        got = 0
        while got < nbytes:
            k = self.proc.stdout.readinto(view[got:])
            if not k:
                raise ServerError("C++ server exited unexpectedly")
            got += k
        return buf

    def request(self, op, *parts, on_progress=None):
        """Send one request; returns the OK payload (bytearray). PROGRESS
        messages are forwarded to on_progress(done, total)."""
        with self._lock:
            try:
                self._write(op, parts)
                while True:
                    code, _, size = _HEADER.unpack(self._read_exact(_HEADER.size))
                    payload = self._read_exact(size)
                    if code == RESP_PROGRESS:
                        if on_progress is not None:
                            on_progress(*struct.unpack_from("<ii", payload))
                        continue
                    if code == RESP_ERROR:
                        raise ServerError(payload.decode("utf-8", "replace"))
                    return payload
            except (OSError, ValueError) as exc:  # broken pipe / closed file
                self.kill()
                raise ServerError(f"lost connection to the C++ server ({exc})") from exc

    def shutdown(self):
        if self.alive():
            try:
                self.request(OP_SHUTDOWN)
                self.proc.wait(timeout=2.0)
            except Exception:
                pass
        self.kill()

    def kill(self):
        try:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=2.0)
        except Exception:
            pass
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f.close()
            except Exception:
                pass


_SERVER = None
_SERVER_LOCK = threading.Lock()


def get_server(path=None):
    """The shared server, (re)started on demand."""
    global _SERVER
    path = path or default_server_path()
    with _SERVER_LOCK:
        if _SERVER is not None and (not _SERVER.alive() or _SERVER.path != path):
            _SERVER.shutdown()
            _SERVER = None
        if _SERVER is None:
            _SERVER = Server(path)
        return _SERVER


def server_cached():
    """The running server or None, without starting one (for UI code)."""
    s = _SERVER
    return s if s is not None and s.alive() else None


def shutdown_server():
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is not None:
            _SERVER.shutdown()
            _SERVER = None


# ================== stylizer sessions

def _f64(a, rows=None):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return a.reshape(-1, 3) if rows is None else a.reshape(rows, 3)


class RemoteStylizer:
    """Server-side CubicStylizer. Same API as the Python solver's class:

    V : (n, 3) rest-pose positions; F : (m, 3) triangle indices
    cubeness : lambda (0 = classic ARAP); cube_axes : (3, 3) rotation
    pins : vertex indices to constrain; threads : 0 = all cores
    flat_relax : edge-weight factor inside regions already facing a target
                 direction, in (0, 1]; < 1 lets flat parts square their
                 outline (1 = off, the paper's energy)
    target : one of TARGETS (the shape surfaces are stylized toward)
    roundness : 0..1, rounded cube only (0 is nearly the cube)
    """

    def __init__(self, server, V, F, cubeness=0.2, cube_axes=None, pins=None,
                 threads=0, flat_relax=1.0, target="CUBE", roundness=0.5):
        target = str(target).upper()
        if target not in TARGETS:
            raise ValueError(f"unknown target shape {target!r}")
        V = _f64(V)
        F = np.ascontiguousarray(F, dtype=np.int32).reshape(-1, 3)
        A = np.eye(3) if cube_axes is None else np.asarray(cube_axes, dtype=np.float64)
        pins = np.ascontiguousarray([] if pins is None else list(pins), dtype=np.int32)

        self.server = server
        self.n = len(V)
        self._lam = float(cubeness)
        payload = server.request(
            OP_CREATE,
            struct.pack("<iiiiiddd", len(V), len(F), len(pins), int(threads),
                        TARGETS.index(target), self._lam, float(flat_relax),
                        float(roundness)),
            np.ascontiguousarray(A.reshape(3, 3), dtype=np.float64), V, F, pins)
        self.id, kp = struct.unpack_from("<Ii", payload)
        self.pins = np.frombuffer(payload, dtype=np.int32, count=kp, offset=8).astype(np.int64)

    # ---- properties mirroring the Python solver

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, value):
        self._lam = float(value)
        self.server.request(OP_SET_LAMBDA, struct.pack("<Id", self.id, self._lam))

    def set_threads(self, threads):
        self.server.request(OP_SET_THREADS, struct.pack("<Ii", self.id, int(threads)))

    # ---- drivers

    def solve(self, pin_pos=None, V_init=None, iterations=30, admm_iters=100,
              on_progress=None, warm_last=False):
        """Run local-global iterations and return the (n, 3) positions.

        pin_pos : (len(self.pins), 3) targets for the pinned vertices
                  (defaults to their rest positions)
        V_init : warm-start positions (defaults to the rest pose)
        warm_last : warm-start from this session's previous result without
                    re-sending it (ignored when V_init is given)
        """
        flags = 0
        parts = [None]
        if pin_pos is not None and len(self.pins):
            flags |= FLAG_PIN_POS
            parts.append(_f64(pin_pos, len(self.pins)))
        if V_init is not None:
            flags |= FLAG_V_INIT
            parts.append(_f64(V_init, self.n))
        elif warm_last:
            flags |= FLAG_WARM_LAST
        if on_progress is not None:
            flags |= FLAG_PROGRESS
        parts[0] = struct.pack("<IiiI", self.id, int(iterations), int(admm_iters), flags)

        payload = self.server.request(OP_SOLVE, *parts, on_progress=on_progress)
        _, n = struct.unpack_from("<ii", payload)
        return np.frombuffer(payload, dtype=np.float64, count=3 * n, offset=8).reshape(n, 3).copy()

    def run(self, iterations=30, admm_iters=100, on_progress=None, pin_pos=None):
        """One-shot stylization from the rest pose. Pinned vertices are held;
        unpinned parts stay seated on what they touch, or centred where they
        were."""
        return self.solve(pin_pos=pin_pos, iterations=iterations,
                          admm_iters=admm_iters, on_progress=on_progress)

    def close(self):
        """Free the server-side session (safe to call more than once)."""
        if self.id is None:
            return
        sid, self.id = self.id, None
        if self.server.alive():
            try:
                self.server.request(OP_DESTROY, struct.pack("<I", sid))
            except ServerError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def create_stylizer(V, F, cubeness=0.2, cube_axes=None, pins=None, threads=0,
                    server_path=None, flat_relax=1.0, target="CUBE", roundness=0.5):
    """Build a stylizer on the C++ server.

    Returns (stylizer, device_label, warning) like the Python add-on's
    solver.create_stylizer; warning is always None (kept for symmetry).
    """
    server = get_server(server_path)
    s = RemoteStylizer(server, V, F, cubeness=cubeness, cube_axes=cube_axes,
                       pins=pins, threads=threads, flat_relax=flat_relax,
                       target=target, roundness=roundness)
    used = threads if threads and threads > 0 else server.hardware_threads
    return s, f"C++ ({used} thread{'s' if used != 1 else ''})", None


def fix_thin_walls(V_rest, V, F, min_thickness=0.1, max_wall=0.04, iterations=60,
                   threads=0, server_path=None):
    """Push apart thin walls that stylization made cross (or nearly cross).

    V_rest : (n, 3) pose the walls are measured in (before stylization)
    V : (n, 3) stylized positions to fix; F : (m, 3) triangles
    min_thickness : walls are pushed back to at least this fraction of their
                    rest thickness
    max_wall : thickest wall considered, fraction of the bounding-box diagonal

    Returns (V_fixed, stats) with stats keys walls, crossed_before,
    crossed_after, moved, passes, max_move.
    """
    V_rest, V = _f64(V_rest), _f64(V)
    F = np.ascontiguousarray(F, dtype=np.int32).reshape(-1, 3)
    if len(V_rest) != len(V):
        raise ValueError("rest and current vertex counts differ")
    payload = get_server(server_path).request(
        OP_FIX_THIN_WALLS,
        struct.pack("<iiiidd", len(V), len(F), int(iterations), int(threads),
                    float(min_thickness), float(max_wall)),
        V_rest, V, F)
    walls, before, after, moved, passes, n = struct.unpack_from("<6i", payload)
    (max_move,) = struct.unpack_from("<d", payload, 24)
    V_out = np.frombuffer(payload, dtype=np.float64, count=3 * n, offset=32).reshape(n, 3).copy()
    return V_out, dict(walls=walls, crossed_before=before, crossed_after=after,
                       moved=moved, passes=passes, max_move=max_move)
