"""Per-run FlexNet licence reservation for Lumerical FDTD jobs.

An optimisation run needs its 9 lum_fdtd_solve licences for its whole lifetime,
not just while a solve is in flight.  Between solves the licences are checked
back in, and if a colleague grabs them the next solve dies -- the driver only
retries for ~30 s (see production_polb_calib_gpu5.log, 2026-08-09).

This module reserves the licences for a private FlexNet PROJECT at job start and
releases them at job end:

    import lum_reserve
    lum_reserve.acquire(tag="polsweep_b")   # BEFORE lumapi.FDTD() is created
    ...run the optimisation...
    # released automatically at interpreter exit / SIGTERM / SIGINT

acquire() sets LM_PROJECT in os.environ; the CAD app and fdtd-engine inherit it
and their checkouts then draw from the reservation.  Verified 2026-08-11:
with LM_PROJECT set the reservation count dropped by exactly 9 during a solve,
without it the count did not move and the solve drew from the general pool.

Reservations are the lab's shared capacity -- 9 of 54 licences, i.e. one of six
concurrent solve slots -- so a leaked block hurts everyone until removed.
Every block records the owning PID and host, and acquire() sweeps blocks whose
process is gone before adding its own, so a hard crash self-heals on the next
run.  Use `python lum_reserve.py sweep` to force a cleanup by hand.
"""

from __future__ import annotations

import atexit
import datetime
import fcntl
import os
import re
import signal
import socket
import subprocess
import sys
import time

# Site configuration.  Every value can be overridden from the environment or by
# calling configure(); the offline test-suite points them at a temporary options
# file and a stub lmutil, which is the only way to exercise the rollback,
# contention and wait paths without disturbing the shared licence server.
OPT_FILE = os.environ.get("MSOPT_RESERVE_OPT_FILE",
                          "/home/eidl/ansys_license/ansyslmd.opt")
LOCK_FILE = os.environ.get("MSOPT_RESERVE_LOCK_FILE",
                           "/home/eidl/ansys_license/.msopt_reserve.lock")
LMUTIL = os.environ.get("MSOPT_RESERVE_LMUTIL",
                        "/opt/lumerical/v261/licensingclient/linx64/lmutil")
SERVER = os.environ.get("MSOPT_RESERVE_SERVER", "1055@localhost")
FEATURE = os.environ.get("MSOPT_RESERVE_FEATURE", "lum_fdtd_solve")
LICENCES_PER_SOLVE = int(os.environ.get("MSOPT_RESERVE_COUNT", "9"))
WAIT_POLL_S = float(os.environ.get("MSOPT_RESERVE_WAIT_POLL_S", "15"))
# ansyslmd applies a reread asynchronously -- the RESERVING line has been seen
# up to 9 s after lmutil returned -- so the reservation must be polled for,
# never checked once.
VERIFY_TIMEOUT_S = float(os.environ.get("MSOPT_RESERVE_VERIFY_TIMEOUT_S", "45"))
VERIFY_POLL_S = float(os.environ.get("MSOPT_RESERVE_VERIFY_POLL_S", "2"))
ALLOW_SUDO = os.environ.get("MSOPT_RESERVE_ALLOW_SUDO", "1") == "1"
MARKER = "msopt-reserve"        # only blocks carrying this marker are ever touched
# First component of an auto-generated project name; a label for humans reading
# lmstat, with no meaning to FlexNet.  Unlike MARKER it is safe to change.
PREFIX = os.environ.get("MSOPT_RESERVE_PREFIX", "PROJECT")


def configure(**kw) -> None:
    """Override site configuration (opt_file, lock_file, lmutil, server,
    feature, licences_per_solve, wait_poll_s, allow_sudo).  Intended for tests
    and for sites whose paths differ."""
    mapping = {"opt_file": "OPT_FILE", "lock_file": "LOCK_FILE",
               "lmutil": "LMUTIL", "server": "SERVER", "feature": "FEATURE",
               "licences_per_solve": "LICENCES_PER_SOLVE",
               "wait_poll_s": "WAIT_POLL_S", "allow_sudo": "ALLOW_SUDO",
               "verify_timeout_s": "VERIFY_TIMEOUT_S",
               "verify_poll_s": "VERIFY_POLL_S", "prefix": "PREFIX"}
    for key, value in kw.items():
        if key not in mapping:
            raise TypeError("unknown configure() key: %s" % key)
        globals()[mapping[key]] = value

_HELD: dict[str, int] = {}      # project -> count, for cleanup
_HANDLERS_INSTALLED = False


# --------------------------------------------------------------------------
# licence server queries
# --------------------------------------------------------------------------
def _lmstat() -> str:
    return subprocess.run([LMUTIL, "lmstat", "-c", SERVER, "-a"],
                          capture_output=True, text=True, timeout=120).stdout


def _inventory_expdate() -> str | None:
    """Return the expiry of the largest FEATURE tranche in the licence file.

    ``lmstat -a`` prints a feature's expiry only while that tranche has a
    checkout/reservation.  At an idle server that left ``expdate`` unset and
    made a reservation use ``EXPDATE=permanent``.  This site has only four
    permanent FDTD solves plus a 50-seat dated tranche, so a nine-seat GPU
    reservation could never be satisfied by the former.  ``lmstat -i`` lists
    both tranches even while idle; selecting the largest is the usable pool.
    """
    out = subprocess.run(
        [LMUTIL, "lmstat", "-c", SERVER, "-i", FEATURE],
        capture_output=True, text=True, timeout=120,
    ).stdout
    choices = []
    for line in out.splitlines():
        m = re.match(
            r"\s*%s\s+\S+\s+(\d+)\s+\S+\s+(\S+)" % re.escape(FEATURE),
            line,
        )
        if m:
            expiry = m.group(2)
            if expiry.startswith("permanent"):
                expiry = "permanent"
            choices.append((int(m.group(1)), expiry))
    return max(choices, default=(0, None))[1]


def feature_status(raw: str | None = None) -> dict:
    """Parse the lum_fdtd_solve section of `lmstat -a`.

    `in_use` counts idle reservations too, so free = issued - in_use is the
    number a new reservation can actually take.
    """
    out = raw if raw is not None else _lmstat()
    issued = in_use = 0
    expdate = None
    reservations: dict[str, int] = {}
    grabbing = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Users of "):
            grabbing = s.startswith("Users of %s:" % FEATURE)
            if grabbing:
                # FlexNet switches to the singular noun when exactly one
                # licence is in use ("1 license in use").  Accept both forms;
                # otherwise the whole status line is missed and a healthy
                # pool is incorrectly reported as 0 issued / 0 free.
                m = re.search(
                    r"Total of (\d+) licenses? issued;\s*"
                    r"Total of (\d+) licenses? in use",
                    s,
                )
                if m:
                    issued, in_use = int(m.group(1)), int(m.group(2))
            continue
        if not grabbing:
            continue
        if s.startswith('"%s"' % FEATURE):
            m = re.search(r"expiry:\s*(\S+)", s)
            if m:
                expdate = m.group(1)
        m = re.match(r"(\d+) RESERVATIONs? for PROJECT (\S+)", s)
        if m:
            reservations[m.group(2)] = int(m.group(1))
    if expdate is None and raw is None and issued > 0:
        expdate = _inventory_expdate()
    return {"issued": issued, "in_use": in_use, "free": issued - in_use,
            "expdate": expdate, "reservations": reservations}


def _lmreread() -> bool:
    r = subprocess.run([LMUTIL, "lmreread", "-c", SERVER, "-vendor", "ansyslmd"],
                       capture_output=True, text=True, timeout=120)
    ok = "lmreread successful" in (r.stdout + r.stderr)
    if not ok:
        print("[lic  ] lmreread failed: %s" % (r.stdout + r.stderr).strip()[:300])
    return ok


# --------------------------------------------------------------------------
# options file I/O
# --------------------------------------------------------------------------
def _read_opt() -> str:
    with open(OPT_FILE) as fh:
        return fh.read()


def _permission_help() -> str:
    return (
        "cannot write %s.\nFix with ONE of:\n"
        "  sudo chmod g+w %s        # lab members edit it directly -- simplest\n"
        "  <a NOPASSWD sudoers rule for tee on that path>\n"
        "NOTE: an interactive `sudo -n true` can succeed purely from sudo's cached\n"
        "credential (tty_tickets) and still fail inside a job, which has no such\n"
        "tty.  Unattended runs need one of the two fixes above."
        % (OPT_FILE, OPT_FILE))


def writable() -> tuple[bool, str]:
    """Can this process rewrite the options file without a password prompt?"""
    if os.access(OPT_FILE, os.W_OK):
        return True, "direct (group-writable)"
    if not ALLOW_SUDO:
        return False, "not writable and the sudo fallback is disabled"
    r = subprocess.run(["sudo", "-n", "true"], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return True, "sudo -n"
    return False, (r.stderr or "").strip()[:120]


def _write_opt(text: str) -> None:
    """Write the options file, preserving its ownership.

    Direct in-place write when the file is group-writable, otherwise
    `sudo -n tee`.  Never renames a temp file over it -- that would silently
    reassign the file to whoever ran the job.
    """
    if os.access(OPT_FILE, os.W_OK):
        with open(OPT_FILE, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        return
    if not ALLOW_SUDO:
        raise RuntimeError(_permission_help())
    r = subprocess.run(["sudo", "-n", "tee", OPT_FILE],
                       input=text, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError("%s\nsudo said: %s"
                           % (_permission_help(), (r.stderr or "").strip()[:200]))


_LOCK_STATE: dict = {"fh": None, "depth": 0}


def _open_lock():
    """Open the shared lock file, keeping it writable by the whole group.

    The lock only works if every lab member can take it, but a file created
    under the usual umask comes out 0644 and locks everyone else out of it --
    whoever ran first would own the lock for good.
    """
    fd = os.open(LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        st = os.fstat(fd)
        if st.st_uid == os.getuid() and st.st_mode & 0o060 != 0o060:
            os.fchmod(fd, 0o664)
    except OSError:
        pass                        # someone else owns it and has already fixed it
    return os.fdopen(fd, "r+")


class _Lock:
    """Cross-process lock so concurrent jobs cannot clobber each other's blocks.

    Re-entrant within a process: flock() is tied to the open file description,
    so a nested `with _Lock()` on a second descriptor would block against our
    own lock instead of passing through (acquire() -> _rollback() does exactly
    that).  Depth counting keeps the outermost holder in charge.
    """

    def __enter__(self):
        if _LOCK_STATE["depth"] > 0:
            _LOCK_STATE["depth"] += 1
            return self
        fh = _open_lock()
        for _ in range(120):
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                _LOCK_STATE["fh"], _LOCK_STATE["depth"] = fh, 1
                return self
            except OSError:
                time.sleep(0.5)
        fh.close()
        raise RuntimeError("timed out waiting for %s" % LOCK_FILE)

    def __exit__(self, *exc):
        _LOCK_STATE["depth"] -= 1
        if _LOCK_STATE["depth"] > 0:
            return
        fh, _LOCK_STATE["fh"] = _LOCK_STATE["fh"], None
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


# --------------------------------------------------------------------------
# block manipulation
# --------------------------------------------------------------------------
def _begin_re(project: str = r"(\S+)") -> re.Pattern:
    return re.compile(r"#\s*BEGIN %s %s\b" % (MARKER, project))


def _parse_blocks(text: str) -> list[dict]:
    blocks, cur = [], None
    for i, line in enumerate(text.splitlines()):
        m = _begin_re().match(line.strip())
        if m:
            cur = {"project": m.group(1), "start": i, "pid": None, "host": None}
            mp = re.search(r"pid=(\d+)", line)
            mh = re.search(r"host=(\S+)", line)
            cur["pid"] = int(mp.group(1)) if mp else None
            cur["host"] = mh.group(1) if mh else None
            continue
        if cur and re.match(r"#\s*END %s %s\b" % (MARKER, re.escape(cur["project"])), line.strip()):
            cur["end"] = i
            blocks.append(cur)
            cur = None
    return blocks


def _strip_block(text: str, project: str) -> str:
    keep, skip = [], False
    for line in text.splitlines():
        s = line.strip()
        if _begin_re(re.escape(project)).match(s):
            skip = True
            continue
        if skip and re.match(r"#\s*END %s %s\b" % (MARKER, re.escape(project)), s):
            skip = False
            continue
        if not skip:
            keep.append(line)
    return "\n".join(keep).rstrip("\n") + "\n"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:      # exists, owned by someone else
        return True


def sweep_stale(verbose: bool = True) -> list[str]:
    """Drop reservation blocks whose owning process is gone on this host."""
    removed = []
    with _Lock():
        text = _read_opt()
        for blk in _parse_blocks(text):
            if blk["host"] != socket.gethostname() or blk["pid"] is None:
                continue          # another machine's block: cannot judge
            if _pid_alive(blk["pid"]):
                continue
            text = _strip_block(text, blk["project"])
            removed.append(blk["project"])
        if removed:
            _write_opt(text)
            _lmreread()
            if verbose:
                print("[lic  ] swept stale reservations: %s" % ", ".join(removed))
    return removed


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def acquire(count: int = LICENCES_PER_SOLVE, tag: str = "", project: str | None = None,
            wait_s: float = 0.0, verbose: bool = True) -> str:
    """Reserve `count` licences for a private project and export LM_PROJECT.

    Must be called BEFORE the lumapi session is created, so the CAD app and the
    solver inherit LM_PROJECT.  Returns the project name.

    wait_s > 0 polls until the pool has `count` free licences -- reservations
    can only take licences that are free right now, so starting while the pool
    is exhausted would reserve nothing.
    """
    ok, how = writable()
    if not ok:                    # fail before waiting on licences we cannot reserve
        raise RuntimeError("%s\nsudo said: %s" % (_permission_help(), how))

    if project is None:
        bits = [PREFIX, os.environ.get("USER", "user")]
        if tag:
            bits.append(re.sub(r"[^A-Za-z0-9_]+", "_", tag))
        bits.append(str(os.getpid()))
        project = "_".join(bits)

    sweep_stale(verbose=verbose)

    deadline = time.time() + max(0.0, wait_s)
    while True:
        st = feature_status()
        if st["free"] >= count:
            break
        if time.time() >= deadline:
            raise RuntimeError(
                "only %d of %d %s licences are free; need %d to reserve. "
                "Start when the pool has room, or pass wait_s=<seconds>."
                % (st["free"], st["issued"], FEATURE, count))
        if verbose:
            print("[lic  ] waiting for %d free licences (have %d) ..."
                  % (count, st["free"]))
        time.sleep(WAIT_POLL_S)

    # feature_status() reads the expiry off `lmstat -a`, which lists EVERY
    # tranche that has activity and hands back whichever it saw last.  On this
    # site that is the 10-seat 25-aug-2026 tranche, not the 50-seat 14-may-2027
    # one, so a second 9-seat reservation fails with "1 of 9" while 21 licences
    # sit free in the big tranche (measured 2026-08-20).  _inventory_expdate()
    # exists to pick the largest pool -- use it, and let the environment pin a
    # tranche explicitly when that guess is wrong too.
    expdate = os.environ.get("MSOPT_RESERVE_EXPDATE", "").strip()
    if not expdate:
        try:
            expdate = _inventory_expdate() or ""
        except Exception as exc:
            if verbose:
                print("[lic  ] inventory expdate lookup failed (%s)" % exc)
            expdate = ""
    expdate = expdate or st["expdate"] or "permanent"
    if verbose:
        print("[lic  ] reserving %d %s from tranche EXPDATE=%s"
              % (count, FEATURE, expdate))
    stamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    block = (
        "# BEGIN %s %s pid=%d host=%s user=%s started=%s\n"
        "RESERVE %d %s:EXPDATE=%s PROJECT %s\n"
        "# END %s %s\n"
        % (MARKER, project, os.getpid(), socket.gethostname(),
           os.environ.get("USER", "?"), stamp,
           count, FEATURE, expdate, project, MARKER, project))

    with _Lock():
        text = _read_opt()
        if _begin_re(re.escape(project)).search(text):
            if verbose:
                print("[lic  ] reservation for %s already present" % project)
        else:
            _write_opt(text.rstrip("\n") + "\n" + block)
            if not _lmreread():
                _rollback(project)
                raise RuntimeError("lmreread failed while reserving %s" % project)

    verify_deadline = time.time() + VERIFY_TIMEOUT_S
    got = 0
    while True:
        got = feature_status()["reservations"].get(project, 0)
        if got >= count or time.time() >= verify_deadline:
            break
        time.sleep(VERIFY_POLL_S)
    if got < count:
        _rollback(project)
        raise RuntimeError(
            "reservation for %s did not take effect within %.0f s (server reports "
            "%d of %d). The options file was rolled back."
            % (project, VERIFY_TIMEOUT_S, got, count))

    os.environ["LM_PROJECT"] = project
    _HELD[project] = count
    _install_handlers()
    if verbose:
        print("[lic  ] reserved %d %s for PROJECT %s; LM_PROJECT exported"
              % (count, FEATURE, project))
    return project


def release(project: str | None = None, verbose: bool = True) -> None:
    """Remove the reservation block(s) and let the daemon reread."""
    targets = [project] if project else list(_HELD)
    if not targets:
        return
    with _Lock():
        text = _read_opt()
        changed = False
        for proj in targets:
            if _begin_re(re.escape(proj)).search(text):
                text = _strip_block(text, proj)
                changed = True
        if changed:
            _write_opt(text)
            _lmreread()
    for proj in targets:
        _HELD.pop(proj, None)
        if os.environ.get("LM_PROJECT") == proj:
            os.environ.pop("LM_PROJECT", None)
    if changed and verbose:
        print("[lic  ] released reservation(s): %s" % ", ".join(targets))


def _rollback(project: str) -> None:
    try:
        with _Lock():
            text = _read_opt()
            if _begin_re(re.escape(project)).search(text):
                _write_opt(_strip_block(text, project))
                _lmreread()
    except Exception as exc:                                  # noqa: BLE001
        print("[lic  ] WARNING: rollback of %s failed: %s" % (project, exc))


def _install_handlers() -> None:
    global _HANDLERS_INSTALLED
    if _HANDLERS_INSTALLED:
        return
    atexit.register(lambda: release(verbose=True))

    def _handler(signum, frame):
        release(verbose=True)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass                    # not on the main thread / not supported
    _HANDLERS_INSTALLED = True


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _cmd_status() -> int:
    st = feature_status()
    print("%s: %d issued, %d in use (incl. reservations), %d free, expiry %s"
          % (FEATURE, st["issued"], st["in_use"], st["free"], st["expdate"]))
    if st["reservations"]:
        for proj, n in sorted(st["reservations"].items()):
            print("  reserved %2d for PROJECT %s" % (n, proj))
    else:
        print("  no reservations")
    print("--- %s ---" % OPT_FILE)
    print(_read_opt().rstrip())
    ok, how = writable()
    print("--- writable by this process: %s (%s) ---"
          % ("yes" if ok else "NO", how))
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "status":
        return _cmd_status()
    if cmd == "sweep":
        removed = sweep_stale()
        if not removed:
            print("[lic  ] nothing stale to sweep")
        return 0
    if cmd == "acquire":
        tag = argv[2] if len(argv) > 2 else ""
        proj = acquire(tag=tag)
        print(proj)
        print("NOTE: this process exits now, so the reservation is released "
              "again.  Call acquire() from inside the job instead.")
        return 0
    if cmd == "release":
        if len(argv) < 3:
            print("usage: lum_reserve.py release <project>")
            return 2
        _HELD[argv[2]] = 0
        release(argv[2])
        return 0
    print(__doc__)
    print("commands: status | sweep | acquire [tag] | release <project>")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
