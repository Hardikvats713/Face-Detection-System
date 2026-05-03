import threading
import platform
import logging
import time

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BEEP_FREQUENCY = 1000   # Hz
BEEP_DURATION  = 500    # ms

# ✅ FIX 1: Track running beep thread to avoid stacking concurrent beeps
#    Original spawns unlimited threads if beep_async() is called rapidly
_beep_lock   = threading.Lock()
_beep_active = False


def _play_beep() -> None:
    """Play a single beep appropriate for the current OS."""
    global _beep_active

    system = platform.system()

    try:
        if system == "Windows":
            import winsound
            winsound.Beep(BEEP_FREQUENCY, BEEP_DURATION)

        elif system == "Darwin":
            # ✅ FIX 2: macOS — \a is unreliable in most terminals;
            #    afplay with a system sound is consistent and audible
            import subprocess
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Ping.aiff"],
                check=False,
                timeout=2.0,
            )

        else:
            # ✅ FIX 3: Linux — try paplay (PulseAudio) first, fall back to
            #    beep command, then \a as last resort
            #    \a alone is silent in most modern Linux terminals
            import subprocess
            played = False

            for cmd in [
                ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                ["beep", "-f", str(BEEP_FREQUENCY), "-l", str(BEEP_DURATION)],
            ]:
                try:
                    result = subprocess.run(
                        cmd, check=True,
                        timeout=2.0,
                        capture_output=True,
                    )
                    played = True
                    break
                except (subprocess.CalledProcessError,
                        FileNotFoundError,
                        subprocess.TimeoutExpired):
                    continue

            if not played:
                # ✅ FIX 4: Log when \a fallback is used so it's not silently ignored
                log.warning("No audio backend found — using terminal bell (\\a). "
                            "Install 'beep' or 'pulseaudio-utils' for reliable audio.")
                print("\a", end="", flush=True)

    except Exception as e:
        # ✅ FIX 5: Never let beep crash the gate — log and swallow all errors
        log.error(f"Beep failed: {e}")

    finally:
        # ✅ FIX 1 applied: Always release the guard so next beep can fire
        with _beep_lock:
            _beep_active = False


def beep_async() -> bool:
    """
    Fire a non-blocking beep on a daemon thread.

    Returns:
        True  if a beep was started.
        False if one is already playing (skipped to avoid overlap).
    """
    global _beep_active

    # ✅ FIX 1 applied: Skip if a beep is already in progress
    with _beep_lock:
        if _beep_active:
            log.debug("Beep skipped — previous beep still playing.")
            return False
        _beep_active = True

    # ✅ FIX 6: Name the thread for easier debugging in thread dumps
    t = threading.Thread(
        target=_play_beep,
        name="beep-worker",
        daemon=True,
    )
    t.start()
    return True