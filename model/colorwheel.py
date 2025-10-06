#!/usr/bin/env python3
"""
Multi-device LED show controller for ESP32-C6 NeoPixel sketches (USB-CDC).

- Controls multiple serial devices simultaneously (each running your echo/HEX/RGB/HUE firmware).
- Ships with multiple predefined patterns:
    police, ambulance, firetruck, firework, tv, candle, rainbow, breathe, party, solid, hue-rotate

Usage examples:
  # Show available ports
  python multi_led_show.py --list

  # Run 'police' on two devices
  python multi_led_show.py -p COM5 -p COM7 --pattern police

  # Different patterns per device (left -> police, right -> tv)
  python multi_led_show.py -p COM5:police -p COM7:tv

  # Linux/mac: use /dev/ttyACM* or /dev/ttyUSB*
  python3 multi_led_show.py -p /dev/ttyACM0 -p /dev/ttyACM1 --pattern rainbow --period 0.02

  # Solid color (RGB) with brightness
  python multi_led_show.py -p COM6 --pattern solid --rgb 255,80,0 --brightness 120

  # Breathe with custom color and slower tempo
  python multi_led_show.py -p COM6 --pattern breathe --rgb 0,128,255 --period 0.03

Install:
  pip install pyserial
"""

import argparse
import sys
import time
import math
import random
import threading
import signal
from typing import Generator, Tuple, Dict, List, Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("This program needs pyserial. Install it via:\n  pip install pyserial")
    sys.exit(1)

# ---------------------- Serial helpers ----------------------

def open_serial(port: str, baud: int = 115200, timeout: float = 0.2) -> serial.Serial:
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = timeout
    ser.write_timeout = 1
    # Avoid auto-reset on connect
    ser.dtr = False
    ser.rts = False
    ser.open()
    # try:
    #     ser.setDTR(False)
    #     ser.setRTS(False)
    # except Exception:
    #     pass
    time.sleep(0.25)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser

def send_line(ser: serial.Serial, s: str):
    data = (s + "\r\n").encode("utf-8")
    ser.write(data)
    ser.flush()

def port_reader(ser: serial.Serial, stop: Dict[str, bool], tag: str):
    # prints device output (echo / status) quietly
    while not stop["stop"]:
        try:
            line = ser.readline()
            if line:
                out = line.decode("utf-8", errors="replace").rstrip()
                # Uncomment to see device chatter:
                # print(f"[{tag}] {out}")
        except Exception:
            time.sleep(0.05)

# ---------------------- Pattern framework ----------------------
# Each pattern is a generator yielding (command_string, delay_seconds)
# The runner for each port will read from its generator and send commands accordingly.

PatternGen = Generator[Tuple[str, float], None, None]

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v

def parse_rgb_arg(rgb: Optional[str]) -> Optional[Tuple[int,int,int]]:
    if not rgb: return None
    parts = [p.strip() for p in rgb.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError("RGB must be r,g,b with 0..255 each")
    r,g,b = [clamp(int(x), 0, 255) for x in parts]
    return (r,g,b)

# --- Utility pulses / strobes ---
def strobe_cmds(rgb: Tuple[int,int,int], on_ms: int, off_ms: int, repeat: int) -> List[Tuple[str, float]]:
    r,g,b = rgb
    out = []
    for _ in range(repeat):
        out.append((f"RGB {r},{g},{b}", on_ms/1000.0))
        out.append(("RGB 0,0,0", off_ms/1000.0))
    return out

# ---------------------- Patterns ----------------------

def pattern_solid(rgb: Tuple[int,int,int], period: float=0.1) -> PatternGen:
    """Hold one color (still sends periodically so device stays 'alive')."""
    r,g,b = rgb
    while True:
        yield (f"RGB {r},{g},{b}", period)

def pattern_hue_rotate(step: int=2, period: float=0.02, sat: int=100, val: int=100) -> PatternGen:
    h = 0
    while True:
        yield (f"H {h},{sat},{val}", period)
        h += step
        if h >= 360: h -= 360

def pattern_rainbow(period: float=0.02) -> PatternGen:
    """Smooth rainbow (alias of hue_rotate with step=2)."""
    return pattern_hue_rotate(step=2, period=period, sat=100, val=100)

def pattern_breathe(rgb: Tuple[int,int,int], period: float=0.02, seconds_per_breath: float=3.0) -> PatternGen:
    """Sinusoidal brightness breathing of a fixed color."""
    r,g,b = rgb
    t = 0.0
    while True:
        # brightness curve 0..255: (sin wave normalized)
        # Use 0.2..1.0 multiplier to avoid going fully off
        phase = (t % seconds_per_breath) / seconds_per_breath
        # 0..1
        x = 0.5 * (1.0 + math.sin(2 * math.pi * (phase - 0.25)))
        mult = 0.2 + 0.8 * x  # keep some floor
        rr = int(clamp(round(r * mult), 0, 255))
        gg = int(clamp(round(g * mult), 0, 255))
        bb = int(clamp(round(b * mult), 0, 255))
        yield (f"RGB {rr},{gg},{bb}", period)
        t += period

def pattern_police(period: float=0.02) -> PatternGen:
    """Blue/Red emergency strobe alternating."""
    blue = (0,0,255)
    red  = (255,0,0)
    seq = []
    seq += strobe_cmds(blue, 60, 60, 3)  # BBB
    seq += [( "OFF", 0.08 )]
    seq += strobe_cmds(red, 60, 60, 3)   # RRR
    seq += [( "OFF", 0.08 )]
    i = 0
    while True:
        cmd, d = seq[i % len(seq)]
        if cmd == "OFF":
            yield ("RGB 0,0,0", d)
        else:
            yield (cmd, d)
        i += 1

def pattern_ambulance(period: float=0.02) -> PatternGen:
    """Red–White–Blue chase, repeating."""
    colors = [(255,0,0), (255,255,255), (0,0,255)]
    idx = 0
    while True:
        r,g,b = colors[idx]
        yield (f"RGB {r},{g},{b}", 0.12)
        yield ("RGB 0,0,0", 0.06)
        idx = (idx + 1) % len(colors)

def pattern_firetruck(period: float=0.02) -> PatternGen:
    """Strong red strobe with brief off frames."""
    seq = strobe_cmds((255,0,0), 70, 50, 6) + [("RGB 0,0,0", 0.2)]
    i = 0
    while True:
        cmd, d = seq[i % len(seq)]
        yield (cmd, d)
        i += 1

def pattern_firework(period: float=0.02) -> PatternGen:
    """Bursts of bright colors that fade quickly."""
    # This uses HSV bursts to simulate fireworks
    while True:
        # pick burst color
        h = random.randrange(0, 360)
        # flash up
        for v in (100, 80, 60):
            yield (f"H {h},100,{v}", 0.04)
        # flicker decay
        for v in (50, 40, 30, 25, 20, 15, 10):
            jitter = random.choice([-5, 0, 5])
            hh = (h + jitter) % 360
            yield (f"H {hh},100,{v}", 0.05)
        # off gap
        yield ("RGB 0,0,0", 0.12)

def pattern_tv(period: float=0.02) -> PatternGen:
    """Random, coolish flicker like a TV in a dark room."""
    while True:
        h = random.choice([200, 210, 220, 230, 240, 250])     # blue-ish hues
        s = random.choice([50, 60, 70, 80])
        v = random.choice([30, 40, 50, 60, 70, 80])
        hold = random.choice([0.03, 0.04, 0.05, 0.07, 0.10, 0.12])
        yield (f"H {h},{s},{v}", hold)
        # occasional bright pop
        if random.random() < 0.10:
            v2 = clamp(v + random.randint(10, 30), 0, 100)
            yield (f"H {h},{s},{v2}", 0.03)

def pattern_candle(period: float=0.02) -> PatternGen:
    """Warm candle flame flicker."""
    base_h = 40  # warm hue
    while True:
        h = base_h + random.randint(-3, 3)
        s = 90 + random.randint(-5, 5)
        v = 60 + random.randint(-15, 10)
        yield (f"H {h},{clamp(s,0,100)},{clamp(v,0,100)}", random.uniform(0.03, 0.08))

def pattern_party(period: float=0.02) -> PatternGen:
    """Quickly changing bright colors."""
    while True:
        h = random.randrange(0, 360)
        s = random.choice([80, 90, 100])
        v = random.choice([80, 90, 100])
        yield (f"H {h},{s},{v}", random.choice([0.05, 0.06, 0.07, 0.08]))

# Map names to factories; some accept kwargs (rgb, period, etc.)
def build_pattern(name: str,
                  period: float,
                  rgb: Optional[Tuple[int,int,int]],
                  hue_step: int,
                  sat: int,
                  val: int,
                  breathe_seconds: float) -> PatternGen:
    key = name.lower()
    if key == "solid":
        if not rgb: rgb = (255, 255, 255)
        return pattern_solid(rgb, period=0.5)
    if key in ("hue", "hue-rotate", "rainbow"):
        step = hue_step if key != "rainbow" else 2
        return pattern_hue_rotate(step=step, period=period, sat=sat, val=val)
    if key == "breathe":
        if not rgb: rgb = (0, 120, 255)
        return pattern_breathe(rgb, period=period, seconds_per_breath=breathe_seconds)
    if key == "police":     return pattern_police(period)
    if key == "ambulance":  return pattern_ambulance(period)
    if key == "firetruck":  return pattern_firetruck(period)
    if key == "firework":   return pattern_firework(period)
    if key == "tv":         return pattern_tv(period)
    if key == "candle":     return pattern_candle(period)
    if key == "party":      return pattern_party(period)
    raise ValueError(f"Unknown pattern: {name}")

# ---------------------- Runner per device ----------------------

def runner(port: str,
           pattern: PatternGen,
           brightness: Optional[int],
           stop: Dict[str, bool],
           banner: bool = True):
    try:
        ser = open_serial(port)
    except Exception as e:
        print(f"[{port}] ERROR opening: {e}")
        return
    tag = port.rsplit("/", 1)[-1]
    rd_stop = {"stop": False}
    t = threading.Thread(target=port_reader, args=(ser, rd_stop, tag), daemon=True)
    t.start()

    try:
        if banner:
            # optional greeting
            try:
                send_line(ser, "HELP")
                time.sleep(0.05)
            except Exception:
                pass

        if brightness is not None:
            b = clamp(int(brightness), 0, 255)
            send_line(ser, f"BRI={b}")
            time.sleep(0.05)

        for cmd, delay_s in pattern:
            if stop["stop"]: break
            try:
                # normalize OFF -> RGB 0,0,0 for compatibility
                if cmd.upper() == "OFF":
                    cmd = "RGB 0,0,0"
                send_line(ser, cmd)
            except Exception as e:
                print(f"[{port}] write error: {e}")
                break
            time.sleep(max(0.0, float(delay_s)))
    finally:
        rd_stop["stop"] = True
        try:
            send_line(ser, "RGB 0,0,0")
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
        # print(f"[{port}] closed.")

# ---------------------- CLI & main ----------------------

def list_all_ports():
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    for p in ports:
        print(f"{p.device:20s}  {p.description}")

def parse_port_and_pattern(arg: str, default_pattern: str) -> Tuple[str, str]:
    """
    Accept either:
      - "COM5" -> (COM5, default_pattern)
      - "COM5:police" -> (COM5, police)
    """
    if ":" in arg:
        port, pat = arg.split(":", 1)
        return port.strip(), pat.strip()
    return arg.strip(), default_pattern

def main():
    ap = argparse.ArgumentParser(description="Multi-device LED show for ESP32 NeoPixel controllers.")
    ap.add_argument("--list", action="store_true", help="List serial ports and exit.")
    ap.add_argument("-p", "--port", action="append", default=[],
                    help="Serial port (e.g., COM3 or /dev/ttyACM0). "
                         "You can also specify per-device pattern with 'PORT:PATTERN'. "
                         "Repeat -p for multiple devices.")
    ap.add_argument("--pattern", default="rainbow",
                    help="Pattern name (default: rainbow). "
                         "Choices: solid, hue, hue-rotate, rainbow, breathe, police, ambulance, firetruck, firework, tv, candle, party")
    ap.add_argument("--rgb", default=None, help="RGB for solid/breathe, e.g., 255,80,0")
    ap.add_argument("--brightness", type=int, default=None, help="NeoPixel brightness 0..255 (sent once at start).")
    ap.add_argument("--period", type=float, default=0.02, help="Base period/tempo for applicable patterns.")
    ap.add_argument("--hue-step", type=int, default=2, help="Hue step for hue/hue-rotate.")
    ap.add_argument("--sat", type=int, default=100, help="Saturation for hue/hue-rotate 0..100.")
    ap.add_argument("--val", type=int, default=100, help="Value (HSV V) for hue/hue-rotate 0..100.")
    ap.add_argument("--breathe-seconds", type=float, default=3.0, help="Seconds per breath for breathe pattern.")
    args = ap.parse_args()

    if args.list:
        list_all_ports()
        return

    if not args.port:
        print("No ports given. Use -p PORT or --list to see available ports.")
        sys.exit(2)

    # Prepare device configs (port, pattern name)
    devices: List[Tuple[str, str]] = []
    for p in args.port:
        devices.append(parse_port_and_pattern(p, args.pattern))

    # Common options
    rgb = parse_rgb_arg(args.rgb) if args.rgb else None
    sat = clamp(args.sat, 0, 100)
    val = clamp(args.val, 0, 100)
    hue_step = clamp(args.hue_step, 1, 60)
    period = max(0.001, float(args.period))

    stop = {"stop": False}
    threads = []

    # Graceful stop on Ctrl+C
    def sigint(_sig, _frm):
        print("\n[!] Stopping...")
        stop["stop"] = True
    signal.signal(signal.SIGINT, sigint)

    # Launch a runner per device
    for (port, pat_name) in devices:
        try:
            pat = build_pattern(pat_name, period, rgb, hue_step, sat, val, args.breathe_seconds)
        except Exception as e:
            print(f"[{port}] pattern error: {e}")
            continue
        th = threading.Thread(target=runner, args=(port, pat, args.brightness, stop), daemon=True)
        th.start()
        threads.append(th)
        print(f"[+] {port}: running pattern '{pat_name}'")

    if not threads:
        print("No runnable devices.")
        sys.exit(1)

    # Wait until interrupted
    try:
        while any(th.is_alive() for th in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop["stop"] = True
        for th in threads:
            th.join(timeout=1.0)
        print("[i] All done. Lights off.")
        time.sleep(0.2)

if __name__ == "__main__":
    main()
