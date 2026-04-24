import argparse
import time

import serial


def parse_line(line: str):
    raw = line.strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) == 3:
        try:
            ra, la, ll = map(float, parts)
            return ra, la, ll
        except ValueError:
            return None
    if len(parts) == 4:
        try:
            _, ra, la, ll = map(float, parts)
            return ra, la, ll
        except ValueError:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ESP32 serial stream format for 3-lead ECG (RA,LA,LL)."
    )
    parser.add_argument("--port", required=True, help="Serial COM port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Capture duration to compute parse quality",
    )
    parser.add_argument(
        "--max-abs-input",
        type=float,
        default=1e7,
        help="Mark parsed sample invalid if abs(RA|LA|LL) exceeds this",
    )
    parser.add_argument(
        "--show-valid",
        type=int,
        default=10,
        help="Print first N valid parsed triples",
    )
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    print(f"Reading {args.port} @ {args.baud} for {args.seconds:.1f}s ...")

    total = 0
    parsed = 0
    in_range = 0
    shown = 0
    ra_min = la_min = ll_min = float("inf")
    ra_max = la_max = ll_max = float("-inf")

    t_end = time.time() + args.seconds
    try:
        while time.time() < t_end:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            total += 1
            values = parse_line(line)
            if values is None:
                continue
            parsed += 1
            ra, la, ll = values
            if (
                abs(ra) <= args.max_abs_input
                and abs(la) <= args.max_abs_input
                and abs(ll) <= args.max_abs_input
            ):
                in_range += 1
                ra_min, ra_max = min(ra_min, ra), max(ra_max, ra)
                la_min, la_max = min(la_min, la), max(la_max, la)
                ll_min, ll_max = min(ll_min, ll), max(ll_max, ll)
                if shown < args.show_valid:
                    shown += 1
                    print(f"valid[{shown:02d}] RA={ra:.2f}, LA={la:.2f}, LL={ll:.2f}")
    finally:
        ser.close()

    parse_pct = (100.0 * parsed / total) if total else 0.0
    range_pct = (100.0 * in_range / parsed) if parsed else 0.0
    print("\n=== Serial Validation Summary ===")
    print(f"Total non-empty lines: {total}")
    print(f"Parsed (3 or 4 numeric fields): {parsed} ({parse_pct:.1f}%)")
    print(f"In range (|value| <= {args.max_abs_input:g}): {in_range} ({range_pct:.1f}%)")
    if in_range > 0:
        print(f"RA range: [{ra_min:.2f}, {ra_max:.2f}]")
        print(f"LA range: [{la_min:.2f}, {la_max:.2f}]")
        print(f"LL range: [{ll_min:.2f}, {ll_max:.2f}]")
    else:
        print("No valid in-range samples. Check ESP32 print format and scaling.")


if __name__ == "__main__":
    main()
