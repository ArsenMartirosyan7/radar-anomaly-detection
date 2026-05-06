from __future__ import annotations

import argparse
import csv
import math
import struct
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


KNOWN_CATS = [1, 2, 34, 48]

NUMERIC_CAT048_FIELDS = [
    "rho_nm",
    "theta_deg",
    "flight_level",
    "x_nm",
    "y_nm",
    "ground_speed_nm_s",
    "heading_deg",
]


class RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.s = 0.0
        self.ss = 0.0
        self.min_v = None
        self.max_v = None

    def add(self, value: Optional[float]) -> None:
        if value is None:
            return
        try:
            x = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(x):
            return

        self.n += 1
        self.s += x
        self.ss += x * x
        self.min_v = x if self.min_v is None else min(self.min_v, x)
        self.max_v = x if self.max_v is None else max(self.max_v, x)

    def mean(self) -> float:
        return self.s / self.n if self.n else 0.0

    def std(self) -> float:
        if self.n <= 1:
            return 0.0
        mean = self.s / self.n
        var = max((self.ss / self.n) - mean * mean, 0.0)
        return math.sqrt(var)

    def min(self) -> float:
        return 0.0 if self.min_v is None else float(self.min_v)

    def max(self) -> float:
        return 0.0 if self.max_v is None else float(self.max_v)


class SecondBucket:
    def __init__(self) -> None:
        self.packet_count = 0
        self.udp_byte_count = 0
        self.asterix_block_count = 0
        self.cat_counts = defaultdict(int)
        self.other_cat_count = 0
        self.cat048_record_count = 0
        self.cat048_parse_error_count = 0
        self.src_ports = set()
        self.dst_ports = set()
        self.track_numbers = set()
        self.aircraft_addresses = set()

        self.block_len = RunningStats()
        self.cat048_len = RunningStats()
        self.stats = {name: RunningStats() for name in NUMERIC_CAT048_FIELDS}

    def add_packet(self, src_port: int, dst_port: int, payload_len: int) -> None:
        self.packet_count += 1
        self.udp_byte_count += payload_len
        self.src_ports.add(src_port)
        self.dst_ports.add(dst_port)

    def add_block(self, cat: int, block_len: int) -> None:
        self.asterix_block_count += 1
        self.block_len.add(block_len)

        if cat in KNOWN_CATS:
            self.cat_counts[cat] += 1
        else:
            self.other_cat_count += 1

        if cat == 48:
            self.cat048_len.add(block_len)

    def add_cat048_record(self, rec: Dict[str, object]) -> None:
        self.cat048_record_count += 1

        for name in NUMERIC_CAT048_FIELDS:
            self.stats[name].add(rec.get(name))

        if rec.get("track_number") is not None:
            self.track_numbers.add(rec["track_number"])

        if rec.get("aircraft_address"):
            self.aircraft_addresses.add(rec["aircraft_address"])

    def as_row(self, file_name: str, second: int) -> Dict[str, object]:
        row: Dict[str, object] = {
            "file": file_name,
            "epoch_second": second,
            "utc_time": datetime.fromtimestamp(second, tz=timezone.utc).isoformat(),
            "packet_count": self.packet_count,
            "udp_byte_count": self.udp_byte_count,
            "asterix_block_count": self.asterix_block_count,
            "other_cat_count": self.other_cat_count,
            "unique_src_ports": len(self.src_ports),
            "unique_dst_ports": len(self.dst_ports),
            "unique_tracks": len(self.track_numbers),
            "unique_aircraft": len(self.aircraft_addresses),
            "cat048_record_count": self.cat048_record_count,
            "cat048_parse_error_count": self.cat048_parse_error_count,
            "block_len_mean": self.block_len.mean(),
            "block_len_std": self.block_len.std(),
            "block_len_min": self.block_len.min(),
            "block_len_max": self.block_len.max(),
            "cat048_len_mean": self.cat048_len.mean(),
            "cat048_len_std": self.cat048_len.std(),
            "cat048_len_min": self.cat048_len.min(),
            "cat048_len_max": self.cat048_len.max(),
        }

        for cat in KNOWN_CATS:
            row[f"cat{cat:03d}_count"] = self.cat_counts[cat]

        for name, st in self.stats.items():
            row[f"{name}_mean"] = st.mean()
            row[f"{name}_std"] = st.std()
            row[f"{name}_min"] = st.min()
            row[f"{name}_max"] = st.max()

        return row


def iter_udp_payloads_from_pcap(path: Path) -> Iterator[Tuple[float, int, int, bytes]]:
    """
    Yield:
        timestamp_seconds, src_port, dst_port, udp_payload

    This supports standard Ethernet IPv4 PCAP files.
    """
    with path.open("rb") as f:
        global_header = f.read(24)

        if len(global_header) != 24:
            raise ValueError(f"Invalid PCAP file: {path}")

        magic = global_header[:4]

        if magic == b"\xd4\xc3\xb2\xa1":
            endian = "<"
            ts_resolution = 1_000_000
        elif magic == b"\xa1\xb2\xc3\xd4":
            endian = ">"
            ts_resolution = 1_000_000
        elif magic == b"\x4d\x3c\xb2\xa1":
            endian = "<"
            ts_resolution = 1_000_000_000
        elif magic == b"\xa1\xb2\x3c\x4d":
            endian = ">"
            ts_resolution = 1_000_000_000
        else:
            raise ValueError(f"Unsupported PCAP magic number {magic.hex()} in {path}")

        _, _, _, _, _, linktype = struct.unpack(endian + "HHIIII", global_header[4:24])

        if linktype != 1:
            raise ValueError(
                f"Only Ethernet PCAP linktype=1 is supported. Found linktype={linktype}"
            )

        while True:
            packet_header = f.read(16)

            if not packet_header:
                break

            if len(packet_header) != 16:
                break

            ts_sec, ts_frac, incl_len, _orig_len = struct.unpack(
                endian + "IIII", packet_header
            )

            packet = f.read(incl_len)

            if len(packet) != incl_len:
                break

            if len(packet) < 42:
                continue

            eth_type = struct.unpack("!H", packet[12:14])[0]

            if eth_type != 0x0800:
                continue

            ip_start = 14
            version_ihl = packet[ip_start]
            version = version_ihl >> 4
            ihl = (version_ihl & 0x0F) * 4

            if version != 4 or len(packet) < ip_start + ihl + 8:
                continue

            protocol = packet[ip_start + 9]

            if protocol != 17:
                continue

            udp_start = ip_start + ihl
            src_port, dst_port, udp_len, _checksum = struct.unpack(
                "!HHHH", packet[udp_start : udp_start + 8]
            )

            if udp_len < 8 or len(packet) < udp_start + udp_len:
                continue

            payload = packet[udp_start + 8 : udp_start + udp_len]
            timestamp = ts_sec + ts_frac / ts_resolution

            yield timestamp, src_port, dst_port, payload


def split_asterix_blocks(payload: bytes) -> List[Tuple[int, bytes]]:
    """
    Split one UDP payload into ASTERIX blocks.

    ASTERIX block format:
        CAT: 1 byte
        LEN: 2 bytes
        data records
    """
    blocks: List[Tuple[int, bytes]] = []
    pos = 0
    n = len(payload)

    while pos + 3 <= n:
        cat = payload[pos]
        length = int.from_bytes(payload[pos + 1 : pos + 3], byteorder="big")

        if length < 3 or pos + length > n:
            break

        blocks.append((cat, payload[pos : pos + length]))
        pos += length

    return blocks


def read_fspec(data: bytes, pos: int) -> Tuple[List[int], int]:
    fields: List[int] = []
    frn = 1

    while pos < len(data):
        b = data[pos]
        pos += 1

        for mask in (0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02):
            if b & mask:
                fields.append(frn)
            frn += 1

        if not (b & 0x01):
            break

    return fields, pos


def skip_fx_item(data: bytes, pos: int) -> int:
    while pos < len(data):
        b = data[pos]
        pos += 1

        if not (b & 0x01):
            break

    return pos


def skip_sp_or_re(data: bytes, pos: int) -> int:
    if pos >= len(data):
        return pos

    length = data[pos]

    if length <= 0:
        return pos + 1

    return min(len(data), pos + length)


def skip_i048_130(data: bytes, pos: int) -> int:
    """
    Best-effort skip for I048/130 Radar Plot Characteristics.
    """
    if pos >= len(data):
        return pos

    spec = data[pos]
    pos += 1

    common_lengths = (1, 1, 1, 2, 1, 1, 1)

    for mask, length in zip(
        (0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02), common_lengths
    ):
        if spec & mask:
            pos += length

    while spec & 0x01 and pos < len(data):
        spec = data[pos]
        pos += 1

        for mask in (0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02):
            if spec & mask:
                pos += 1

    return min(pos, len(data))


def skip_i048_120(data: bytes, pos: int) -> int:
    """
    Best-effort skip for I048/120 Radial Doppler Speed.
    """
    if pos >= len(data):
        return pos

    spec = data[pos]
    pos += 1

    if spec & 0x80:
        pos += 2

    if spec & 0x40 and pos < len(data):
        rep = data[pos]
        pos += 1 + rep * 2

    while spec & 0x01 and pos < len(data):
        spec = data[pos]
        pos += 1

        for mask in (0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02):
            if spec & mask:
                pos += 1

    return min(pos, len(data))


def u16(data: bytes) -> int:
    return int.from_bytes(data, byteorder="big", signed=False)


def i16(data: bytes) -> int:
    return int.from_bytes(data, byteorder="big", signed=True)


def signed_14_bit(value: int) -> int:
    value &= 0x3FFF

    if value & 0x2000:
        value -= 0x4000

    return value


def parse_cat048_block(block: bytes) -> Tuple[List[Dict[str, object]], int]:
    """
    Decode useful CAT048 fields for anomaly-detection features.

    Returns:
        records, parse_error_count
    """
    data = block[3:]
    pos = 0
    records: List[Dict[str, object]] = []
    errors = 0

    fixed_lengths = {
        17: 2,
        18: 4,
        19: 2,
        21: 2,
        22: 7,
        23: 1,
        24: 2,
        25: 1,
        26: 2,
    }

    while pos < len(data):
        start = pos

        try:
            fields, pos = read_fspec(data, pos)

            if not fields:
                break

            rec: Dict[str, object] = {}

            for frn in fields:
                if pos >= len(data):
                    raise ValueError("short CAT048 record")

                if frn == 1:
                    rec["sac"] = data[pos]
                    rec["sic"] = data[pos + 1]
                    pos += 2

                elif frn == 2:
                    rec["time_of_day_sec"] = (
                        int.from_bytes(data[pos : pos + 3], "big") / 128.0
                    )
                    pos += 3

                elif frn == 3:
                    pos = skip_fx_item(data, pos)

                elif frn == 4:
                    rec["rho_nm"] = u16(data[pos : pos + 2]) / 256.0
                    rec["theta_deg"] = (
                        u16(data[pos + 2 : pos + 4]) * 360.0 / 65536.0
                    )
                    pos += 4

                elif frn == 5:
                    pos += 2

                elif frn == 6:
                    rec["flight_level"] = (
                        signed_14_bit(u16(data[pos : pos + 2])) / 4.0
                    )
                    pos += 2

                elif frn == 7:
                    pos = skip_i048_130(data, pos)

                elif frn == 8:
                    rec["aircraft_address"] = data[pos : pos + 3].hex().upper()
                    pos += 3

                elif frn == 9:
                    rec["aircraft_id_raw"] = data[pos : pos + 6].hex().upper()
                    pos += 6

                elif frn == 10:
                    rep = data[pos]
                    pos += 1 + rep * 8

                elif frn == 11:
                    rec["track_number"] = u16(data[pos : pos + 2]) & 0x0FFF
                    pos += 2

                elif frn == 12:
                    rec["x_nm"] = i16(data[pos : pos + 2]) / 128.0
                    rec["y_nm"] = i16(data[pos + 2 : pos + 4]) / 128.0
                    pos += 4

                elif frn == 13:
                    rec["ground_speed_nm_s"] = (
                        u16(data[pos : pos + 2]) * (2**-14)
                    )
                    rec["heading_deg"] = (
                        u16(data[pos + 2 : pos + 4]) * 360.0 / 65536.0
                    )
                    pos += 4

                elif frn == 14:
                    pos = skip_fx_item(data, pos)

                elif frn == 15:
                    pos += 4

                elif frn == 16:
                    pos = skip_fx_item(data, pos)

                elif frn == 20:
                    pos = skip_i048_120(data, pos)

                elif frn in fixed_lengths:
                    pos += fixed_lengths[frn]

                elif frn in (27, 28):
                    pos = skip_sp_or_re(data, pos)

                else:
                    raise ValueError(f"unsupported CAT048 FRN {frn}")

            if pos <= start:
                break

            records.append(rec)

        except Exception:
            errors += 1
            break

    return records, errors


def process_one_pcap(path: Path, output_csv: Path, only_port: Optional[int]) -> None:
    buckets: Dict[int, SecondBucket] = defaultdict(SecondBucket)

    packet_total = 0
    block_total = 0
    cat048_records_total = 0

    for timestamp, src_port, dst_port, payload in iter_udp_payloads_from_pcap(path):
        if only_port is not None and src_port != only_port and dst_port != only_port:
            continue

        second = int(timestamp)
        bucket = buckets[second]
        bucket.add_packet(src_port, dst_port, len(payload))
        packet_total += 1

        blocks = split_asterix_blocks(payload)

        for cat, block in blocks:
            block_total += 1
            bucket.add_block(cat, len(block))

            if cat == 48:
                records, errors = parse_cat048_block(block)
                bucket.cat048_parse_error_count += errors

                for rec in records:
                    bucket.add_cat048_record(rec)

                cat048_records_total += len(records)

    rows = [buckets[second].as_row(path.name, second) for second in sorted(buckets)]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError(f"No UDP/ASTERIX data extracted from {path}")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"OK: {path.name}")
    print(f"  output: {output_csv}")
    print(f"  seconds: {len(rows)}")
    print(f"  UDP packets used: {packet_total}")
    print(f"  ASTERIX blocks: {block_total}")
    print(f"  CAT048 records decoded: {cat048_records_total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract second-level radar features from ASTERIX PCAP files."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_pcaps",
        help="Input PCAP file or directory",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/decoded",
        help="Output directory for CSV files",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8600,
        help="UDP port to keep. Use -1 to keep all UDP packets.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    only_port = None if args.port == -1 else args.port

    if input_path.is_file():
        pcap_files = [input_path]
    else:
        pcap_files = sorted(
            list(input_path.glob("*.pcap")) + list(input_path.glob("*.pcapng"))
        )

    if not pcap_files:
        raise FileNotFoundError(f"No .pcap or .pcapng files found in {input_path}")

    for pcap_file in pcap_files:
        out_csv = output_dir / f"{pcap_file.stem}_features.csv"
        process_one_pcap(pcap_file, out_csv, only_port)


if __name__ == "__main__":
    main()