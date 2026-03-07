#!/usr/bin/env python3
"""
Batch ERA5 pressure-level downloader.

Why this script:
- CDS API often rejects very large one-shot requests.
- This tool splits requests by month or by day, with retries and resumable behavior.
"""

from __future__ import annotations

import argparse
import calendar
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


DATASET = "reanalysis-era5-pressure-levels"
DEFAULT_VARIABLES = [
    "geopotential",
    "relative_humidity",
    "specific_humidity",
    "temperature",
]
DEFAULT_PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10",
    "20", "30", "50", "70", "100", "125",
    "150", "175", "200", "225", "250", "300",
    "350", "400", "450", "500", "550", "600",
    "650", "700", "750", "775", "800", "825",
    "850", "875", "900", "925", "950", "975",
    "1000",
]
DEFAULT_TIMES = [f"{h:02d}:00" for h in range(24)]


@dataclass(frozen=True)
class Chunk:
    year: int
    month: int
    days: List[int]

    def key(self) -> str:
        if len(self.days) == 1:
            return f"{self.year:04d}{self.month:02d}{self.days[0]:02d}"
        return f"{self.year:04d}{self.month:02d}"


def parse_range_list(text: str, min_value: int, max_value: int) -> List[int]:
    out = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            items = token.split("-", 1)
            if len(items) != 2:
                raise ValueError(f"invalid range token: {token}")
            start = int(items[0].strip())
            end = int(items[1].strip())
            if start > end:
                start, end = end, start
            for val in range(start, end + 1):
                if val < min_value or val > max_value:
                    raise ValueError(
                        f"value {val} out of range [{min_value}, {max_value}]"
                    )
                out.add(val)
        else:
            val = int(token)
            if val < min_value or val > max_value:
                raise ValueError(f"value {val} out of range [{min_value}, {max_value}]")
            out.add(val)
    if not out:
        raise ValueError("range list is empty")
    return sorted(out)


def parse_area(text: str) -> List[float]:
    vals = [v.strip() for v in text.split(",")]
    if len(vals) != 4:
        raise ValueError("--area must contain exactly 4 numbers: N,W,S,E")
    return [float(v) for v in vals]


def build_chunks(year: int, months: Sequence[int], chunk_mode: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    for month in months:
        _, n_days = calendar.monthrange(year, month)
        if chunk_mode == "month":
            chunks.append(Chunk(year=year, month=month, days=list(range(1, n_days + 1))))
        else:
            for day in range(1, n_days + 1):
                chunks.append(Chunk(year=year, month=month, days=[day]))
    return chunks


def build_request(
    chunk: Chunk,
    variables: Sequence[str],
    pressure_levels: Sequence[str],
    area: Sequence[float],
    data_format: str,
    download_format: str,
) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": list(variables),
        "year": [f"{chunk.year:04d}"],
        "month": [f"{chunk.month:02d}"],
        "day": [f"{d:02d}" for d in chunk.days],
        "time": list(DEFAULT_TIMES),
        "pressure_level": list(pressure_levels),
        "data_format": data_format,
        "download_format": download_format,
        "area": list(area),
    }


def target_file_name(chunk: Chunk, data_format: str, download_format: str) -> str:
    stem = f"era5_pressure_levels_{chunk.key()}"
    if download_format == "zip":
        return f"{stem}.zip"
    if data_format == "netcdf":
        return f"{stem}.nc"
    return f"{stem}.grib"


def retrieve_with_retry(
    client,
    dataset: str,
    request: dict,
    target_path: Path,
    retries: int,
    retry_wait_sec: float,
) -> None:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            client.retrieve(dataset, request, str(target_path))
            return
        except Exception as exc:  # pragma: no cover - runtime/network path
            last_err = exc
            if attempt < retries:
                wait_sec = retry_wait_sec * attempt
                print(
                    f"[retry] {target_path.name} failed on attempt {attempt}/{retries}: {exc}"
                )
                print(f"[retry] sleeping {wait_sec:.1f}s")
                time.sleep(wait_sec)
    raise RuntimeError(f"download failed after {retries} attempts: {last_err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch download ERA5 pressure-level data via CDS API."
    )
    parser.add_argument("--year", type=int, default=2025, help="Target year")
    parser.add_argument(
        "--months",
        type=str,
        default="1-12",
        help='Months list/range, e.g. "1-12" or "1,2,3,10-12"',
    )
    parser.add_argument(
        "--chunk-mode",
        type=str,
        choices=["month", "day"],
        default="month",
        help="Split requests by month or by day",
    )
    parser.add_argument(
        "--area",
        type=str,
        default="40,106,32,112",
        help="N,W,S,E (lat/lon) comma-separated",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/l1_space/data/era5_pressure_levels",
        help="Directory for downloaded files",
    )
    parser.add_argument(
        "--data-format",
        type=str,
        choices=["netcdf", "grib"],
        default="netcdf",
        help="CDS data_format",
    )
    parser.add_argument(
        "--download-format",
        type=str,
        choices=["zip", "unarchived"],
        default="zip",
        help="CDS download_format",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for each request",
    )
    parser.add_argument(
        "--retry-wait-sec",
        type=float,
        default=20.0,
        help="Base backoff seconds (multiplied by attempt index)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: skip existing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned requests without downloading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    months = parse_range_list(args.months, min_value=1, max_value=12)
    area = parse_area(args.area)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks(args.year, months, args.chunk_mode)
    total = len(chunks)
    print(
        f"[plan] year={args.year} months={months} chunk_mode={args.chunk_mode} "
        f"total_requests={total}"
    )
    print(f"[plan] area(N,W,S,E)={area}")
    print(f"[plan] output_dir={out_dir}")

    manifest_path = out_dir / f"manifest_{args.year}_{args.chunk_mode}.jsonl"
    if args.dry_run:
        client = None
    else:
        try:
            import cdsapi  # type: ignore
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "cdsapi is not installed. Install it first: pip install cdsapi"
            ) from exc
        client = cdsapi.Client()

    done = 0
    skipped = 0
    failed = 0
    start_t = time.time()

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for idx, chunk in enumerate(chunks, start=1):
            request = build_request(
                chunk=chunk,
                variables=DEFAULT_VARIABLES,
                pressure_levels=DEFAULT_PRESSURE_LEVELS,
                area=area,
                data_format=args.data_format,
                download_format=args.download_format,
            )
            target = out_dir / target_file_name(
                chunk=chunk,
                data_format=args.data_format,
                download_format=args.download_format,
            )

            if target.exists() and not args.overwrite:
                skipped += 1
                done += 1
                print(f"[{idx}/{total}] skip existing: {target.name}")
                manifest.write(
                    json.dumps(
                        {
                            "chunk": chunk.key(),
                            "target": str(target),
                            "status": "skipped_existing",
                            "timestamp_unix": int(time.time()),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                continue

            print(f"[{idx}/{total}] downloading: {target.name}")
            if args.dry_run:
                done += 1
                manifest.write(
                    json.dumps(
                        {
                            "chunk": chunk.key(),
                            "target": str(target),
                            "status": "dry_run",
                            "request": request,
                            "timestamp_unix": int(time.time()),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                continue

            try:
                retrieve_with_retry(
                    client=client,  # type: ignore[arg-type]
                    dataset=DATASET,
                    request=request,
                    target_path=target,
                    retries=max(args.retries, 1),
                    retry_wait_sec=max(args.retry_wait_sec, 0.0),
                )
                status = "ok"
                print(f"[{idx}/{total}] done: {target.name}")
            except Exception as exc:  # pragma: no cover - runtime/network path
                status = "failed"
                failed += 1
                print(f"[{idx}/{total}] failed: {target.name} | {exc}")

            done += 1
            elapsed = time.time() - start_t
            rate = done / max(elapsed, 1e-6)
            remain = total - done
            eta_sec = int(remain / max(rate, 1e-9))
            print(
                f"[progress] {done}/{total} | skipped={skipped} failed={failed} "
                f"| rate={rate:.2f} req/s | eta={eta_sec}s"
            )

            manifest.write(
                json.dumps(
                    {
                        "chunk": chunk.key(),
                        "target": str(target),
                        "status": status,
                        "timestamp_unix": int(time.time()),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    print(
        f"[summary] total={total} done={done} skipped={skipped} failed={failed} "
        f"manifest={manifest_path}"
    )
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
