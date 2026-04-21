from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.download import download_chbmit, get_chbmit_data_dir, print_chbmit_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the full CHB-MIT dataset from PhysioNet.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=get_chbmit_data_dir(),
        help="Destination directory for CHB-MIT files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject IDs to download, e.g. chb01 chb02. Default is the full dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_chbmit_info()
    print(f"\nDownloading to: {args.dest}")

    result = download_chbmit(
        dest_dir=args.dest,
        force=args.force,
        subjects=args.subjects,
        download_all_files=True,
    )

    total_files = sum(len(paths) for paths in result.values())
    print(f"\nDownload complete: {len(result)} subject folders, {total_files} EDF files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
