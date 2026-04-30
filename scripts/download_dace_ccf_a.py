#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils


DEFAULT_CATALOG = Path("sample_sweetcat.csv")
DEFAULT_CORRECTED_PATH = Path("observations/Best_RM/linear_corrected_with_uncertainties")
DEFAULT_OBSERVATION_PATH = Path("observations/Best_RM")
DEFAULT_OUTPUT_ROOT = Path("downloads/dace_ccf_a")
DEFAULT_FILE_TYPE = "CCF_A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download DACE spectroscopy products. You can either use the "
            "observation set built from local corrected files or query DACE "
            "directly by target/instrument."
        )
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--from-observations",
        action="store_true",
        help=(
            "Build the download list from local observation files, using the same "
            "selection logic as the RM notebooks."
        ),
    )
    source_group.add_argument(
        "--star",
        help="Target name to query directly in DACE, e.g. 'HD 209458' or 'TOI178'.",
    )

    parser.add_argument(
        "--instrument",
        help=(
            "Instrument filter for direct DACE queries, e.g. ESPRESSO19, ESPRESSO18, HARPS, HARPN."
        ),
    )
    parser.add_argument(
        "--file-type",
        default=DEFAULT_FILE_TYPE,
        help=(
            "DACE file type to browse/download. Examples: CCF_A, CCF_B, ccf, S1D_A, s1d. "
            f"Default: {DEFAULT_FILE_TYPE}"
        ),
    )
    parser.add_argument(
        "--drs-version",
        default="latest",
        help="DRS version passed to DACE. Default: latest",
    )
    parser.add_argument(
        "--date-night",
        action="append",
        default=[],
        help="Optional DACE date_night filter for direct queries. Repeatable, format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--program",
        action="append",
        dest="program_names",
        default=[],
        help="Optional DACE program_name filter for direct queries. Repeatable.",
    )
    parser.add_argument(
        "--dpr-type",
        action="append",
        dest="dpr_types",
        default=[],
        help="Optional DACE dpr_type filter for direct queries. Repeatable.",
    )
    parser.add_argument(
        "--dpr-catg",
        action="append",
        dest="dpr_categories",
        default=[],
        help="Optional DACE dpr_catg filter for direct queries. Repeatable. Example: SCIENCE",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory where downloads and manifests will be written. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only query DACE for matching products; do not download files.",
    )
    parser.add_argument(
        "--skip-browse",
        action="store_true",
        help="Skip the preview query and go straight to download().",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Request compressed archives for multi-file downloads.",
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"Path to the stellar catalog CSV for --from-observations mode. Default: {DEFAULT_CATALOG}",
    )
    parser.add_argument(
        "--corrected-path",
        type=Path,
        default=DEFAULT_CORRECTED_PATH,
        help=(
            "Directory containing *_linear_corrected.csv files used to build "
            "observation_sample_df in --from-observations mode. "
            f"Default: {DEFAULT_CORRECTED_PATH}"
        ),
    )
    parser.add_argument(
        "--observation-path",
        type=Path,
        default=DEFAULT_OBSERVATION_PATH,
        help=(
            "Directory containing source .rdb observation files in "
            f"--from-observations mode. Default: {DEFAULT_OBSERVATION_PATH}"
        ),
    )
    parser.add_argument(
        "--include-observation",
        action="append",
        dest="include_observations",
        default=[],
        help="Restrict --from-observations mode to one or more observation filenames.",
    )
    parser.add_argument(
        "--exclude-observation",
        action="append",
        dest="exclude_observations",
        default=[],
        help="Skip one or more observation filenames in --from-observations mode.",
    )

    args = parser.parse_args()

    if not args.from_observations and not args.star:
        args.from_observations = True

    if args.instrument and args.from_observations:
        parser.error("--instrument is only used with direct DACE queries via --star.")

    if args.date_night and args.from_observations:
        parser.error("--date-night is only used with direct DACE queries via --star.")

    if args.program_names and args.from_observations:
        parser.error("--program is only used with direct DACE queries via --star.")

    if args.dpr_types and args.from_observations:
        parser.error("--dpr-type is only used with direct DACE queries via --star.")

    if args.dpr_categories and args.from_observations:
        parser.error("--dpr-catg is only used with direct DACE queries via --star.")

    return args


def load_observation_sample_df(
    catalog_path: Path,
    corrected_path: Path,
    include_observations: list[str],
    exclude_observations: list[str],
) -> pd.DataFrame:
    sample_catalog_df = pd.read_csv(catalog_path)
    _, observation_sample_df = utils.create_rm_analysis_df(
        corrected_path=corrected_path,
        sample_catalog_df=sample_catalog_df,
        include_observations=include_observations or None,
        exclude_observations=exclude_observations or None,
    )
    return observation_sample_df.copy()


def load_raw_file_roots(observation_rdb_path: Path) -> list[str]:
    obs_df = pd.read_csv(observation_rdb_path, sep="\t", skiprows=[1])
    if "file_rootpath" not in obs_df.columns:
        raise ValueError(f"{observation_rdb_path} does not contain a file_rootpath column")

    raw_roots = obs_df["file_rootpath"].dropna().astype(str).str.strip()
    raw_roots = [value for value in dict.fromkeys(raw_roots) if value]
    if not raw_roots:
        raise ValueError(f"{observation_rdb_path} does not contain any raw frame names")
    return raw_roots


def load_spectroscopy():
    try:
        from dace_query.spectroscopy import Spectroscopy
    except ImportError as exc:
        raise SystemExit(
            "dace_query is not installed. Install it first, for example with "
            "`pip install dace-query`, and make sure your DACE API key is configured in ~/.dacerc."
        ) from exc
    return Spectroscopy


def count_products(result) -> int | None:
    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return len(result)
    if isinstance(result, dict):
        lengths = []
        for value in result.values():
            if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                try:
                    lengths.append(len(value))
                except TypeError:
                    continue
        if lengths:
            return max(lengths)
        return len(result)
    if hasattr(result, "__len__"):
        try:
            return len(result)
        except TypeError:
            return None
    return None


def sanitize_for_path(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "download"


def build_direct_filters(args: argparse.Namespace) -> dict[str, dict[str, list[str]]]:
    filters: dict[str, dict[str, list[str]]] = {"target_name": {"equal": [args.star]}}

    if args.instrument:
        filters["instrument_name"] = {"equal": [args.instrument]}
    if args.date_night:
        filters["date_night"] = {"equal": args.date_night}
    if args.program_names:
        filters["program_name"] = {"equal": args.program_names}
    if args.dpr_types:
        filters["dpr_type"] = {"equal": args.dpr_types}
    if args.dpr_categories:
        filters["dpr_catg"] = {"equal": args.dpr_categories}

    return filters


def browse_products(Spectroscopy, filters: dict, file_type: str, drs_version: str) -> int | None:
    browse_result = Spectroscopy.browse_products(
        filters=filters,
        file_type=file_type,
        drs_version=drs_version,
        output_format="dict",
    )
    return count_products(browse_result)


def download_products(
    Spectroscopy,
    filters: dict,
    file_type: str,
    drs_version: str,
    compressed: bool,
    output_directory: Path,
) -> None:
    Spectroscopy.download(
        filters=filters,
        file_type=file_type,
        drs_version=drs_version,
        compressed=compressed,
        output_directory=str(output_directory),
    )


def write_manifest_row(
    manifest_writer: csv.DictWriter,
    request_label: str,
    star: str,
    raw_file_count: int | None,
    browse_count: int | None,
    status: str,
    note: str,
    filters: dict,
) -> None:
    manifest_writer.writerow(
        {
            "request_label": request_label,
            "star": star,
            "raw_file_count": "" if raw_file_count is None else raw_file_count,
            "browse_count": "" if browse_count is None else browse_count,
            "status": status,
            "note": note,
            "filters": repr(filters),
        }
    )


def iter_local_requests(args: argparse.Namespace) -> list[dict]:
    observation_sample_df = load_observation_sample_df(
        catalog_path=args.catalog,
        corrected_path=args.corrected_path,
        include_observations=args.include_observations,
        exclude_observations=args.exclude_observations,
    ).sort_values(["star", "observation_file"])

    requests = []
    for row in observation_sample_df.itertuples(index=False):
        observation_file = row.observation_file
        observation_rdb_path = args.observation_path / observation_file
        requests.append(
            {
                "request_label": observation_file,
                "star": row.star,
                "display_name": getattr(row, "Name", row.star),
                "rdb_path": observation_rdb_path,
            }
        )
    return requests


def run_from_observations(args: argparse.Namespace, Spectroscopy, manifest_writer: csv.DictWriter) -> int:
    requests = iter_local_requests(args)
    if not requests:
        print("No observations matched the requested filters.", file=sys.stderr)
        return 1

    total = len(requests)
    for index, request in enumerate(requests, start=1):
        request_label = request["request_label"]
        star = request["star"]
        display_name = request["display_name"]
        observation_rdb_path = request["rdb_path"]

        print(f"[{index}/{total}] {request_label} ({display_name})")

        if not observation_rdb_path.exists():
            note = f"Observation file not found: {observation_rdb_path}"
            print(f"  missing: {note}")
            write_manifest_row(
                manifest_writer,
                request_label=request_label,
                star=star,
                raw_file_count=0,
                browse_count=None,
                status="missing_rdb",
                note=note,
                filters={},
            )
            continue

        try:
            raw_roots = load_raw_file_roots(observation_rdb_path)
        except Exception as exc:
            note = str(exc)
            print(f"  error reading raw frame list: {note}")
            write_manifest_row(
                manifest_writer,
                request_label=request_label,
                star=star,
                raw_file_count=0,
                browse_count=None,
                status="read_error",
                note=note,
                filters={},
            )
            continue

        filters = {"file_rootname": {"equal": raw_roots}}
        browse_count = None

        if not args.skip_browse:
            try:
                browse_count = browse_products(
                    Spectroscopy=Spectroscopy,
                    filters=filters,
                    file_type=args.file_type,
                    drs_version=args.drs_version,
                )
                print(f"  preview matched products: {browse_count if browse_count is not None else 'unknown'}")
            except Exception as exc:
                note = f"browse_products failed: {exc}"
                print(f"  error: {note}")
                write_manifest_row(
                    manifest_writer,
                    request_label=request_label,
                    star=star,
                    raw_file_count=len(raw_roots),
                    browse_count=None,
                    status="browse_error",
                    note=note,
                    filters=filters,
                )
                continue

        if args.preview_only:
            write_manifest_row(
                manifest_writer,
                request_label=request_label,
                star=star,
                raw_file_count=len(raw_roots),
                browse_count=browse_count,
                status="previewed",
                note="Preview completed; no download requested.",
                filters=filters,
            )
            continue

        output_directory = args.output_root.resolve() / sanitize_for_path(Path(request_label).stem)
        output_directory.mkdir(parents=True, exist_ok=True)

        try:
            download_products(
                Spectroscopy=Spectroscopy,
                filters=filters,
                file_type=args.file_type,
                drs_version=args.drs_version,
                compressed=args.compressed,
                output_directory=output_directory,
            )
            note = f"Downloaded to {output_directory}"
            print(f"  downloaded: {note}")
            write_manifest_row(
                manifest_writer,
                request_label=request_label,
                star=star,
                raw_file_count=len(raw_roots),
                browse_count=browse_count,
                status="downloaded",
                note=note,
                filters=filters,
            )
        except Exception as exc:
            note = f"download failed: {exc}"
            print(f"  error: {note}")
            write_manifest_row(
                manifest_writer,
                request_label=request_label,
                star=star,
                raw_file_count=len(raw_roots),
                browse_count=browse_count,
                status="download_error",
                note=note,
                filters=filters,
            )

    return 0


def run_direct_query(args: argparse.Namespace, Spectroscopy, manifest_writer: csv.DictWriter) -> int:
    filters = build_direct_filters(args)
    request_label_parts = [args.star]
    if args.instrument:
        request_label_parts.append(args.instrument)
    request_label = "__".join(sanitize_for_path(part) for part in request_label_parts)

    print(f"[1/1] direct query for {args.star}")
    print(f"  filters: {filters}")

    browse_count = None
    if not args.skip_browse:
        try:
            browse_count = browse_products(
                Spectroscopy=Spectroscopy,
                filters=filters,
                file_type=args.file_type,
                drs_version=args.drs_version,
            )
            print(f"  preview matched products: {browse_count if browse_count is not None else 'unknown'}")
        except Exception as exc:
            note = f"browse_products failed: {exc}"
            print(f"  error: {note}")
            write_manifest_row(
                manifest_writer,
                request_label=request_label,
                star=args.star,
                raw_file_count=None,
                browse_count=None,
                status="browse_error",
                note=note,
                filters=filters,
            )
            return 1

    if args.preview_only:
        write_manifest_row(
            manifest_writer,
            request_label=request_label,
            star=args.star,
            raw_file_count=None,
            browse_count=browse_count,
            status="previewed",
            note="Preview completed; no download requested.",
            filters=filters,
        )
        return 0

    output_directory = args.output_root.resolve() / request_label
    output_directory.mkdir(parents=True, exist_ok=True)

    try:
        download_products(
            Spectroscopy=Spectroscopy,
            filters=filters,
            file_type=args.file_type,
            drs_version=args.drs_version,
            compressed=args.compressed,
            output_directory=output_directory,
        )
        note = f"Downloaded to {output_directory}"
        print(f"  downloaded: {note}")
        write_manifest_row(
            manifest_writer,
            request_label=request_label,
            star=args.star,
            raw_file_count=None,
            browse_count=browse_count,
            status="downloaded",
            note=note,
            filters=filters,
        )
        return 0
    except Exception as exc:
        note = f"download failed: {exc}"
        print(f"  error: {note}")
        write_manifest_row(
            manifest_writer,
            request_label=request_label,
            star=args.star,
            raw_file_count=None,
            browse_count=browse_count,
            status="download_error",
            note=note,
            filters=filters,
        )
        return 1


def main() -> int:
    args = parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "download_manifest.csv"

    Spectroscopy = load_spectroscopy()

    with manifest_path.open("w", newline="") as manifest_file:
        manifest_writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "request_label",
                "star",
                "raw_file_count",
                "browse_count",
                "status",
                "note",
                "filters",
            ],
        )
        manifest_writer.writeheader()

        if args.from_observations:
            exit_code = run_from_observations(args, Spectroscopy, manifest_writer)
        else:
            exit_code = run_direct_query(args, Spectroscopy, manifest_writer)

    print(f"Wrote manifest to {manifest_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
