from __future__ import annotations

import argparse
from typing import Any, Dict

from yaml_utils import load_yaml, get
from bids_io import BIDSLoader
from preprocessor import EEGPreprocessor
from time_domain import TimeDomainModule
from eda_engine import EDAEngine
from bot_diagrams import DiagramBuilder


def _enabled(cfg: Dict[str, Any], path: str, default: bool = False) -> bool:
    return bool(get(cfg, path, default))


def main():
    ap = argparse.ArgumentParser(description="EEG preprocessing + EDA runner (YAML-driven)")
    ap.add_argument("--config", default="src/configs/config.yaml", help="Path to YAML config")
    ap.add_argument("--max_recordings", type=int, default=None, help="Limit recordings processed")
    ap.add_argument("--subject", default=None, help="Filter by subject (without 'sub-'), e.g. '01'")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    run_diagrams = _enabled(cfg, "run.diagrams", True)
    run_preprocess = _enabled(cfg, "run.preprocess", True)
    run_write_bids = _enabled(cfg, "run.write_bids", True)
    run_eda = _enabled(cfg, "run.eda", True)

    if not run_write_bids:
        raise ValueError("run.write_bids must be true because your final output must be BIDS (.fif).")

    if run_diagrams:
        DiagramBuilder(cfg).save_all()

    bids_root = get(cfg, "dataset.bids_root", "results/bids_dataset")
    datatype = get(cfg, "bids.datatype", "eeg")
    suffix = get(cfg, "bids.suffix", "eeg")
    extension = get(cfg, "bids.extension", None)
    task = get(cfg, "bids.task", None)
    preload = bool(get(cfg, "preprocess.preload", True))

    loader = BIDSLoader(bids_root, datatype=datatype, suffix=suffix)
    if not loader.is_available():
        raise RuntimeError(
            f"BIDS input not available at {bids_root}. "
            f"Missing dataset_description.json or mne-bids not installed."
        )

    recs = loader.list_recordings(task=task)
    if args.subject is not None:
        recs = [r for r in recs if r.subject == args.subject]
    if not recs:
        raise RuntimeError("No BIDS recordings found for given filters.")
    if args.max_recordings is not None:
        recs = recs[: int(args.max_recordings)]

    pre = EEGPreprocessor(cfg)
    td = TimeDomainModule(cfg)
    eda = EDAEngine(cfg)

    do_resample = run_preprocess and _enabled(cfg, "preprocess.resample.enabled", True)
    do_reref = run_preprocess and _enabled(cfg, "preprocess.rereference.enabled", False)
    do_filter = run_preprocess and _enabled(cfg, "preprocess.filtering.enabled", True)

    bads_enabled = _enabled(cfg, "analysis.time_domain.bad_channels.enabled", True)
    interp_enabled = bads_enabled and _enabled(cfg, "analysis.time_domain.bad_channels.interpolate", True)
    qc_enabled = _enabled(cfg, "analysis.time_domain.qc.enabled", True)

    labels_enabled = _enabled(cfg, "analysis.labels.enabled", False)
    use_bids_events = _enabled(cfg, "analysis.labels.tsv.use_bids_events", True)

    print(f"[INFO] recordings: {len(recs)} | preprocess={run_preprocess} | eda={run_eda} | write_bids={run_write_bids}")
    print(f"[INFO] steps: resample={do_resample} reref={do_reref} filter={do_filter} qc={qc_enabled} interp_bads={interp_enabled} labels={labels_enabled}")

    for rec in recs:
        rid = loader.recording_id(rec)

        try:
            raw = loader.load_raw(rec, preload=preload, extension=extension)
        except Exception as e:
            print(f"[SKIP] {rid} could not load: {e}")
            continue

        raw_before = raw.copy()

        if run_preprocess:
            cleaned, steps = pre.process(
                raw,
                do_load=True,
                do_resample=do_resample,
                do_reref=do_reref,
                do_filter=do_filter,
            )
        else:
            cleaned, steps = raw, ["preprocess_skipped"]

        # QC -> mark bads -> interpolate (optional)
        qc = td.qc(cleaned) if qc_enabled else {}
        if bads_enabled and qc:
            td.mark_bads_from_qc(cleaned, qc)
        if interp_enabled:
            cleaned = td.interpolate_bads(cleaned)

        # REQUIRED: write BIDS cleaned output
        try:
            out_fpath = pre.write_bids_cleaned(
                cleaned,
                subject=rec.subject,
                session=rec.session,
                task=rec.task,
                run=rec.run,
                datatype=rec.datatype,
                suffix=rec.suffix,
            )
        except Exception as e:
            print(f"[ERROR] {rid} failed to write BIDS output: {e}")
            continue

        # Locate events.tsv for seizure vs non-seizure comparisons
        events_path = None
        if labels_enabled and use_bids_events:
            ep = loader.events_tsv_path(rec)
            events_path = str(ep) if ep is not None else None

        # Optional EDA
        if run_eda:
            try:
                eda_out = eda.run(cleaned, rid, raw_before=raw_before, events_tsv_path=events_path)
            except Exception as e:
                eda_out = {"error": str(e)}
        else:
            eda_out = {"eda": "skipped"}

        print(f"[OK] {rid}")
        print(f"  steps_applied: {steps}")
        print(f"  bads_marked: {cleaned.info.get('bads', [])}")
        print(f"  events_tsv: {events_path}")
        print(f"  bids_out: {out_fpath}")
        print(f"  eda_out: {eda_out}")

    print("[DONE]")


if __name__ == "__main__":
    main()
