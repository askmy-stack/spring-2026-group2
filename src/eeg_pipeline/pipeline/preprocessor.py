from __future__ import annotations

from typing import Any, Dict, Optional, List

import mne

from eeg_pipeline.core.yaml_utils import get
from eeg_pipeline.pipeline.filtering import FilterApplier
from eeg_pipeline.core.bids_derivatives import BIDSCleanedWriter


class EEGPreprocessor:
    """
    Preprocessing toolbox (NO ICA):
      - ensure_loaded
      - resample
      - rereference
      - filtering (+ optional wavelet)
      - write cleaned BIDS (.fif)
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def ensure_loaded(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        out = raw.copy()
        if bool(get(self.cfg, "preprocess.preload", True)) and not out.preload:
            out.load_data()
        return out

    def apply_filtering(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        if not bool(get(self.cfg, "preprocess.filtering.enabled", True)):
            return raw.copy()

        f_l = get(self.cfg, "preprocess.filtering.l_freq", None)
        f_h = get(self.cfg, "preprocess.filtering.h_freq", None)
        notch = get(self.cfg, "preprocess.filtering.notch_freqs", None)
        method = get(self.cfg, "preprocess.filtering.method", "fir")
        iir_params = get(self.cfg, "preprocess.filtering.iir_params", None)

        use_wavelet = bool(get(self.cfg, "preprocess.wavelet_denoise.enabled", False))
        wav_family = get(self.cfg, "preprocess.wavelet_denoise.family", "db4")
        wav_level = get(self.cfg, "preprocess.wavelet_denoise.level", None)
        wav_thr = get(self.cfg, "preprocess.wavelet_denoise.threshold", "soft")

        applier = FilterApplier(
            l_freq=f_l,
            h_freq=f_h,
            notch_freqs=notch,
            method=method,
            iir_params=iir_params,
            use_wavelet=use_wavelet,
            wavelet_family=wav_family,
            wavelet_level=wav_level,
            wavelet_threshold=wav_thr,
        )
        return applier.apply(raw)

    def process(
        self,
        raw: mne.io.BaseRaw,
        *,
        do_load: bool = True,
        do_resample: bool = True,
        do_reref: bool = False,
        do_filter: bool = True,
    ) -> tuple[mne.io.BaseRaw, List[str]]:
        steps: List[str] = []
        out = raw.copy()

        if do_load:
            out = self.ensure_loaded(out)
            steps.append("load_data")

        if do_resample:
            out = self.resample(out)
            steps.append("resample")

        if do_reref:
            out = self.rereference(out)
            steps.append("rereference")

        if do_filter:
            out = self.apply_filtering(out)
            steps.append("filtering(+wavelet_if_enabled)")

        return out, steps

    def write_bids_cleaned(
        self,
        cleaned_raw: mne.io.BaseRaw,
        *,
        subject: str,
        session: Optional[str],
        task: Optional[str],
        run: Optional[str],
        datatype: str,
        suffix: str,
    ) -> str:
        out_root = get(self.cfg, "export_cleaned.out_root", "results/preprocess/bids_dataset")
        writer = BIDSCleanedWriter(out_root)
        return writer.write_cleaned_raw(
            cleaned_raw,
            subject=subject,
            session=session,
            task=task,
            run=run,
            datatype=datatype,
            suffix=suffix,
            overwrite=True,
        )
