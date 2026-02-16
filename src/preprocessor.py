from __future__ import annotations

from typing import Any, Dict, Optional, List

import mne

from yaml_utils import get
from filtering import FilterApplier
from bids_derivatives import BIDSCleanedWriter


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

    def resample(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        if not bool(get(self.cfg, "preprocess.resample.enabled", True)):
            return raw.copy()
        tgt = get(self.cfg, "preprocess.resample.target_sfreq", None)
        if tgt is None:
            return raw.copy()

        out = raw.copy()
        cur = float(out.info["sfreq"])
        if abs(cur - float(tgt)) > 1e-6:
            out.resample(sfreq=float(tgt), npad="auto", verbose=False)
        return out

    def rereference(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        if not bool(get(self.cfg, "preprocess.rereference.enabled", False)):
            return raw.copy()

        out = raw.copy()
        ref = get(self.cfg, "preprocess.rereference.ref_channels", "average")
        if isinstance(ref, str) and ref.lower() == "average":
            out.set_eeg_reference("average", projection=False, verbose=False)
        else:
            out.set_eeg_reference(ref_channels=ref, projection=False, verbose=False)
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
        out_root = get(self.cfg, "outputs.bids_out_root", "results/preprocess/bids")
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
