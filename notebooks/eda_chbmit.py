"""
CHB-MIT EEG Dataset — Exploratory Data Analysis
=================================================
Saves all figures and a summary CSV to:
  /home/amir/Desktop/GWU/Research/EEG/results/EDA/

Run:
  python3 src/EDA/eda_chbmit.py
"""

from __future__ import annotations

import os
import csv
import warnings
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_ROOT  = Path("/home/amir/Desktop/GWU/Research/EEG/data/raw_data/chbmit")
OUT_ROOT  = Path("/home/amir/Desktop/GWU/Research/EEG/results/EDA")

for d in ["overview", "raw_signals", "psd", "stats", "seizure_vs_bg", "correlation"]:
    (OUT_ROOT / d).mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUT_ROOT}")


# ── Helpers ────────────────────────────────────────────────────────────────
def load_edf(path: Path) -> mne.io.BaseRaw | None:
    try:
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
        return raw
    except Exception as e:
        print(f"  [WARN] Could not load {path.name}: {e}")
        return None


def parse_events_tsv(tsv_path: Path):
    """Return list of (onset_sec, duration_sec, label) tuples."""
    events = []
    if not tsv_path.exists():
        return events
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                onset    = float(row.get("onset", 0))
                duration = float(row.get("duration", 0))
                label    = row.get("trial_type", row.get("label", "unknown"))
                events.append((onset, duration, label))
            except Exception:
                continue
    return events


def find_tsv(edf_path: Path) -> Path:
    return edf_path.with_name(edf_path.stem + "_events.tsv")


# ═══════════════════════════════════════════════════════════════════════════
# PASS 1 – Inventory scan (no preload)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PASS 1 – Scanning dataset …")
print("="*60)

subjects       = sorted(p for p in RAW_ROOT.iterdir() if p.is_dir())
summary_rows   = []          # per-file summary
subject_info   = {}          # subj → {n_files, total_dur, n_seizure_files, channels_set}
all_channels   = defaultdict(int)   # channel_name → count across files
seizure_events = []          # (subject, file, onset, duration)

for subj_dir in subjects:
    subj = subj_dir.name
    edf_files = sorted(subj_dir.glob("*.edf"))
    subj_dur = 0.0
    subj_sz_files = 0
    subj_channels = set()

    for edf_path in edf_files:
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        except Exception as e:
            print(f"  [WARN] {edf_path.name}: {e}")
            continue

        dur    = raw.times[-1]
        sfreq  = raw.info["sfreq"]
        n_ch   = len(raw.ch_names)
        subj_dur += dur
        subj_channels.update(raw.ch_names)
        for ch in raw.ch_names:
            all_channels[ch] += 1

        tsv     = find_tsv(edf_path)
        events  = parse_events_tsv(tsv)
        has_sz  = any("seiz" in ev[2].lower() or "sz" in ev[2].lower() or "ictal" in ev[2].lower()
                      for ev in events)
        if has_sz:
            subj_sz_files += 1
            for ev in events:
                if "seiz" in ev[2].lower() or "sz" in ev[2].lower() or "ictal" in ev[2].lower():
                    seizure_events.append((subj, edf_path.name, ev[0], ev[1]))

        summary_rows.append({
            "subject":       subj,
            "file":          edf_path.name,
            "duration_s":    round(dur, 2),
            "sfreq":         sfreq,
            "n_channels":    n_ch,
            "has_seizure":   int(has_sz),
            "n_seizure_evts": sum(1 for ev in events
                                  if "seiz" in ev[2].lower() or "sz" in ev[2].lower()),
        })

    subject_info[subj] = {
        "n_files":       len(edf_files),
        "total_dur_h":   round(subj_dur / 3600, 2),
        "n_seizure_files": subj_sz_files,
        "channels":      subj_channels,
    }
    print(f"  {subj}: {len(edf_files)} files  |  {subj_dur/3600:.2f} h  |  "
          f"{subj_sz_files} seizure files")

# Save per-file CSV
csv_path = OUT_ROOT / "overview" / "file_inventory.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"\nFile inventory saved → {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 – Dataset overview bar charts
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Dataset overview …")

subj_names   = list(subject_info.keys())
dur_hours    = [subject_info[s]["total_dur_h"]    for s in subj_names]
n_files_list = [subject_info[s]["n_files"]        for s in subj_names]
sz_files     = [subject_info[s]["n_seizure_files"] for s in subj_names]

fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle("CHB-MIT Dataset Overview", fontsize=15, fontweight="bold")

axes[0].bar(subj_names, dur_hours, color="steelblue")
axes[0].set_title("Total Recording Duration per Subject (hours)")
axes[0].set_ylabel("Hours")
axes[0].tick_params(axis="x", rotation=45)

axes[1].bar(subj_names, n_files_list, color="darkorange")
axes[1].set_title("Number of EDF Files per Subject")
axes[1].set_ylabel("Files")
axes[1].tick_params(axis="x", rotation=45)

axes[2].bar(subj_names, sz_files, color="crimson")
axes[2].set_title("Number of Seizure-Containing Files per Subject")
axes[2].set_ylabel("Files")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
fig.savefig(OUT_ROOT / "overview" / "dataset_overview.pdf")
plt.close()
print("  Saved dataset_overview.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 – Channel frequency across all files
# ═══════════════════════════════════════════════════════════════════════════
print("[Fig 2] Channel frequency …")

ch_sorted  = sorted(all_channels.items(), key=lambda x: -x[1])
ch_names_s = [c[0] for c in ch_sorted]
ch_counts  = [c[1] for c in ch_sorted]

fig, ax = plt.subplots(figsize=(max(14, len(ch_names_s) * 0.5), 6))
ax.bar(ch_names_s, ch_counts, color="teal")
ax.set_title("Channel Occurrence Count Across All EDF Files")
ax.set_ylabel("Number of Files")
ax.set_xlabel("Channel Name")
ax.tick_params(axis="x", rotation=90)
plt.tight_layout()
fig.savefig(OUT_ROOT / "overview" / "channel_frequency.pdf")
plt.close()
print("  Saved channel_frequency.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 – Seizure event duration distribution
# ═══════════════════════════════════════════════════════════════════════════
print("[Fig 3] Seizure duration distribution …")

if seizure_events:
    sz_durations = [ev[3] for ev in seizure_events]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sz_durations, bins=30, color="crimson", edgecolor="black")
    ax.set_title(f"Seizure Duration Distribution  (n={len(sz_durations)} events)")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(OUT_ROOT / "overview" / "seizure_duration_dist.pdf")
    plt.close()
    print(f"  Saved seizure_duration_dist.pdf  ({len(sz_durations)} seizures)")
else:
    print("  No seizure events parsed from TSV files.")


# ═══════════════════════════════════════════════════════════════════════════
# PASS 2 – Deep analysis on chb01_03 (has seizures, standard example)
# ═══════════════════════════════════════════════════════════════════════════
SAMPLE_SUBJ = "chb01"
SAMPLE_FILE = "chb01_03.edf"        # known to have a seizure
sample_path = RAW_ROOT / SAMPLE_SUBJ / SAMPLE_FILE
fallback    = sorted((RAW_ROOT / SAMPLE_SUBJ).glob("*.edf"))[0]
if not sample_path.exists():
    sample_path = fallback
    SAMPLE_FILE = sample_path.name

print(f"\n{'='*60}")
print(f"PASS 2 – Deep EDA on {SAMPLE_SUBJ}/{SAMPLE_FILE}")
print(f"{'='*60}")

raw = load_edf(sample_path)
if raw is None:
    raise RuntimeError(f"Cannot load sample file {sample_path}")

sfreq   = raw.info["sfreq"]
data, _ = raw[:, :]          # (n_ch, n_times)
ch_names = raw.ch_names
n_ch     = len(ch_names)
n_times  = data.shape[1]
dur_s    = n_times / sfreq

print(f"  Channels : {n_ch}")
print(f"  Sfreq    : {sfreq} Hz")
print(f"  Duration : {dur_s:.1f} s  ({dur_s/60:.1f} min)")
print(f"  Shape    : {data.shape}")


# ── NaN / Inf check ────────────────────────────────────────────────────────
nan_count = int(np.isnan(data).sum())
inf_count = int(np.isinf(data).sum())
print(f"  NaNs: {nan_count}   Infs: {inf_count}")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 – Raw signal: ALL channels, full file
# ═══════════════════════════════════════════════════════════════════════════
print("[Fig 4] Raw signal – all channels …")

PLOT_DUR  = min(60.0, dur_s)       # plot up to 60 s
n_plot    = int(PLOT_DUR * sfreq)
t_axis    = np.arange(n_plot) / sfreq

# normalise each channel for display (zero-mean, unit-std)
disp_data = data[:, :n_plot].copy()
for i in range(n_ch):
    std = disp_data[i].std()
    if std > 0:
        disp_data[i] = (disp_data[i] - disp_data[i].mean()) / std

# vertical spacing
spacing = 5.0
offsets = np.arange(n_ch) * spacing

fig, ax = plt.subplots(figsize=(22, max(12, n_ch * 0.55)))
for i in range(n_ch):
    ax.plot(t_axis, disp_data[i] + offsets[i], linewidth=0.6, color="black")
    ax.text(-0.5, offsets[i], ch_names[i], ha="right", va="center",
            fontsize=7, color="navy")

# Overlay seizure windows
tsv_path = find_tsv(sample_path)
sz_evts  = parse_events_tsv(tsv_path)
for onset, dur_ev, label in sz_evts:
    if "seiz" in label.lower() or "sz" in label.lower() or "ictal" in label.lower():
        ax.axvspan(onset, min(onset + dur_ev, PLOT_DUR),
                   color="red", alpha=0.15, label="Seizure")

ax.set_yticks([])
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_xlim(0, PLOT_DUR)
ax.set_title(f"All Channels – {SAMPLE_SUBJ}/{SAMPLE_FILE}  (first {PLOT_DUR:.0f} s)",
             fontsize=13, fontweight="bold")
handles, labels_ = ax.get_legend_handles_labels()
if handles:
    ax.legend(handles[:1], labels_[:1], loc="upper right")
plt.tight_layout()
fig.savefig(OUT_ROOT / "raw_signals" / f"{SAMPLE_SUBJ}_{SAMPLE_FILE.replace('.edf','')}_all_channels.pdf",
            dpi=120)
plt.close()
print("  Saved all_channels raw signal plot")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 5 – Per-channel amplitude statistics
# ═══════════════════════════════════════════════════════════════════════════
print("[Fig 5] Per-channel amplitude statistics …")

ch_mean  = data.mean(axis=1)
ch_std   = data.std(axis=1)
ch_min   = data.min(axis=1)
ch_max   = data.max(axis=1)
ch_range = ch_max - ch_min

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle(f"Per-Channel Amplitude Statistics – {SAMPLE_FILE}", fontsize=13, fontweight="bold")

x = np.arange(n_ch)
for ax_, vals, title, color in zip(
        axes.flat,
        [ch_std * 1e6, ch_mean * 1e6, ch_min * 1e6, ch_range * 1e6],
        ["Std Dev (µV)", "Mean (µV)", "Min (µV)", "Peak-to-peak (µV)"],
        ["steelblue", "darkorange", "forestgreen", "purple"]):
    ax_.bar(x, vals, color=color)
    ax_.set_xticks(x)
    ax_.set_xticklabels(ch_names, rotation=90, fontsize=7)
    ax_.set_title(title)
    ax_.set_ylabel("µV")

plt.tight_layout()
fig.savefig(OUT_ROOT / "stats" / "per_channel_amplitude_stats.pdf")
plt.close()
print("  Saved per_channel_amplitude_stats.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 6 – Amplitude distribution (violin) per channel
# ═══════════════════════════════════════════════════════════════════════════
print("[Fig 6] Amplitude violin plots …")

# subsample to speed up
step    = max(1, n_times // 50000)
sampled = data[:, ::step] * 1e6   # µV

fig, ax = plt.subplots(figsize=(max(16, n_ch * 0.8), 7))
parts = ax.violinplot([sampled[i] for i in range(n_ch)],
                      positions=np.arange(n_ch), showmedians=True,
                      showextrema=False)
for pc in parts["bodies"]:
    pc.set_facecolor("steelblue")
    pc.set_alpha(0.6)
ax.set_xticks(np.arange(n_ch))
ax.set_xticklabels(ch_names, rotation=90, fontsize=8)
ax.set_ylabel("Amplitude (µV)")
ax.set_title(f"Channel Amplitude Distribution – {SAMPLE_FILE}", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_ROOT / "stats" / "channel_amplitude_violin.pdf")
plt.close()
print("  Saved channel_amplitude_violin.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 7 – Power Spectral Density per channel
# ═══════════════════════════════════════════════════════════════════════════
print("[Fig 7] PSD per channel …")

from scipy.signal import welch

FREQ_BANDS = {
    "delta":  (1,  4),
    "theta":  (4,  8),
    "alpha":  (8, 13),
    "beta":   (13, 30),
    "gamma":  (30, 50),
}

nperseg = int(sfreq * 4)
band_power = {band: [] for band in FREQ_BANDS}   # band → [power per channel]

n_cols = 4
n_rows = int(np.ceil(n_ch / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
fig.suptitle(f"PSD per Channel – {SAMPLE_FILE}", fontsize=13, fontweight="bold")

axes_flat = axes.flat
for i, (ax_, ch) in enumerate(zip(axes_flat, ch_names)):
    freqs, psd = welch(data[i], fs=sfreq, nperseg=nperseg)
    ax_.semilogy(freqs, psd, linewidth=0.8, color="steelblue")
    ax_.set_xlim(0, 55)
    ax_.set_title(ch, fontsize=8)
    ax_.set_xlabel("Hz", fontsize=7)
    ax_.set_ylabel("PSD", fontsize=7)
    ax_.tick_params(labelsize=6)
    # shade bands
    colors = ["#cce5ff", "#d4edda", "#fff3cd", "#f8d7da", "#e2d9f3"]
    for (band, (lo, hi)), bc in zip(FREQ_BANDS.items(), colors):
        ax_.axvspan(lo, hi, alpha=0.25, color=bc)
        mask = (freqs >= lo) & (freqs <= hi)
        band_power[band].append(float(np.trapz(psd[mask], freqs[mask])))

# hide empty subplots
for ax_ in list(axes_flat)[n_ch:]:
    ax_.set_visible(False)

plt.tight_layout()
fig.savefig(OUT_ROOT / "psd" / "psd_per_channel.pdf")
plt.close()
print("  Saved psd_per_channel.pdf")


# FIG 7b – Band power summary bar chart
print("[Fig 7b] Band power summary …")

fig, axes = plt.subplots(1, len(FREQ_BANDS), figsize=(22, 5))
fig.suptitle(f"Frequency Band Power per Channel – {SAMPLE_FILE}", fontsize=13, fontweight="bold")

band_colors = ["#4472C4", "#70AD47", "#ED7D31", "#FF0000", "#7030A0"]
for ax_, (band, powers), color in zip(axes, band_power.items(), band_colors):
    ax_.bar(np.arange(n_ch), powers, color=color)
    ax_.set_xticks(np.arange(n_ch))
    ax_.set_xticklabels(ch_names, rotation=90, fontsize=7)
    ax_.set_title(band.capitalize())
    ax_.set_ylabel("Power (V²/Hz)")

plt.tight_layout()
fig.savefig(OUT_ROOT / "psd" / "band_power_per_channel.pdf")
plt.close()
print("  Saved band_power_per_channel.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 8 – Seizure vs Background signal comparison
# ═══════════════════════════════════════════════════════════════════════════
print("[Fig 8] Seizure vs Background comparison …")

if sz_evts:
    sz_onset = sz_evts[0][0]
    sz_dur   = sz_evts[0][1]

    sz_start  = int(sz_onset * sfreq)
    sz_end    = min(int((sz_onset + sz_dur) * sfreq), n_times)
    seg_len   = sz_end - sz_start

    # background: same length, taken from start (or end) of file – avoid seizure window
    bg_start  = 0 if sz_start > seg_len + int(10 * sfreq) else max(0, n_times - seg_len)
    bg_end    = bg_start + seg_len

    sz_seg    = data[:, sz_start:sz_end]   * 1e6   # µV
    bg_seg    = data[:, bg_start:bg_end]   * 1e6

    t_seg     = np.arange(seg_len) / sfreq

    # plot first 8 channels side by side
    show_ch   = min(8, n_ch)
    fig, axes = plt.subplots(show_ch, 2, figsize=(18, show_ch * 2.2), sharex=True)
    fig.suptitle("Seizure vs Background EEG  (same duration segment)", fontsize=13, fontweight="bold")

    for i in range(show_ch):
        axes[i, 0].plot(t_seg, bg_seg[i], linewidth=0.6, color="steelblue")
        axes[i, 0].set_ylabel(ch_names[i], fontsize=8, rotation=0, labelpad=50)
        axes[i, 1].plot(t_seg, sz_seg[i],  linewidth=0.6, color="crimson")
        for j in range(2):
            axes[i, j].tick_params(labelsize=7)

    axes[0, 0].set_title("Background", fontsize=11)
    axes[0, 1].set_title("Seizure",    fontsize=11)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig.savefig(OUT_ROOT / "seizure_vs_bg" / "seizure_vs_background_raw.pdf")
    plt.close()
    print("  Saved seizure_vs_background_raw.pdf")

    # FIG 8b – PSD comparison seizure vs background
    fig, axes = plt.subplots(int(np.ceil(show_ch / 2)), 2,
                             figsize=(16, int(np.ceil(show_ch / 2)) * 3))
    fig.suptitle("PSD: Seizure vs Background", fontsize=13, fontweight="bold")
    for i, ax_ in enumerate(axes.flat):
        if i >= show_ch:
            ax_.set_visible(False)
            continue
        f_bg,  p_bg  = welch(bg_seg[i] / 1e6, fs=sfreq, nperseg=nperseg)
        f_sz,  p_sz  = welch(sz_seg[i] / 1e6, fs=sfreq, nperseg=nperseg)
        ax_.semilogy(f_bg, p_bg, label="Background", color="steelblue", linewidth=0.9)
        ax_.semilogy(f_sz, p_sz, label="Seizure",    color="crimson",   linewidth=0.9)
        ax_.set_xlim(0, 55)
        ax_.set_title(ch_names[i], fontsize=9)
        ax_.set_xlabel("Hz", fontsize=8)
        ax_.legend(fontsize=7)
        ax_.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(OUT_ROOT / "seizure_vs_bg" / "psd_seizure_vs_background.pdf")
    plt.close()
    print("  Saved psd_seizure_vs_background.pdf")

else:
    print("  No seizure events in TSV for this file – skipping seizure comparison.")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 9 – Channel correlation matrix
# ═══════════════════════════════════════════════════════════════════════════
print("[Fig 9] Channel correlation matrix …")

step_corr = max(1, n_times // 100000)
corr_data = data[:, ::step_corr]
corr_mat  = np.corrcoef(corr_data)

fig, ax = plt.subplots(figsize=(max(10, n_ch * 0.55), max(9, n_ch * 0.5)))
im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
ax.set_xticks(np.arange(n_ch))
ax.set_yticks(np.arange(n_ch))
ax.set_xticklabels(ch_names, rotation=90, fontsize=8)
ax.set_yticklabels(ch_names, fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title(f"Channel Correlation Matrix – {SAMPLE_FILE}", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_ROOT / "correlation" / "channel_correlation_matrix.pdf")
plt.close()
print("  Saved channel_correlation_matrix.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 10 – Spectrogram for 2 representative channels
# ═══════════════════════════════════════════════════════════════════════════
print("[Fig 10] Spectrograms …")

from scipy.signal import spectrogram as sp_spectrogram

SHOW_SPEC_CH = min(4, n_ch)

fig, axes = plt.subplots(SHOW_SPEC_CH, 1, figsize=(22, SHOW_SPEC_CH * 3))
fig.suptitle(f"Spectrogram – {SAMPLE_FILE}  (first {dur_s:.0f} s)", fontsize=13, fontweight="bold")

for i in range(SHOW_SPEC_CH):
    f, t_spec, Sxx = sp_spectrogram(data[i], fs=sfreq,
                                    nperseg=int(sfreq * 2),
                                    noverlap=int(sfreq * 1.5))
    mask = f <= 60
    axes[i].pcolormesh(t_spec, f[mask],
                       10 * np.log10(Sxx[mask] + 1e-30),
                       shading="gouraud", cmap="inferno")
    axes[i].set_ylabel("Hz", fontsize=8)
    axes[i].set_title(ch_names[i], fontsize=9, loc="left")
    # Overlay seizure bands
    for onset, dur_ev, label in sz_evts:
        if "seiz" in label.lower() or "sz" in label.lower() or "ictal" in label.lower():
            axes[i].axvspan(onset, onset + dur_ev, color="cyan", alpha=0.3)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
fig.savefig(OUT_ROOT / "psd" / "spectrograms.pdf")
plt.close()
print("  Saved spectrograms.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 11 – Class balance (seizure vs background windows) across full dataset
# ═══════════════════════════════════════════════════════════════════════════
print("[Fig 11] Class balance …")

total_sz_dur  = sum(ev[3] for ev in seizure_events)
total_rec_dur = sum(r["duration_s"] for r in summary_rows)
total_bg_dur  = total_rec_dur - total_sz_dur

labels_pie = ["Background", "Seizure"]
sizes_pie  = [total_bg_dur, total_sz_dur]
colors_pie = ["steelblue", "crimson"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Class Balance Across Full Dataset", fontsize=13, fontweight="bold")

axes[0].pie(sizes_pie, labels=labels_pie, colors=colors_pie, autopct="%1.2f%%",
            startangle=90, textprops={"fontsize": 12})
axes[0].set_title("Duration Fraction")

axes[1].bar(labels_pie,
            [total_bg_dur / 3600, total_sz_dur / 3600],
            color=colors_pie)
axes[1].set_ylabel("Hours")
axes[1].set_title("Absolute Duration (hours)")
for ax_ in axes[1:]:
    for bar, val in zip(ax_.patches, [total_bg_dur/3600, total_sz_dur/3600]):
        ax_.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f} h", ha="center", fontsize=10)

plt.tight_layout()
fig.savefig(OUT_ROOT / "overview" / "class_balance.pdf")
plt.close()
print("  Saved class_balance.pdf")


# ─── Final summary ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EDA COMPLETE")
print("="*60)
print(f"Subjects       : {len(subjects)}")
print(f"Total files    : {len(summary_rows)}")
print(f"Total duration : {total_rec_dur/3600:.1f} h")
print(f"Seizure events : {len(seizure_events)}")
print(f"Seizure dur    : {total_sz_dur:.0f} s  ({total_sz_dur/total_rec_dur*100:.3f}%)")
print(f"Unique channels: {len(all_channels)}")
print(f"\nAll figures saved to: {OUT_ROOT}")
