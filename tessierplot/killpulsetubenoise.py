"""
iv_artifact_removal.py
======================
Remove a sinusoidal interference artifact from IV-curve measurements of
superconducting devices.

Algorithm
---------
1.  FREQUENCY DETECTION (segmentation-free)
    - Differentiate voltage → kills slow IV baseline
    - Bandpass 1.0–1.6 Hz (Butterworth order 4) → isolates artifact band
    - Exclude spike samples (|dV/dt| > median + 8·MAD, dilated 20 samples)
    - Scan f_scan_lo–f_scan_hi Hz on bandpassed dV, minimise residual std → f₀
    - Sub-bin refinement via scipy minimize_scalar

2.  SLIDING-WINDOW PHASE TRACKING (segmentation-free)
    - Window = 3 artifact cycles, step = window/4
    - Per window: fit local poly(I) baseline (degree 3) + sin/cos at f₀
      directly in voltage space → extract phase φ = arctan2(C, S)
    - Unwrap phase across windows

3.  FREQUENCY REFINEMENT
    - Fit SNR-weighted linear trend to unwrapped phase
    - Slope → correction to f₀ giving f_refined
    - Subtract linear trend → detrended nonlinear phase residual

4.  PHASE OUTLIER REMOVAL
    - Compute second derivative d²φ/dt² of the full raw phase sequence
    - Fix threshold = thresh_mult × median(|d²φ/dt²|)  (computed once only)
    - Iterate: recompute d²φ on current valid set, remove middle point of
      any triplet exceeding the FIXED threshold; repeat until convergence
    - Fixed threshold prevents adaptive cascade from eating valid points

5.  AMPLITUDE ESTIMATION
    - SNR²-weighted average of per-window amplitude estimates

6.  ARTIFACT SUBTRACTION
    - CubicSpline through valid detrended phase points → φ_smooth(t)
    - Full phase = linear trend (from f_refined) + φ_smooth(t)
    - Subtract A₀ · sin(2π f₀ t + φ_full(t))

Usage
-----
Standalone:
    python iv_artifact_removal.py input.dat output.dat --duration 67 --plot

As a library:
    from iv_artifact_removal import remove_artifact, plot_results
    voltage_clean, meta = remove_artifact(voltage, current, duration=67.0)

Input format
------------
ASCII, two columns, comment lines start with '#':
    column 0 : source-drain current  (A)
    column 1 : drain voltage         (V)
"""

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline, PchipInterpolator


# ─────────────────────────────────────────────────────────────────────────────
# 1. Frequency detection
# ─────────────────────────────────────────────────────────────────────────────

def _spike_mask(voltage, dilation=20):
    """Return (spike_bool_mask, dV/dt array)."""
    dv     = np.gradient(voltage)
    abs_dv = np.abs(dv)
    mad    = np.median(np.abs(abs_dv - np.median(abs_dv)))
    thresh = np.median(abs_dv) + 8.0 * mad
    return binary_dilation(abs_dv > thresh, iterations=dilation), dv


def detect_frequency(voltage, t,
                     f_scan_lo=1.30, f_scan_hi=1.45, n_scan=300,
                     bp_lo=1.0, bp_hi=1.6, bp_order=4):
    """
    Detect artifact frequency from bandpassed dV/dt (no segmentation).

    Returns
    -------
    f0 : float – estimated artifact frequency (Hz)
    """
    fs = 1.0 / (t[1] - t[0])
    spikes, dv = _spike_mask(voltage)
    clean = ~spikes

    b, a  = butter(bp_order, [bp_lo / (fs/2), bp_hi / (fs/2)], btype='band')
    dv_bp = filtfilt(b, a, dv)

    def _score(f):
        M = np.column_stack([np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t)])
        c, _, _, _ = np.linalg.lstsq(M[clean], dv_bp[clean], rcond=None)
        return np.std(dv_bp[clean] - M[clean] @ c)

    freqs    = np.linspace(f_scan_lo, f_scan_hi, n_scan)
    f_coarse = freqs[np.argmin([_score(f) for f in freqs])]
    res = minimize_scalar(_score,
                          bounds=(f_coarse - 0.02, f_coarse + 0.02),
                          method='bounded', options={'xatol': 1e-7})
    return res.x


# ─────────────────────────────────────────────────────────────────────────────
# 2 & 3. Sliding-window phase tracking + frequency refinement
# ─────────────────────────────────────────────────────────────────────────────

def _window_phase(voltage, current, t, f0, win, step):
    """
    Slide a window across the full record; fit local poly(I) + sin/cos.

    Returns wt, phi_unwrapped, snr, amp  (all length = number of windows).
    """
    N = len(t)
    wt, phi_raw, snr_arr, amp_arr = [], [], [], []

    for s in range(0, N - win, step):
        e   = s + win
        tw  = t[s:e];  vw = voltage[s:e];  I_s = current[s:e]
        I_n = (I_s - I_s.mean()) / (I_s.max() - I_s.min() + 1e-20)
        M   = np.column_stack([
            np.ones(win), I_n, I_n**2, I_n**3,
            np.sin(2*np.pi*f0*tw),
            np.cos(2*np.pi*f0*tw),
        ])
        c, _, _, _ = np.linalg.lstsq(M, vw, rcond=None)
        S, C  = c[4], c[5]
        A_loc = np.sqrt(S**2 + C**2)
        resid = np.std(vw - M @ c)
        wt.append(tw.mean())
        phi_raw.append(np.arctan2(C, S))
        snr_arr.append(float(A_loc / (resid + 1e-30)))
        amp_arr.append(float(A_loc))

    return (np.array(wt),
            np.unwrap(np.array(phi_raw)),
            np.array(snr_arr),
            np.array(amp_arr))


def _refine_frequency(wt, phi_uw, snr, f0):
    """Fit SNR-weighted linear phase trend; return refined f, slope, intercept, residual."""
    slope, intercept = np.polyfit(wt, phi_uw, 1, w=snr)
    f_refined = f0 + slope / (2*np.pi)
    phi_det   = phi_uw - (slope * wt + intercept)
    return f_refined, slope, intercept, phi_det


# ─────────────────────────────────────────────────────────────────────────────
# 4. Phase outlier removal
# ─────────────────────────────────────────────────────────────────────────────

def filter_phase_outliers(wt, phi_det, thresh_mult=5.0, max_iter=20):
    """
    Iteratively remove phase points with implausible d²φ/dt².

    Threshold is computed ONCE from the full raw sequence and held fixed
    throughout all iterations — prevents the adaptive cascade that would
    tighten the threshold and remove valid points near transition regions.

    Parameters
    ----------
    wt          : window centre times (s)
    phi_det     : detrended phase (rad)
    thresh_mult : threshold = thresh_mult × median(|d²φ/dt²|), default 5.0
    max_iter    : iteration cap

    Returns
    -------
    valid        : bool array, True = kept
    fixed_thresh : threshold used (rad/s², for diagnostics)
    """
    def _d2(t_v, phi_v):
        dt = np.diff(t_v)
        return np.diff(np.diff(phi_v) / dt) / dt[:-1]

    # Fix threshold from the full raw sequence (never updated)
    fixed_thresh = thresh_mult * np.median(np.abs(_d2(wt, phi_det)))

    valid = np.ones(len(wt), dtype=bool)
    for _ in range(max_iter):
        t_v, phi_v = wt[valid], phi_det[valid]
        if len(t_v) < 4:
            break
        bad_v = np.abs(_d2(t_v, phi_v)) > fixed_thresh
        if not bad_v.any():
            break
        vi = np.where(valid)[0]
        valid[vi[1:-1][bad_v]] = False

    return valid, fixed_thresh


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def remove_artifact(voltage, current, duration=None, fs=None,
                    f_scan_lo=1.30, f_scan_hi=1.45,
                    thresh_mult=5.0):
    """
    Remove sinusoidal artifact from IV-curve voltage data.

    Parameters
    ----------
    voltage    : 1-D array, measured voltage (V)
    current    : 1-D array, source-drain current (A)
    duration   : total sweep duration in seconds  (provide this OR fs)
    fs         : sample rate in Hz                (provide this OR duration)
    f_scan_lo  : lower bound of frequency scan (Hz), default 1.30
    f_scan_hi  : upper bound of frequency scan (Hz), default 1.45
    thresh_mult: d²φ/dt² threshold multiplier, default 5.0

    Returns
    -------
    voltage_clean : artifact-subtracted voltage (V)
    meta          : dict with diagnostic fields —
                      f0, f_refined, A0, t, wt, phi_det, phi_smooth,
                      valid, fixed_thresh, snr, amp
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    N = len(voltage)

    if duration is not None:
        fs = N / duration
    elif fs is None:
        raise ValueError("Provide either 'duration' or 'fs'.")
    t = np.arange(N) / fs

    # 1. Frequency
    f0 = detect_frequency(voltage, t, f_scan_lo=f_scan_lo, f_scan_hi=f_scan_hi)

    # 2. Sliding window
    win  = int(3 * fs / f0)
    step = max(1, win // 4)
    wt, phi_uw, snr, amp = _window_phase(voltage, current, t, f0, win, step)

    # 3. Frequency refinement
    f_refined, slope, intercept, phi_det = _refine_frequency(wt, phi_uw, snr, f0)

    # 4. Phase outlier removal — fixed threshold
    valid, fixed_thresh = filter_phase_outliers(wt, phi_det,
                                                thresh_mult=thresh_mult)

    # 5. Amplitude
    A0 = np.average(amp, weights=snr**2)

    # 6. Spline + subtract
    # Interpolate through valid points; clamp flat beyond the boundary points
    t_v   = wt[valid];  phi_v = phi_det[valid]
    cs    = PchipInterpolator(t_v, phi_v, extrapolate=False)
    phi_smooth = np.where(t < t_v[0],  phi_v[0],
                 np.where(t > t_v[-1], phi_v[-1],
                          cs(t)))
    phi_full      = slope * t + intercept + phi_smooth
    artifact      = A0 * np.sin(2*np.pi*f0*t + phi_full)
    voltage_clean = voltage - artifact

    meta = dict(
        f0=f0, f_refined=f_refined, A0=A0,
        t=t, wt=wt, phi_det=phi_det, phi_smooth=phi_smooth,
        valid=valid, fixed_thresh=fixed_thresh,
        snr=snr, amp=amp,
    )
    return voltage_clean, meta


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(voltage, current, voltage_clean, meta, save_path=None):
    """Five-panel diagnostic plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    t          = meta['t']
    wt         = meta['wt']
    phi_det    = meta['phi_det']
    phi_smooth = meta['phi_smooth']
    valid      = meta['valid']
    snr        = meta['snr']

    N    = len(t)
    dv_r = uniform_filter1d(np.gradient(voltage),       size=6)
    dv_c = uniform_filter1d(np.gradient(voltage_clean), size=6)

    fig, axes = plt.subplots(5, 1, figsize=(13, 20))
    fig.suptitle(
        f"IV artifact removal\n"
        f"f₀={meta['f0']:.5f} Hz  f_refined={meta['f_refined']:.5f} Hz  "
        f"A={meta['A0']*1e9:.0f} nV\n"
        f"d²φ threshold={np.degrees(meta['fixed_thresh']):.1f} deg/s²  "
        f"valid={meta['valid'].sum()}/{len(meta['valid'])} windows",
        fontsize=11, fontweight='bold'
    )

    # Phase track
    ax = axes[0]; ax2 = ax.twinx()
    sc = ax.scatter(wt, np.degrees(phi_det), c=snr, cmap='RdYlGn',
                    s=15, zorder=4, vmin=0, vmax=np.percentile(snr, 90))
    ax.scatter(wt[~valid], np.degrees(phi_det[~valid]),
               color='red', s=30, marker='x', zorder=5, label='excluded')
    ax.plot(t, np.degrees(phi_smooth), color='steelblue', lw=1.5, label='spline')
    ax2.plot(t, np.degrees(np.gradient(phi_smooth, t)),
             color='orange', lw=1, alpha=0.6, label='dφ/dt (deg/s)')
    ax.axhline(0, color='k', lw=0.5)
    plt.colorbar(sc, ax=ax, label='SNR', fraction=0.02)
    ax.set_ylabel('Δφ (deg)'); ax2.set_ylabel('dφ/dt (deg/s)', color='orange')
    ax.set_title('Detrended phase  (colour = SNR,  red × = excluded)')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # Differential — full record
    ax = axes[1]
    ax.plot(t, dv_r*1e9, lw=0.8, color='steelblue', alpha=0.5, label='Raw')
    ax.plot(t, dv_c*1e9, lw=0.9, color='darkgreen',             label='Cleaned')
    ylim = np.percentile(np.abs(dv_r*1e9), 97)
    ax.set_ylim(-ylim*0.3, ylim*1.2); ax.set_ylabel('nV/sample')
    ax.set_title('Smoothed differential'); ax.legend(fontsize=8)

    # Zoom second quarter
    ax = axes[2]
    q1, q2 = int(N*0.25), int(N*0.55)  # approximate flat region
    ax.plot(t[q1:q2], voltage[q1:q2]*1e6,       lw=0.8, color='steelblue', alpha=0.5, label='Raw')
    ax.plot(t[q1:q2], voltage_clean[q1:q2]*1e6, lw=1.0, color='darkgreen',             label='Cleaned')
    ax.set_ylabel('Voltage (µV)'); ax.set_title('Zoom 25–55 % of record'); ax.legend(fontsize=8)

    # Zoom first quarter
    ax = axes[3]
    q0 = int(N*0.05); q1 = int(N*0.42)
    ax.plot(t[q0:q1], voltage[q0:q1]*1e6,       lw=0.8, color='steelblue', alpha=0.5, label='Raw')
    ax.plot(t[q0:q1], voltage_clean[q0:q1]*1e6, lw=1.0, color='darkgreen',             label='Cleaned')
    ax.set_ylabel('Voltage (µV)'); ax.set_title('Zoom 5–42 % of record'); ax.legend(fontsize=8)

    # Full IV
    ax = axes[4]
    ax.plot(current*1e6, voltage*1e6,       lw=0.8, color='steelblue', alpha=0.5, label='Raw')
    ax.plot(current*1e6, voltage_clean*1e6, lw=1.0, color='darkgreen',             label='Cleaned')
    ax.set_xlabel('Current (µA)'); ax.set_ylabel('Voltage (µV)')
    ax.set_title('Full IV curve'); ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    from scipy.ndimage import uniform_filter1d

    parser = argparse.ArgumentParser(description='Remove sinusoidal artifact from IV data')
    parser.add_argument('input',  help='Input  .dat file  (current, voltage)')
    parser.add_argument('output', help='Output .dat file  (current, voltage_clean)')
    parser.add_argument('--duration',    type=float, default=None, help='Sweep duration (s)')
    parser.add_argument('--fs',          type=float, default=None, help='Sample rate (Hz)')
    parser.add_argument('--f-lo',        type=float, default=1.30, help='Freq scan lower bound (Hz)')
    parser.add_argument('--f-hi',        type=float, default=1.45, help='Freq scan upper bound (Hz)')
    parser.add_argument('--thresh-mult', type=float, default=5.0,  help='d²φ threshold multiplier')
    parser.add_argument('--plot', action='store_true', help='Save diagnostic PNG alongside output')
    args = parser.parse_args()

    data    = np.loadtxt(args.input, comments='#')
    current = data[:, 0];  voltage = data[:, 1]

    vc, meta = remove_artifact(
        voltage, current,
        duration=args.duration, fs=args.fs,
        f_scan_lo=args.f_lo, f_scan_hi=args.f_hi,
        thresh_mult=args.thresh_mult,
    )

    np.savetxt(args.output,
               np.column_stack([current, vc]),
               fmt='%.6e', delimiter='\t',
               header='current(A)\tvoltage_clean(V)')
    print(f'Saved: {args.output}')

    t    = meta['t']
    dv_r = uniform_filter1d(np.gradient(voltage), size=6)
    dv_c = uniform_filter1d(np.gradient(vc),       size=6)
    total = t[-1]

    print(f"\nf0          = {meta['f0']:.6f} Hz")
    print(f"f_refined   = {meta['f_refined']:.6f} Hz")
    print(f"A0          = {meta['A0']*1e9:.1f} nV")
    print(f"d²φ thresh  = {np.degrees(meta['fixed_thresh']):.2f} deg/s²  "
          f"(= {args.thresh_mult:.1f} × median)")
    print(f"Valid wins  = {meta['valid'].sum()} / {len(meta['valid'])}")
    print()
    print(f"{'Region':<14} {'Raw (nV/s)':>12} {'Clean (nV/s)':>13} {'Suppression':>12}")
    for label, ts, te in [('First third',  0,         total/3),
                           ('Middle third', total/3,   2*total/3),
                           ('Last third',   2*total/3, total)]:
        seg = (t >= ts) & (t < te)
        r   = np.std(dv_r[seg]); c = np.std(dv_c[seg])
        print(f"  {label:<12} {r*1e9:>12.1f} {c*1e9:>13.1f} {r/(c+1e-30):>11.1f}x")

    if args.plot:
        plot_path = args.output.replace('.dat', '_diagnostic.png')
        plot_results(voltage, current, vc, meta, save_path=plot_path)
        print(f'\nDiagnostic plot: {plot_path}')