"""
killpulsetubenoise.py
======================
Remove a sinusoidal interference artifact from IV-curve measurements of
superconducting devices.

Background
----------
The artifact is a ~1.4 Hz sine (e.g. mains-related mechanical or electrical
pickup) superimposed on the measured voltage.  Key properties exploited:

  * The artifact is physically independent of the device, so its phase is
    CONTINUOUS across sharp voltage jumps in the IV curve.
  * The true frequency is close to but not exactly 1.4 Hz and must be
    estimated from the data.
  * The amplitude and phase may drift slowly over the sweep duration
    (slow_drift compensation).
  * The IV curve baseline is captured per-segment (between transitions) as
    a low-order polynomial in normalised current, which is non-degenerate
    with the time-domain sine.

Algorithm
---------
1. Detect voltage transitions (sharp jumps) and build a spike mask.
2. Estimate the true artifact frequency via sliding-window phase tracking
   in the (Gaussian-smoothed, differentiated) signal.
3. Build a single global design matrix:
     - Per-segment polynomial in normalised current  (IV baseline)
     - Chebyshev-modulated sin/cos carriers           (slowly-varying sine)
       S(t)*sin(2π f t) + C(t)*cos(2π f t)
       where S(t), C(t) are degree-`envelope_degree` Chebyshev polynomials.
4. Solve one linear least-squares problem on the clean (non-spike) samples.
5. Subtract the reconstructed artifact from the raw voltage.

Usage
-----
Standalone (file in, file out):
    python killpulsetubenoise.py input.dat output.dat

As a library:
    from killpulsetubenoise import remove_artifact
    voltage_clean, meta = remove_artifact(voltage, current, fs=18.77)

Input file format
-----------------
ASCII, two tab-separated columns, comment lines start with '#':
    column 0 : source-drain current  (A)
    column 1 : drain voltage         (V)
"""

import numpy as np
from scipy.ndimage import uniform_filter1d, binary_dilation, gaussian_filter1d
from scipy.signal import find_peaks


# ──────────────────────────────────────────────────────────────────────────────
#  Blind frequency discovery
# ──────────────────────────────────────────────────────────────────────────────

def discover_frequency(voltage, fs, f_min=0.1, f_max=None, verbose=True):
    """
    Blindly discover the dominant periodic interference frequency from a
    voltage trace, with no prior knowledge of the expected frequency.

    Method: Gaussian smooth (sigma=1) -> gradient -> FFT -> find dominant
    non-DC spectral peak.  The Gaussian+gradient step suppresses the slow
    IV-curve drift (low frequency) and high-frequency noise, leaving the
    periodic artifact as the clearest spectral feature.

    Parameters
    ----------
    voltage : 1-D array  (V)
    fs      : float      Sample rate (S/s)
    f_min   : float      Lower bound for candidate peaks (Hz, default 0.1)
    f_max   : float      Upper bound (Hz, default fs/4)
    verbose : bool

    Returns
    -------
    f_peak : float   Frequency of the dominant peak (Hz)
    meta   : dict    Keys: freqs, amplitude_spectrum, peak_bin, peak_prominence
    """
    voltage = np.asarray(voltage, dtype=float)
    N       = len(voltage)
    if f_max is None:
        f_max = fs / 4.0

    # Gaussian smooth + gradient: amplifies periodic signal, kills slow drift
    smoothed  = gaussian_filter1d(voltage, sigma=1)
    grad      = np.gradient(smoothed)
    spectrum  = np.fft.rfft(grad)
    freqs     = np.fft.rfftfreq(N, d=1.0 / fs)
    amplitude = np.abs(spectrum)

    # Restrict search to the requested band
    band           = (freqs >= f_min) & (freqs <= f_max)
    amp_band       = amplitude.copy()
    amp_band[~band] = 0.0

    peaks, props = find_peaks(amp_band, prominence=amp_band[band].max() * 0.05)
    if len(peaks) == 0:
        raise RuntimeError(
            f"No spectral peak found between {f_min} and {f_max} Hz. "
            "Check f_min/f_max or verify the data contains a periodic artifact."
        )

    best       = peaks[np.argmax(props["prominences"])]
    f_peak     = float(freqs[best])
    prominence = float(props["prominences"][np.argmax(props["prominences"])])

    if verbose:
        print("Frequency discovery (Gaussian -> gradient -> FFT):")
        print(f"  Dominant peak : bin {best},  f = {f_peak:.5f} Hz")
        print(f"  Amplitude     : {amplitude[best]:.4e}")
        print(f"  Prominence    : {prominence:.4e}  "
              f"({100*prominence / amp_band[band].max():.0f}% of band max)")
        order = np.argsort(props["prominences"])[::-1][:4]
        print("  Top peaks     : " +
              "  ".join(f"{freqs[peaks[i]]:.4f} Hz" for i in order))
        print()

    return f_peak, dict(
        freqs              = freqs,
        amplitude_spectrum = amplitude,
        peak_bin           = int(best),
        peak_prominence    = prominence,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def remove_artifact(
    voltage,
    current,
    fs,
    f_nominal        = None,
    envelope_degree  = 4,
    poly_iv_degree   = 4,
    spike_thresh_mad = 8,
    spike_dilation   = 20,
    verbose          = True,
):
    """
    Remove a sinusoidal artifact from a voltage trace.

    Parameters
    ----------
    voltage : 1-D array  (V)
    current : 1-D array  (A), same length as voltage
    fs      : float      Sample rate (samples / second)

    f_nominal        : float or None
                        Expected artifact frequency (Hz).
                        If None (default), the frequency is discovered
                        automatically via Gaussian-smooth -> gradient -> FFT
                        peak detection (no prior knowledge required).
                        Supply a value to skip discovery and seed the
                        phase-tracking refinement directly — useful when
                        the frequency is known from a reference channel or
                        a previous sweep in the same dataset.
    envelope_degree  : int    Chebyshev degree for slow amplitude/phase drift.
                              0 = static sine, 3-5 = recommended for long sweeps.
    poly_iv_degree   : int    Polynomial degree for IV baseline per segment.
    spike_thresh_mad : float  Threshold for transition detection (× MAD of |dV/dn|).
    spike_dilation   : int    Half-width (samples) of the exclusion zone around
                              each detected transition.
    verbose          : bool

    Returns
    -------
    voltage_clean : 1-D array   Corrected voltage (V)
    meta          : dict        Fit diagnostics (see keys below)

    meta keys
    ---------
    f_nominal       : float    Seed frequency (discovered or supplied, Hz)
    f_true          : float    Refined artifact frequency (Hz)
    amplitude_mean  : float    Mean artifact amplitude (V)
    amplitude_std   : float    Std of slowly-varying amplitude (V)
    phase_swing_deg : float    Peak-to-peak phase variation (degrees)
    condition       : float    Condition number of design matrix
    envelope_degree : int      As supplied
    segments        : list     [(start, end), ...] clean plateau segments
    artifact        : 1-D array  The subtracted waveform (V)
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    N = len(voltage)
    t = np.arange(N) / fs


    # ── 0. Blind frequency discovery (when no nominal given) ────────────────────
    if f_nominal is None:
        if verbose:
            print('f_nominal not supplied — running blind frequency discovery …')
        f_nominal, _disc_meta = discover_frequency(voltage, fs, verbose=verbose)
    # ── 1. Transition detection ───────────────────────────────────────────────
    dv        = np.gradient(voltage)
    dv_smooth = uniform_filter1d(dv, size=6)
    abs_dv    = np.abs(dv_smooth)
    mad       = np.median(np.abs(abs_dv - np.median(abs_dv)))
    thresh    = np.median(abs_dv) + spike_thresh_mad * mad

    spikes_narrow = binary_dilation(abs_dv > thresh, iterations=4)
    spikes_wide   = binary_dilation(abs_dv > thresh, iterations=spike_dilation)
    edges         = np.where(np.diff(spikes_narrow.astype(int)))[0]

    # Build plateau segments: clean regions between transitions
    boundaries = [0] + list(edges) + [N]
    segments   = []
    for i in range(0, len(boundaries) - 1, 2):
        s = boundaries[i]
        e = boundaries[i + 1] if i + 1 < len(boundaries) else N
        if (e - s) > int(2 * fs / f_nominal):   # need at least 2 cycles
            segments.append((int(s), int(e)))

    if not segments:
        raise RuntimeError("No clean segments found — check spike_thresh_mad.")

    # Fit mask: inside segments, outside dilated spikes
    mask = np.zeros(N, bool)
    for s, e in segments:
        mask[s:e] = True
    mask &= ~spikes_wide

    if verbose:
        print(f"Segments : {[(s, e) for s,e in segments]}")
        print(f"Clean    : {mask.sum()} / {N} samples ({100*mask.mean():.1f} %)")

    # ── 2. Frequency estimation (phase tracking in differential domain) ───────
    dv_sg   = uniform_filter1d(dv_smooth, size=81)   # slow IV slope
    dv_res  = dv_smooth - dv_sg
    win_ph  = int(4 * fs / f_nominal)
    step_ph = win_ph // 4
    wt_ph, wph = [], []
    for s in range(0, N - win_ph, step_ph):
        e = s + win_ph
        if spikes_wide[s:e].any():
            continue
        Am = np.column_stack([np.cos(2*np.pi*f_nominal*t[s:e]),
                              np.sin(2*np.pi*f_nominal*t[s:e])])
        c, _, _, _ = np.linalg.lstsq(Am, dv_res[s:e], rcond=None)
        wt_ph.append(t[s + win_ph // 2])
        wph.append(np.arctan2(-c[1], c[0]))

    if len(wph) < 3:
        raise RuntimeError("Too few clean windows for frequency refinement.")

    sl, _ = np.polyfit(np.array(wt_ph), np.unwrap(np.array(wph)), 1)
    f_true = f_nominal + sl / (2 * np.pi)

    if verbose:
        print(f"f_nominal: {f_nominal:.6f} Hz  (coarse, from FFT peak)")
        print(f"f_true   : {f_true:.6f} Hz  "
              f"(refined; offset {(f_true - f_nominal)*1000:+.3f} mHz)")

    # ── 3. Design matrix ──────────────────────────────────────────────────────
    t_norm      = 2*(t - t[0])/(t[-1] - t[0]) - 1      # [-1, 1]
    sin_carrier = np.sin(2*np.pi*f_true*t)
    cos_carrier = np.cos(2*np.pi*f_true*t)
    cheby       = np.polynomial.chebyshev.chebvander(t_norm, envelope_degree)
    # (N, envelope_degree+1)

    # IV baseline block: per-segment polynomial in normalised current
    n_iv = len(segments) * (poly_iv_degree + 1)
    M_iv = np.zeros((N, n_iv))
    for k, (s, e) in enumerate(segments):
        I_seg  = current[s:e]
        span   = I_seg.max() - I_seg.min()
        I_norm = (I_seg - I_seg.mean()) / (span if span > 0 else 1.0)
        for d in range(poly_iv_degree + 1):
            M_iv[s:e, k*(poly_iv_degree+1) + d] = I_norm**d

    # Sine envelope block: [T_0*sin … T_K*sin | T_0*cos … T_K*cos]
    M_sin = cheby * sin_carrier[:, None]
    M_cos = cheby * cos_carrier[:, None]

    M = np.hstack([M_iv, M_sin, M_cos])

    # ── 4. Solve ──────────────────────────────────────────────────────────────
    coeffs, _, rank, sv = np.linalg.lstsq(M[mask], voltage[mask], rcond=None)
    condition = float(sv[0] / sv[-1])

    ne = envelope_degree + 1
    S_t = cheby @ coeffs[n_iv       : n_iv +   ne]
    C_t = cheby @ coeffs[n_iv + ne  : n_iv + 2*ne]

    A_t   = np.sqrt(S_t**2 + C_t**2)
    phi_t = np.arctan2(C_t, S_t)

    if verbose:
        print(f"Amplitude: {A_t.mean()*1e9:.1f} ± {A_t.std()*1e9:.1f} nV  "
              f"(range {A_t.min()*1e9:.1f}–{A_t.max()*1e9:.1f} nV)")
        print(f"Phase swing: {np.degrees(phi_t.max()-phi_t.min()):.1f}°")
        print(f"Condition: {condition:.2e}  rank {rank}/{M.shape[1]}")

    # ── 5. Subtract ───────────────────────────────────────────────────────────
    artifact      = S_t * sin_carrier + C_t * cos_carrier
    voltage_clean = voltage - artifact

    meta = dict(
        f_nominal       = f_nominal,
        f_true          = f_true,
        amplitude_mean  = float(A_t.mean()),
        amplitude_std   = float(A_t.std()),
        phase_swing_deg = float(np.degrees(phi_t.max() - phi_t.min())),
        condition       = condition,
        envelope_degree = envelope_degree,
        segments        = segments,
        artifact        = artifact,
        A_t             = A_t,
        phi_t           = phi_t,
    )
    return voltage_clean, meta


def plot_results(current, voltage, voltage_clean, meta, fs, save_path=None):
    """
    Diagnostic plot: IV curve, smoothed differential, amplitude/phase envelope,
    and zoom around the largest transition.
    """
    import matplotlib.pyplot as plt

    N  = len(voltage)
    t  = np.arange(N) / fs
    dv_raw   = uniform_filter1d(np.gradient(voltage),       size=6)
    dv_clean = uniform_filter1d(np.gradient(voltage_clean), size=6)

    fig, axes = plt.subplots(4, 1, figsize=(13, 16))
    f_true = meta['f_true']
    fig.suptitle(
        f'IV Artifact Removal\n'
        f'f = {f_true:.5f} Hz   '
        f'A = {meta["amplitude_mean"]*1e9:.0f} ± {meta["amplitude_std"]*1e9:.0f} nV   '
        f'phase swing = {meta["phase_swing_deg"]:.1f}°   '
        f'envelope degree = {meta["envelope_degree"]}',
        fontsize=11, fontweight='bold',
    )

    # Panel 1: slowly-varying amplitude and phase
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(t, meta['A_t']*1e9,            lw=1.5, color='steelblue', label='A(t) (nV)')
    ax2.plot(t, np.degrees(meta['phi_t']), lw=1.5, color='orange',    label='φ(t) (°)')
    ax.set_ylabel('Amplitude (nV)', color='steelblue')
    ax2.set_ylabel('Phase (°)',     color='orange')
    ax.set_title('Slowly-varying sine envelope')
    ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)

    # Panel 2: full smoothed differential
    ax = axes[1]
    ax.plot(t, dv_raw*1e9,   lw=0.8, color='steelblue', alpha=0.5, label='Raw dV/dn')
    ax.plot(t, dv_clean*1e9, lw=0.9, color='darkgreen',            label='Cleaned dV/dn')
    ylim = np.percentile(np.abs(dv_raw*1e9), 97)
    ax.set_ylim(-ylim*0.3, ylim*1.2)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('nV/sample')
    ax.set_title('Smoothed differential: before vs after')
    ax.legend(fontsize=8)

    # Panel 3: IV curve
    ax = axes[2]
    ax.plot(current*1e6, voltage*1e6,       lw=0.8, color='steelblue', alpha=0.5, label='Original')
    ax.plot(current*1e6, voltage_clean*1e6, lw=1.0, color='darkgreen',            label='Cleaned')
    ax.set_xlabel('Current (µA)'); ax.set_ylabel('Voltage (µV)')
    ax.set_title('IV curve before and after artifact removal')
    ax.legend(fontsize=8)

    # Panel 4: zoom around largest transition
    segs = meta['segments']
    if len(segs) >= 2:
        # gap between first two segments
        gap_centre = (segs[0][1] + segs[1][0]) // 2
        hw = int(3 * fs)   # ±3 s window
        sl = slice(max(0, gap_centre-hw), min(N, gap_centre+hw))
        ax = axes[3]
        ax.plot(current[sl]*1e6, voltage[sl]*1e6,
                lw=0.8, color='steelblue', alpha=0.5, label='Original')
        ax.plot(current[sl]*1e6, voltage_clean[sl]*1e6,
                lw=1.0, color='darkgreen', label='Cleaned')
        ax.set_xlabel('Current (µA)'); ax.set_ylabel('Voltage (µV)')
        ax.set_title('Zoom around first transition — step should be clean, no ringing')
        ax.legend(fontsize=8)
    else:
        axes[3].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if save_path: print(f"Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
#  Command-line entry point
# ──────────────────────────────────────────────────────────────────────────────

def _main():
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Remove sinusoidal interference from an IV-curve dat file."
    )
    parser.add_argument("input",  help="Input  .dat file (two-column ASCII)")
    parser.add_argument("output", help="Output .dat file (cleaned voltage)")
    parser.add_argument("--fs",             type=float, default=None,
                        help="Sample rate (S/s). If omitted, estimated from "
                             "--duration or --n-samples + --duration.")
    parser.add_argument("--duration",       type=float, default=None,
                        help="Total sweep duration (s). Used with N to compute fs.")
    parser.add_argument("--f-nominal",      type=float, default=None,
                        help="Nominal artifact frequency in Hz. If omitted, "
                             "the frequency is discovered automatically from "
                             "the data (Gaussian -> gradient -> FFT).")
    parser.add_argument("--envelope-degree",type=int,   default=4,
                        help="Chebyshev degree for slow drift compensation "
                             "(0=static, 3-5 recommended; default 4).")
    parser.add_argument("--poly-iv-degree", type=int,   default=4,
                        help="IV baseline polynomial degree per segment (default 4).")
    parser.add_argument("--plot",           action="store_true",
                        help="Save a diagnostic PNG alongside the output file.")
    args = parser.parse_args()

    # Load
    data    = np.loadtxt(args.input, comments='#')
    current = data[:, 0]
    voltage = data[:, 1]
    N       = len(voltage)

    # Sample rate
    if args.fs is not None:
        fs = args.fs
    elif args.duration is not None:
        fs = N / args.duration
    else:
        sys.exit("ERROR: supply --fs or --duration so the sample rate is known.")

    print(f"Input  : {args.input}  ({N} samples, fs={fs:.4f} S/s)")

    # Run
    voltage_clean, meta = remove_artifact(
        voltage, current, fs,
        f_nominal       = args.f_nominal,
        envelope_degree = args.envelope_degree,
        poly_iv_degree  = args.poly_iv_degree,
    )

    # Save cleaned data
    header = (
        f"Artifact removed by iv_artifact_removal.py\n"
        f"f_true={meta['f_true']:.6f} Hz  "
        f"A={meta['amplitude_mean']*1e9:.1f} nV  "
        f"envelope_degree={meta['envelope_degree']}\n"
        f"Current (A)\tVoltage_cleaned (V)"
    )
    np.savetxt(args.output,
               np.column_stack([current, voltage_clean]),
               header=header, delimiter='\t')
    print(f"Output : {args.output}")

    # Optional plot
    if args.plot:
        plot_path = args.output.replace('.dat', '_diagnostic.png')
        plot_results(current, voltage, voltage_clean, meta, fs,
                     save_path=plot_path)


if __name__ == "__main__":
    _main()
