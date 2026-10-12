"""
Mock Chandra ACIS-I mosaics of a BAHAMAS (or FLAMINGO, via
FLAMINGO/flamingo_io.py, which hands over the same fields in the same units)
cluster, ported from
Sandbox_notebooks/mock_xray_maps/init_xray_map_code.ipynb.

Two changes from the notebook:
  - the cluster is centred on its BCG rather than on the FoF centre of potential,
  - the projection is the sample's viewing direction rather than the box z-axis,
    handed to pyXSIM as (normal=w, north_vector=v) so the mosaic lands in the
    same frame as the galaxy and lensing channels.

Every assumption the notebook tagged with FLAG: is carried over unchanged and
lives in CFG_XRAY; read the notebook's caveats section before trusting a number
that comes out of here.
"""
import os
import glob
import numpy as np

# SOXS must be configured before it is imported (it sets up its config and
# downloads response files on import).
from soxs.utils import soxs_cfg
soxs_cfg.set("soxs", "bkgnd_nH", "0.018")
import soxs
import yt
import pyxsim
from yt.utilities.cosmology import Cosmology
from astropy.io import fits

from config import (CFG_XRAY, CFG_COSMO, CFG_DATA, BAHAMAS_BOXSIZE_MPC_OVER_H,
                    cluster_boxsize_mpc_over_h)

# --- physical constants (CGS), as in the notebook ---
MP = 1.67262192369e-24        # proton mass, g
MSUN = 1.98892e33             # solar mass, g (yt's value)
KPC = 3.0856775809623245e21   # cm
MPC = 1.0e3 * KPC
KEV_TO_ERG = 1.602176634e-9
K_TO_KEV = 8.617333262e-8
# Critical density at z=0 in Msun/Mpc^3, divided by h^2.
RHO_CRIT_OVER_H2 = 2.77536627e11

# BAHAMAS box (cMpc/h). Kept as the fallback only: every function below reads the
# box from the loaded cluster (config.cluster_boxsize_mpc_over_h), so FLAMINGO's
# 1 and 2.8 Gpc boxes wrap correctly while BAHAMAS files, which carry no box
# size, still get 400.
BOXSIZE_MPC_OVER_H = BAHAMAS_BOXSIZE_MPC_OVER_H

# The sky-plane orientation pyXSIM gives a (normal, north_vector) pair, verified
# against the projected gas mass by tests/check_xray_alignment.py:
#   image East  = -u  (u = viewing-frame +x),  image North = +v.
# Maps are stored in the viewing frame, so East is flipped back onto +u.
EAST_IS_MINUS_U = True


def quiet_logs():
    """pyXSIM/SOXS/yt are chatty; a 10k-sample run drowns in progress bars."""
    yt.set_log_level("error")
    for name in ("pyxsim", "soxs", "yt"):
        import logging
        logging.getLogger(name).setLevel(logging.ERROR)


def bcg_index(data, boxsize=None) -> int:
    """
    Index of the BCG: the most massive stellar subhalo within
    CFG_DATA.bcg_search_radius_r200 * R200 of the FoF centre of potential.

    Shared by the photon stage and the dataset build so both centre the cluster
    identically. The aperture matters: a global argmax picks a neighbouring
    cluster's BCG, several Mpc away, in a handful of the 5 R200 cutouts.
    """
    if boxsize is None:
        boxsize = cluster_boxsize_mpc_over_h(data)
    sub_mstar = data["sub_massTotal"][:, 4]
    d_cop = data["sub_pos"] - data["CoP"]
    d_cop = (d_cop + 0.5 * boxsize) % boxsize - 0.5 * boxsize
    near_center = (np.linalg.norm(d_cop, axis=1)
                   <= CFG_DATA.bcg_search_radius_r200 * float(data["R200"]))
    if not np.any(near_center):
        near_center = np.ones(sub_mstar.shape, dtype=bool)
    near_inds = np.flatnonzero(near_center)
    return int(near_inds[np.argmax(sub_mstar[near_inds])])


def viewing_basis(proj_vec):
    """
    The (u, v, w) basis TNG_DA.rotate_to_viewing_frame builds for this viewing
    direction: u is the image +x axis, v the image +y axis, w the line of sight.

    Kept byte-for-byte identical to that function; any change there has to be
    mirrored here, or the X-ray maps end up rotated relative to the galaxy and
    lensing channels.
    """
    w = np.asarray(proj_vec, dtype=np.float64)
    w = w / np.linalg.norm(w)
    a = np.zeros(3)
    a[np.argmin(np.abs(w))] = 1.0
    u = np.cross(w, a)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return u, v, w


def photon_sim_dir(sim, cfg=CFG_XRAY):
    """The photon-list directory for one dark-matter model."""
    return os.path.join(cfg.photon_dir, str(sim))


def photon_prefix(sim, cluster_idx, proj_index, cfg=CFG_XRAY):
    """
    Where the photon stage writes, and the dataset build reads, the photon list
    for one sample:

        <photon_dir>/<sim>/<sim>_<stem>_<cluster>_p<projection>_photons.h5

    with <stem> = CFG_DATA.cluster_file_stem ('GrNm' for BAHAMAS, 'cluster'
    for FLAMINGO), matching the cluster file the list was made from.

    Keyed on the sample's identity, not on its redshift: two projections of a
    cluster can draw z_lens values that agree to four decimals, and a
    redshift-formatted name would silently collide. The redshift each list was
    actually built at is recorded inside the file as parameters/fid_redshift and
    checked on read against config.sample_z_lens.

    The model name is repeated in the filename even though the directory already
    carries it, so a list stays identifiable if it is ever copied out on its own.
    """
    stem = CFG_DATA.cluster_file_stem
    return os.path.join(photon_sim_dir(sim, cfg),
                        f"{sim}_{stem}_{cluster_idx:03d}_p{int(proj_index):d}_photons")


def photon_list_redshift(prefix):
    """The redshift a photon list was generated at, or nan if unreadable."""
    try:
        import h5py
        with h5py.File(prefix + ".h5", "r") as f:
            return float(f["/parameters/fid_redshift"][()])
    except Exception:
        return float("nan")


def density_unit_cgs(h, a):
    """
    gas_dens is Gadget code density, 1e10 Msun/h per (cMpc/h)^3 -> g/cm^3.

    Equals h^2 a^-3 * 6.770e-31, which reproduces the GADGET CGSConversionFactor
    6.76991e-31 quoted for these BAHAMAS snapshots to 4e-5. The particle *masses*,
    by contrast, already have h divided out: dm_mass = 5.5065e9 Msun is exactly
    the published 3.85e9 h^-1 Msun at h = 0.7, and gas_mass's floor of 1.0942e9
    is the published 7.66e8 h^-1 Msun. So gas_dens needs the h^2 and the masses
    need nothing -- see the units check in tests/.
    """
    return (1.0e10 * MSUN / h) / (MPC / h) ** 3 / a ** 3


def bahamas_cosmology(h, Om0=None):
    """
    The simulation's flat cosmology for yt/pyXSIM. Om0 defaults to the BAHAMAS
    value; FLAMINGO passes each run's own (config.sim_cosmology).
    """
    if Om0 is None:
        Om0 = CFG_COSMO.Om0_sim
    return Cosmology(hubble_constant=h,
                     omega_matter=Om0,
                     omega_lambda=1.0 - Om0)


def load_gas_dataset(data, center, cfg=CFG_XRAY):
    """
    Gas within cfg.n_R200 * R200 of `center` (the BCG), as an in-memory yt SPH
    dataset with the fields pyXSIM needs. Returns (ds, sphere_radius_pMpc).

    `center` is in comoving Mpc/h, like data['sub_pos'].
    """
    BOX = cluster_boxsize_mpc_over_h(data)

    h = float(data["h"])
    a = float(data["a"])
    R200 = float(data["R200"])                      # comoving Mpc/h

    dx_all = (data["gas_pos"] - center + 0.5 * BOX) % BOX - 0.5 * BOX
    r_all = np.linalg.norm(dx_all, axis=1)
    sel = r_all < cfg.n_R200 * R200

    pos_kpc = dx_all[sel] * (a / h) * 1.0e3         # physical kpc, cluster-centric
    vel = data["gas_vel"][sel]                      # km/s
    mass_g = data["gas_mass"][sel] * MSUN           # g
    rho = data["gas_dens"][sel] * density_unit_cgs(h, a)   # physical g/cm^3
    temp = data["gas_temp"][sel]                    # K

    X_H = cfg.X_H
    Y_He = 1.0 - X_H
    volume = mass_g / rho                           # cm^3, SPH volume
    nH = X_H * rho / MP
    ne = (X_H + 0.5 * Y_He) * rho / MP              # fully ionised H + He
    emission_measure = ne * nH * volume

    smoothing_length = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0) / KPC
    smoothing_length = np.maximum(smoothing_length, cfg.eps_grav_kpc)

    fields = {
        ("gas", "particle_position_x"): (pos_kpc[:, 0], "kpc"),
        ("gas", "particle_position_y"): (pos_kpc[:, 1], "kpc"),
        ("gas", "particle_position_z"): (pos_kpc[:, 2], "kpc"),
        ("gas", "particle_velocity_x"): (vel[:, 0], "km/s"),
        ("gas", "particle_velocity_y"): (vel[:, 1], "km/s"),
        ("gas", "particle_velocity_z"): (vel[:, 2], "km/s"),
        ("gas", "particle_mass"): (mass_g, "g"),
        ("gas", "density"): (rho, "g/cm**3"),
        ("gas", "temperature"): (temp, "K"),
        ("gas", "smoothing_length"): (smoothing_length, "kpc"),
        ("gas", "emission_measure"): (emission_measure, "cm**-3"),
    }

    half = cfg.n_R200 * R200 * (a / h) * 1.0e3      # physical kpc
    bbox = np.array([[-half, half], [-half, half], [-half, half]])
    ds = yt.load_particles(fields, length_unit=(1.0, "kpc"), mass_unit=(1.0, "g"),
                           velocity_unit=(1.0, "km/s"), time_unit=(1.0, "Myr"),
                           bbox=bbox, periodicity=(False, False, False))

    rho_max = cfg.nH_max * MP / X_H

    def hot_gas(pfilter, data_):
        t = data_[pfilter.filtered_type, "temperature"].to_value("K") > cfg.T_min_K
        d = data_[pfilter.filtered_type, "density"].to_value("g/cm**3") < rho_max
        return t & d

    yt.add_particle_filter("hot_gas", function=hot_gas, filtered_type="gas",
                           requires=["temperature", "density"])
    ds.add_particle_filter("hot_gas")

    return ds, cfg.n_R200 * R200 * a / h            # physical Mpc


# ============================================================
# Intrinsic L_X, for checking the sample against LoVoCCS's X-ray selection
# ============================================================

_LAMBDA_TABLE = None


def cooling_function_table(cfg=CFG_XRAY):
    """
    Band-limited APEC energy cooling function Lambda(T) over cfg.lx_band_kev, in
    erg cm^3 s^-1, as (T_grid_K, Lambda_grid).

    Built with SOXS's ApecGenerator rather than pyatomdb: it is the same
    APEC v3.1.3 table (soxs already caches apec_v3.1.3_{coco,line}.fits), so
    there is no extra dependency and nothing to download.

    Lambda is defined by L_X = Lambda(T, Z) * EM with EM = n_e n_H V, which is
    the convention the emission_measure field already uses.
    """
    global _LAMBDA_TABLE
    if _LAMBDA_TABLE is not None:
        return _LAMBDA_TABLE

    from soxs.spectra import ApecGenerator

    emin, emax = cfg.lx_band_kev
    agen = ApecGenerator(emin, emax, cfg.lx_nbins, broadening=False)

    T_grid = np.logspace(np.log10(cfg.lx_T_grid_min_K),
                         np.log10(cfg.lx_T_grid_max_K), cfg.lx_T_grid_n)
    # A norm of 1e-14/(4 pi D^2) at z = 0 corresponds to EM = 1 cm^-3, so
    # 4 pi D^2 * (band-integrated energy flux) is Lambda directly.
    D = MPC
    norm = 1.0e-14 / (4.0 * np.pi * D ** 2)
    four_pi_d2 = 4.0 * np.pi * D ** 2

    Lambda = np.empty_like(T_grid)
    for i, T in enumerate(T_grid):
        sp = agen.get_spectrum(T * K_TO_KEV, cfg.Z_met, 0.0, norm)
        e = sp.emid.value                      # keV
        de = np.diff(sp.ebins.value)           # keV
        flux_erg = np.sum(sp.flux.value * de * e) * KEV_TO_ERG
        Lambda[i] = four_pi_d2 * flux_erg

    _LAMBDA_TABLE = (T_grid, Lambda)
    return _LAMBDA_TABLE


def cooling_function(T_K, cfg=CFG_XRAY):
    """
    Lambda(T) in erg cm^3 s^-1 for each temperature. Zero below the grid (gas
    that cold contributes nothing in band) and clamped to the top value above
    it, matching the treatment in the collaborator's APEC notebook.
    """
    T_grid, Lambda = cooling_function_table(cfg)
    out = np.interp(np.asarray(T_K, dtype=np.float64), T_grid, Lambda,
                    left=0.0, right=Lambda[-1])
    return np.where(np.asarray(T_K) < T_grid[0], 0.0, out)


def r500c_mpc_over_h(data, center, cfg=CFG_XRAY):
    """
    R500c about `center`, in comoving Mpc/h, from the particles themselves:
    the largest radius whose enclosed mean density is >= 500 rho_crit(z=0).

    The npz carries R200/M200 but not R500, and LoVoCCS quotes L_X inside R500c.
    """
    h = float(data["h"])
    a = float(data["a"])
    BOX = cluster_boxsize_mpc_over_h(data)

    r_list, m_list = [], []
    for pos_key, mass_key in (("dm_pos", "dm_mass"), ("gas_pos", "gas_mass"),
                              ("star_pos", "star_mass"), ("bh_pos", "bh_mass")):
        if pos_key not in data.files or mass_key not in data.files:
            continue
        if data[pos_key].shape[0] == 0:
            continue
        dx = (data[pos_key] - center + 0.5 * BOX) % BOX - 0.5 * BOX
        r_list.append(np.linalg.norm(dx, axis=1))
        m_list.append(data[mass_key])          # Msun, h already divided out
    r = np.concatenate(r_list)
    m = np.concatenate(m_list)

    order = np.argsort(r)
    r_sorted = r[order]                        # comoving Mpc/h
    m_cum = np.cumsum(m[order])                # Msun

    # Mean enclosed density in physical Msun/Mpc^3.
    r_phys = np.maximum(r_sorted * a / h, 1e-6)
    rho_mean = m_cum / ((4.0 / 3.0) * np.pi * r_phys ** 3)
    rho_500 = 500.0 * RHO_CRIT_OVER_H2 * h ** 2

    # The *innermost* crossing, not the outermost: a massive neighbour inside
    # the 5 R200 cutout can push the running mean back above 500 rho_c at large
    # radius, and the outermost crossing would then return its radius instead of
    # the cluster's. n_min skips the shot-noise-dominated first few particles,
    # where the running mean is meaningless.
    n_min = 100
    if r_sorted.size <= n_min:
        return float("nan")
    below = np.flatnonzero(rho_mean[n_min:] < rho_500)
    if below.size == 0:
        return float("nan")                      # never drops: cutout too small
    return float(r_sorted[n_min + below[0] - 1])


def lx500c_erg_s(data, center, cfg=CFG_XRAY):
    """
    Rest-frame L_X inside R500c over cfg.lx_band_kev, in erg s^-1, from the same
    hot-gas selection the photon lists use (T > T_min_K, n_H < nH_max).

    Recorded for comparison against LoVoCCS's L_X500c > 1e44 erg/s selection.
    It is a diagnostic only: no simulated cluster is dropped on it.

    Returns (L_X, R500c in comoving Mpc/h, number of hot particles used).
    """
    h = float(data["h"])
    a = float(data["a"])
    BOX = cluster_boxsize_mpc_over_h(data)

    r500 = r500c_mpc_over_h(data, center, cfg)
    if not np.isfinite(r500):
        return float("nan"), float("nan"), 0

    dx = (data["gas_pos"] - center + 0.5 * BOX) % BOX - 0.5 * BOX
    r = np.linalg.norm(dx, axis=1)

    rho = data["gas_dens"] * density_unit_cgs(h, a)      # physical g/cm^3
    T = data["gas_temp"]                                 # K
    X_H = cfg.X_H
    nH = X_H * rho / MP
    ne = (X_H + 0.5 * (1.0 - X_H)) * rho / MP

    sel = (r < r500) & (T > cfg.T_min_K) & (nH < cfg.nH_max)
    if not np.any(sel):
        return 0.0, r500, 0

    volume = (data["gas_mass"][sel] * MSUN) / rho[sel]   # cm^3
    em = ne[sel] * nH[sel] * volume                      # cm^-3
    lx = float(np.sum(em * cooling_function(T[sel], cfg)))
    return lx, r500, int(sel.sum())


def source_model(cfg=CFG_XRAY):
    return pyxsim.CIESourceModel(
        "apec", cfg.emin_kev, cfg.emax_kev, cfg.nbins,
        Zmet=cfg.Z_met,
        temperature_field=("hot_gas", "temperature"),
        emission_measure_field=("hot_gas", "emission_measure"),
        kT_min=cfg.kT_min_kev,
        h_fraction=cfg.X_H,
        thermal_broad=True,
    )


def photon_list_is_complete(prefix):
    """
    True if prefix + '.h5' exists and is a readable, fully written photon list.

    A task killed at the wall clock leaves a truncated .h5 behind. Since the
    photon stage skips bins whose file already exists, a bare existence check
    would let that corpse survive every restart and only fail later, in the
    dataset build. Opening it is cheap next to regenerating it.
    """
    path = prefix + ".h5"
    if not os.path.exists(path):
        return False
    try:
        import h5py
        with h5py.File(path, "r") as f:
            # data/energy is the largest array and the last one filled, so a
            # truncated write shows up as a length mismatch against the
            # per-cell photon counts.
            n_expected = int(f["/data/num_photons"][:].sum())
            return (f["/data/energy"].shape[0] == n_expected
                    and "/parameters/fid_exp_time" in f)
    except Exception:
        return False


def make_photon_list(prefix, ds, radius_mpc, redshift, cosmo, cfg=CFG_XRAY):
    """
    The expensive step (~2.5 min): projection-independent, redshift-dependent.

    Written to a scratch prefix and renamed into place, so an interrupted run
    can never leave a half-written .h5 that the restart mistakes for a finished
    one. os.replace is atomic within a filesystem.
    """
    c = ds.arr([0.0, 0.0, 0.0], "kpc")
    src = ds.sphere(c, (radius_mpc, "Mpc"))

    tmp_prefix = prefix + ".tmp"
    for stale in glob.glob(tmp_prefix + "*"):
        try:
            os.remove(stale)
        except OSError:
            pass

    try:
        n_ph, n_cells = pyxsim.make_photons(
            tmp_prefix, src, redshift,
            (cfg.sample_area_cm2, "cm**2"), (cfg.sample_exp_ks, "ks"),
            source_model(cfg), cosmology=cosmo, center=c,
        )
        os.replace(tmp_prefix + ".h5", prefix + ".h5")
    except BaseException:
        for junk in glob.glob(tmp_prefix + "*"):
            try:
                os.remove(junk)
            except OSError:
                pass
        raise

    return int(n_ph), int(n_cells)


def project_and_observe(photon_prefix, work_prefix, proj_vec, t_exp_ks,
                        cfg=CFG_XRAY):
    """
    Project the photon list along the sample's viewing direction and observe it
    for t_exp_ks with cfg.n_side^2 ACIS pointings (one, by default, matching the
    archival Chandra coverage a LoVoCCS cluster actually has).

    Returns the paths of the products, all under work_prefix.
    """
    u, v, w = viewing_basis(proj_vec)

    event_prefix = f"{work_prefix}_events"
    pyxsim.project_photons(
        photon_prefix, event_prefix, w, cfg.sky_center,
        absorb_model=cfg.absorb_model, nH=cfg.nH_gal, north_vector=v,
    )

    events = pyxsim.EventList(f"{event_prefix}.h5")

    # (a) the idealised map: every projected band photon, no instrument, written
    # straight onto the stored grid so it registers with the mosaic.
    ideal_path = f"{work_prefix}_img_ideal.fits"
    events.write_fits_image(ideal_path, (cfg.fov_arcmin, "arcmin"), cfg.grid_size,
                            emin=cfg.band_emin_kev, emax=cfg.band_emax_kev,
                            overwrite=True)

    # (c) the observation: SOXS points the instrument at the SIMPUT catalogue.
    # make_mosaic_events handles the single-pointing case too, so the products
    # and their names are the same whether n_side is 1 or 3.
    events.write_to_simput(work_prefix, overwrite=True)
    ra0, dec0 = cfg.sky_center
    pointings = [(ra0 + ddx / np.cos(np.deg2rad(dec0)), dec0 + ddy)
                 for ddx, ddy in cfg.pointing_offsets_deg()]

    mosaic_table = soxs.make_mosaic_events(
        pointings, f"{work_prefix}_simput.fits", f"{work_prefix}_mosaic",
        (t_exp_ks, "ks"), cfg.instrument, overwrite=True,
        instr_bkgnd=cfg.instr_bkgnd, foreground=cfg.foreground,
        ptsrc_bkgnd=cfg.ptsrc_bkgnd,
    )
    counts_path = f"{work_prefix}_mosaic_img.fits"
    soxs.make_mosaic_image(mosaic_table, counts_path,
                           emin=cfg.band_emin_kev, emax=cfg.band_emax_kev,
                           use_expmap=True, expmap_energy=1.0,
                           reblock=cfg.reblock, overwrite=True)

    # Per-pointing exposure, read back rather than taken from the config: the
    # exposure map is normalised (cm^2, not cm^2 s), so this is what turns
    # counts/expmap into a count rate.
    with fits.open(f"{work_prefix}_mosaic_0_evt.fits") as f:
        t_exp_s = float(f["EVENTS"].header["EXPOSURE"])

    return {
        "ideal": ideal_path,
        "counts": counts_path,
        "expmap": counts_path.replace(".fits", ".expmap"),
        "flux": counts_path.replace(".fits", ".flux"),
        "t_exp_s": t_exp_s,
    }


# ============================================================
# Resampling onto the stored grid
# ============================================================

def _pixel_sky_offsets(header):
    """
    Per-axis offsets from the reference pixel, in arcsec, as (east, north).

    SOXS and pyXSIM both write an unrotated tangent-plane WCS centred on the
    pointing centre, i.e. the cluster, so the offsets are separable and the
    CDELT signs are the only convention to respect.
    """
    nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
    cd1 = float(header.get("CDELT1", header.get("CD1_1")))
    cd2 = float(header.get("CDELT2", header.get("CD2_2")))
    crpix1, crpix2 = float(header["CRPIX1"]), float(header["CRPIX2"])

    # +RA is East; CDELT1 < 0 means image +x runs West.
    east = (np.arange(nx) + 1.0 - crpix1) * cd1 * 3600.0
    north = (np.arange(ny) + 1.0 - crpix2) * cd2 * 3600.0
    return east, north


def to_frame_grid(img, header, cfg=CFG_XRAY, how="sum"):
    """
    Aggregate a FITS image onto the stored (grid_size x grid_size) grid covering
    fov_arcmin about the pointing centre, in the viewing frame: image +x is the
    frame's +u, +y is +v.

    how='sum'  : counts (flux-conserving)
    how='mean' : exposure maps and already-normalised surface brightness
    """
    east, north = _pixel_sky_offsets(header)

    # Sky -> viewing frame. North is +v; East is -u for pyXSIM's (normal,
    # north_vector) convention, so +u = -East.
    x_arcsec = -east if EAST_IS_MINUS_U else east
    y_arcsec = north

    G = cfg.grid_size
    half = cfg.fov_arcmin * 60.0 / 2.0
    ix = np.floor((x_arcsec + half) / (2.0 * half) * G).astype(np.int64)
    iy = np.floor((y_arcsec + half) / (2.0 * half) * G).astype(np.int64)

    ok_x = (ix >= 0) & (ix < G)
    ok_y = (iy >= 0) & (iy < G)
    if not (ok_x.any() and ok_y.any()):
        return np.zeros((G, G), dtype=np.float32), np.zeros((G, G), dtype=np.int64)

    sub = np.asarray(img, dtype=np.float64)[np.ix_(ok_y, ok_x)]
    cell = (iy[ok_y][:, None] * G + ix[ok_x][None, :]).ravel()

    total = np.bincount(cell, weights=np.nan_to_num(sub).ravel(), minlength=G * G)
    n_native = np.bincount(cell, minlength=G * G)

    out = total.reshape(G, G)
    if how == "mean":
        out = np.divide(out, n_native.reshape(G, G),
                        out=np.zeros_like(out), where=n_native.reshape(G, G) > 0)
    return out.astype(np.float32), n_native.reshape(G, G)


def arcsec_per_pixel(cfg=CFG_XRAY):
    return cfg.fov_arcmin * 60.0 / cfg.grid_size


def mpc_per_pixel(redshift, h, cfg=CFG_XRAY, Om0=None):
    """
    Physical Mpc per stored pixel at the sample's redshift, in the simulation's
    cosmology (BAHAMAS unless Om0 is given).
    """
    cosmo = bahamas_cosmology(h, Om0)
    D_A = float(cosmo.angular_diameter_distance(0.0, redshift).to("Mpc").d)
    return arcsec_per_pixel(cfg) * (np.pi / 180.0 / 3600.0) * D_A


def build_sample_maps(products, redshift, h, cfg=CFG_XRAY, Om0=None):
    """
    Turn the FITS products into the small arrays the dataset stores.

      xray_counts : mosaic counts per stored pixel, 0.5-2 keV
      xray_expmap : mean exposure map per stored pixel. SOXS normalises it by
                    the exposure time, so the units are cm^2, not cm^2 s.
      xray_mosaic : exposure-corrected surface brightness,
                    counts s^-1 cm^-2 arcsec^-2  (panel c of the notebook figure)
      xray_ideal  : the same units with no instrument, PSF or background (panel a)

    Because the exposure map is normalised, counts/expmap is a fluence per unit
    area and still has to be divided by the per-pointing exposure to become a
    rate -- the notebook's `sb_mo = flux_mos / t_mos / pix_as**2`. Overlapping
    pointings need no special handling: they add counts and effective area in
    the same proportion, as long as every pointing shares an exposure time.
    """
    pix_arcsec = arcsec_per_pixel(cfg)
    pix_area = pix_arcsec ** 2
    t_exp_s = float(products["t_exp_s"])

    with fits.open(products["counts"]) as f:
        counts, _ = to_frame_grid(f[0].data, f[0].header, cfg, how="sum")
    with fits.open(products["expmap"]) as f:
        expmap, _ = to_frame_grid(f[0].data, f[0].header, cfg, how="mean")

    mosaic = np.divide(counts, expmap * t_exp_s * pix_area,
                       out=np.zeros_like(counts), where=expmap > 0)

    # The ideal map is already on the stored grid; only the axis convention and
    # the normalisation (sampling area x photon-list exposure) have to be applied.
    with fits.open(products["ideal"]) as f:
        ideal_raw = np.asarray(f[0].data, dtype=np.float64)
        hdr = f[0].header
        exposure = float(hdr["EXPOSURE"])
        east, north = _pixel_sky_offsets(hdr)
        if (east[1] - east[0]) < 0:       # +x runs West -> flip onto East
            ideal_raw = ideal_raw[:, ::-1]
        if EAST_IS_MINUS_U:
            ideal_raw = ideal_raw[:, ::-1]
        if (north[1] - north[0]) < 0:
            ideal_raw = ideal_raw[::-1, :]
    ideal = ideal_raw / (cfg.sample_area_cm2 * exposure * pix_area)

    return {
        "xray_counts": counts.astype(np.float32),
        "xray_expmap": expmap.astype(np.float32),
        "xray_mosaic": mosaic.astype(np.float32),
        "xray_ideal": ideal.astype(np.float32),
        "xray_arcsec_per_pix": np.float32(pix_arcsec),
        "xray_mpc_per_pix": np.float32(mpc_per_pixel(redshift, h, cfg, Om0)),
        # float64 to match the group's z_lens attr exactly: the photon list was
        # generated at that redshift, and the two should compare equal.
        "xray_redshift": np.float64(redshift),
        # The realised depth, read back from the event header rather than taken
        # from the draw, so it is whatever SOXS actually integrated.
        "xray_t_exp_ks": np.float32(t_exp_s / 1.0e3),
    }


def clean_intermediates(work_prefix, photon_prefix=None):
    """Delete the ~1.2 GB of pyXSIM/SOXS products for one sample."""
    for path in glob.glob(f"{work_prefix}*"):
        try:
            os.remove(path)
        except OSError:
            pass
    if photon_prefix is not None:
        for path in glob.glob(f"{photon_prefix}*"):
            try:
                os.remove(path)
            except OSError:
                pass
