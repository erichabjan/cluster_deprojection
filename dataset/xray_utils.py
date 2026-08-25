"""
Mock Chandra ACIS-I mosaics of a BAHAMAS cluster, ported from
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

from config import CFG_XRAY, CFG_COSMO, CFG_DATA, CFG_LENS

# --- physical constants (CGS), as in the notebook ---
MP = 1.67262192369e-24        # proton mass, g
MSUN = 1.98892e33             # solar mass, g (yt's value)
KPC = 3.0856775809623245e21   # cm
MPC = 1.0e3 * KPC

BOXSIZE_MPC_OVER_H = 400.0    # BAHAMAS box (cMpc/h)

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


def bcg_index(data, boxsize=BOXSIZE_MPC_OVER_H) -> int:
    """
    Index of the BCG: the most massive stellar subhalo within
    CFG_DATA.bcg_search_radius_r200 * R200 of the FoF centre of potential.

    Shared by the photon stage and the dataset build so both centre the cluster
    identically. The aperture matters: a global argmax picks a neighbouring
    cluster's BCG, several Mpc away, in a handful of the 5 R200 cutouts.
    """
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


def z_bin_centers():
    """Redshift bin centres the photon lists are generated on."""
    edges = np.arange(CFG_LENS.z_lens_min, CFG_LENS.z_lens_max + 1e-9,
                      CFG_XRAY.z_bin_width)
    return [round(float(e + 0.5 * CFG_XRAY.z_bin_width), 6) for e in edges[:-1]]


def nearest_z_bin(z_lens):
    """The photon-list bin a sample at z_lens should be observed from."""
    centers = z_bin_centers()
    return centers[int(np.argmin(np.abs(np.asarray(centers) - z_lens)))]


def photon_prefix(sim, cluster_idx, z_bin):
    """Where the photon stage writes, and the dataset build reads, a photon list."""
    return os.path.join(CFG_XRAY.photon_dir,
                        f"{sim}_GrNm_{cluster_idx:03d}_z{z_bin:.4f}_photons")


def density_unit_cgs(h, a):
    """gas_dens is Gadget code density, 1e10 Msun/h per (cMpc/h)^3 -> g/cm^3."""
    return (1.0e10 * MSUN / h) / (MPC / h) ** 3 / a ** 3


def bahamas_cosmology(h):
    return Cosmology(hubble_constant=h,
                     omega_matter=CFG_COSMO.Om0_sim,
                     omega_lambda=1.0 - CFG_COSMO.Om0_sim)


def load_gas_dataset(data, center, cfg=CFG_XRAY):
    """
    Gas within cfg.n_R200 * R200 of `center` (the BCG), as an in-memory yt SPH
    dataset with the fields pyXSIM needs. Returns (ds, sphere_radius_pMpc).

    `center` is in comoving Mpc/h, like data['sub_pos'].
    """
    BOX = BOXSIZE_MPC_OVER_H

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


def make_photon_list(prefix, ds, radius_mpc, redshift, cosmo, cfg=CFG_XRAY):
    """The expensive step (~2.5 min): projection-independent, redshift-dependent."""
    c = ds.arr([0.0, 0.0, 0.0], "kpc")
    src = ds.sphere(c, (radius_mpc, "Mpc"))
    n_ph, n_cells = pyxsim.make_photons(
        prefix, src, redshift,
        (cfg.sample_area_cm2, "cm**2"), (cfg.sample_exp_ks, "ks"),
        source_model(cfg), cosmology=cosmo, center=c,
    )
    return int(n_ph), int(n_cells)


def project_and_observe(photon_prefix, work_prefix, proj_vec, cfg=CFG_XRAY):
    """
    Project the photon list along the sample's viewing direction and run the
    3x3 ACIS-I mosaic. Returns the paths of the products, all under work_prefix.
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

    # (c) the mosaic: SOXS observes the SIMPUT catalogue from n_side^2 pointings.
    events.write_to_simput(work_prefix, overwrite=True)
    ra0, dec0 = cfg.sky_center
    pointings = [(ra0 + ddx / np.cos(np.deg2rad(dec0)), dec0 + ddy)
                 for ddx, ddy in cfg.pointing_offsets_deg()]

    mosaic_table = soxs.make_mosaic_events(
        pointings, f"{work_prefix}_simput.fits", f"{work_prefix}_mosaic",
        (cfg.t_exp_ks, "ks"), cfg.instrument, overwrite=True,
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


def mpc_per_pixel(redshift, h, cfg=CFG_XRAY):
    """Physical Mpc per stored pixel at the sample's redshift (BAHAMAS cosmology)."""
    cosmo = bahamas_cosmology(h)
    D_A = float(cosmo.angular_diameter_distance(0.0, redshift).to("Mpc").d)
    return arcsec_per_pixel(cfg) * (np.pi / 180.0 / 3600.0) * D_A


def build_sample_maps(products, redshift, h, cfg=CFG_XRAY):
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
        "xray_mpc_per_pix": np.float32(mpc_per_pixel(redshift, h, cfg)),
        "xray_redshift": np.float32(redshift),
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
