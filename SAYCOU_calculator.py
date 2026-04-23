#!/usr/bin/env python3

import sys
import math
import numpy as np
import re
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit


@njit(cache=True)
def scattered_energy(E, mu, A):
    """
    returns energy of elastically scattered neutron under angle theta = arccos (mu)
      mu : float
        mu = cos (theta)
      E : float
        incident neutron energy [eV]
      A : integer
        mass number
    """
    return E * (1.0 / (1.0 + A))**2 * (mu + ((A*A - 1.0 + mu*mu)**0.5))**2


@njit(cache=True)
def get_nearest_xs(E_query, energy_grid, sigma_tot, sigma_el, sigma_cap):
    """
    Return cross sections from the nearest energy-grid point.

    Assumes energy_grid is sorted in ascending order.
    """
    idx = np.searchsorted(energy_grid, E_query)

    if idx == 0:
        nearest = 0
    elif idx >= len(energy_grid):
        nearest = len(energy_grid) - 1
    else:
        if abs(E_query - energy_grid[idx - 1]) <= abs(energy_grid[idx] - E_query):
            nearest = idx - 1
        else:
            nearest = idx

    return sigma_tot[nearest], sigma_el[nearest], sigma_cap[nearest]


@njit(cache=True)
def exit_length_cylinder(r0_mm, z0_mm, mu, phi, sample_radius_mm, thickness_mm):
    """
    Distance from a point inside a cylinder to the first boundary
    along direction (mu, phi).
    """
    sin_theta = math.sqrt(max(0.0, 1.0 - mu * mu))

    ux = sin_theta * math.cos(phi)
    uy = sin_theta * math.sin(phi)
    uz = mu

    candidates = np.empty(4, dtype=np.float64)
    ncand = 0

    if uz > 0.0:
        s_top = (thickness_mm - z0_mm) / uz
        if s_top > 0.0:
            candidates[ncand] = s_top
            ncand += 1
    elif uz < 0.0:
        s_bottom = -z0_mm / uz
        if s_bottom > 0.0:
            candidates[ncand] = s_bottom
            ncand += 1

    a = ux * ux + uy * uy
    if a > 0.0:
        b = 2.0 * r0_mm * ux
        c = r0_mm * r0_mm - sample_radius_mm * sample_radius_mm
        disc = b * b - 4.0 * a * c

        if disc >= 0.0:
            sqrt_disc = math.sqrt(disc)
            s1 = (-b - sqrt_disc) / (2.0 * a)
            s2 = (-b + sqrt_disc) / (2.0 * a)

            if s1 > 0.0:
                candidates[ncand] = s1
                ncand += 1
            if s2 > 0.0:
                candidates[ncand] = s2
                ncand += 1

    if ncand == 0:
        return 0.0

    smin = candidates[0]
    for i in range(1, ncand):
        if candidates[i] < smin:
            smin = candidates[i]
    return smin


@njit(cache=True)
def propagate_point(r0_mm, z0_mm, mu, phi, s_mm):
    """
    Starting from (x0,y0,z0)=(r0_mm,0,z0_mm), move by distance s_mm
    along direction (mu,phi). Return Cartesian coordinates.
    """
    sin_theta = math.sqrt(max(0.0, 1.0 - mu * mu))

    x0 = r0_mm
    y0 = 0.0
    z1 = z0_mm

    ux = sin_theta * math.cos(phi)
    uy = sin_theta * math.sin(phi)
    uz = mu

    x = x0 + s_mm * ux
    y = y0 + s_mm * uy
    z = z1 + s_mm * uz

    return x, y, z


@njit(cache=True)
def exit_length_cylinder_cartesian(x0_mm, y0_mm, z0_mm, mu, phi, sample_radius_mm, thickness_mm):
    """
    Distance from arbitrary point (x0,y0,z0) inside cylinder to first boundary
    along direction (mu,phi).
    """
    sin_theta = math.sqrt(max(0.0, 1.0 - mu * mu))

    ux = sin_theta * math.cos(phi)
    uy = sin_theta * math.sin(phi)
    uz = mu

    candidates = np.empty(4, dtype=np.float64)
    ncand = 0

    if uz > 0.0:
        s_top = (thickness_mm - z0_mm) / uz
        if s_top > 0.0:
            candidates[ncand] = s_top
            ncand += 1
    elif uz < 0.0:
        s_bottom = -z0_mm / uz
        if s_bottom > 0.0:
            candidates[ncand] = s_bottom
            ncand += 1

    a = ux * ux + uy * uy
    if a > 0.0:
        b = 2.0 * (x0_mm * ux + y0_mm * uy)
        c = x0_mm * x0_mm + y0_mm * y0_mm - sample_radius_mm * sample_radius_mm
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sqrt_disc = math.sqrt(disc)
            s1 = (-b - sqrt_disc) / (2.0 * a)
            s2 = (-b + sqrt_disc) / (2.0 * a)
            if s1 > 0.0:
                candidates[ncand] = s1
                ncand += 1
            if s2 > 0.0:
                candidates[ncand] = s2
                ncand += 1

    if ncand == 0:
        return 0.0

    smin = candidates[0]
    for i in range(1, ncand):
        if candidates[i] < smin:
            smin = candidates[i]
    return smin


@njit(cache=True)
def calculate_y1_single_energy(
    E_in,
    A,
    energy_grid,
    sigma_tot_grid,
    sigma_el_grid,
    sigma_cap_grid,
    n_areal,
    thickness_mm,
    diameter_mm,
    beam_diameter_mm,
    nr=12,
    nz=16,
    nmu=24,
    nphi=24
):
    """
    Deterministic estimate of Y1(E): first interaction = elastic scatter,
    then capture before escape.
    """
    sample_radius_mm = diameter_mm / 2.0
    beam_radius_mm = beam_diameter_mm / 2.0
    illum_radius_mm = min(sample_radius_mm, beam_radius_mm)

    sigma_tot_in, sigma_el_in, _ = get_nearest_xs(
        E_in, energy_grid, sigma_tot_grid, sigma_el_grid, sigma_cap_grid
    )

    Sigma_tot_in = n_areal * sigma_tot_in / thickness_mm
    Sigma_el_in  = n_areal * sigma_el_in  / thickness_mm

    dr = illum_radius_mm / nr
    dz = thickness_mm / nz
    dmu = 2.0 / nmu
    dphi = 2.0 * math.pi / nphi

    total = 0.0

    for ir in range(nr):
        r0_mm = (ir + 0.5) * dr
        w_r = 2.0 * r0_mm / (illum_radius_mm * illum_radius_mm) * dr

        for iz in range(nz):
            z0_mm = (iz + 0.5) * dz
            p_first_scatter_density = Sigma_el_in * math.exp(-Sigma_tot_in * z0_mm)

            for imu in range(nmu):
                mu = -1.0 + (imu + 0.5) * dmu
                w_mu = 0.5 * dmu

                E_scatt = scattered_energy(E_in, mu, A)
                sigma_tot_scatt, _, sigma_cap_scatt = get_nearest_xs(
                    E_scatt, energy_grid, sigma_tot_grid, sigma_el_grid, sigma_cap_grid
                )
                Sigma_tot_scatt = n_areal * sigma_tot_scatt / thickness_mm
                Sigma_cap_scatt = n_areal * sigma_cap_scatt / thickness_mm

                prefactor = p_first_scatter_density * w_r * dz * w_mu

                if Sigma_tot_scatt > 0.0:
                    branch_cap_scatt = Sigma_cap_scatt / Sigma_tot_scatt
                else:
                    branch_cap_scatt = 0.0

                for iphi in range(nphi):
                    phi = (iphi + 0.5) * dphi
                    w_phi = dphi / (2.0 * math.pi)

                    L_mm = exit_length_cylinder(
                        r0_mm, z0_mm, mu, phi, sample_radius_mm, thickness_mm
                    )
                    p_cap_after = branch_cap_scatt * (1.0 - math.exp(-Sigma_tot_scatt * L_mm))
                    total += prefactor * w_phi * p_cap_after

    return total


@njit(cache=True)
def calculate_y2_and_y3plus_single_energy(
    E_in,
    A,
    energy_grid,
    sigma_tot_grid,
    sigma_el_grid,
    sigma_cap_grid,
    n_areal,
    thickness_mm,
    diameter_mm,
    beam_diameter_mm,
    nr=4,
    nz=4,
    nmu1=6,
    nphi1=6,
    ns1=6,
    nmu2=6,
    nphi2=6,
    nmu3=6
):
    """
    Deterministic estimate of strict Y2 and approximate Y3plus,
    with full mu3 sampling for the Y3+ tail.
    """
    sample_radius_mm = diameter_mm / 2.0
    beam_radius_mm = beam_diameter_mm / 2.0
    illum_radius_mm = min(sample_radius_mm, beam_radius_mm)

    sigma_tot_0, sigma_el_0, _ = get_nearest_xs(
        E_in, energy_grid, sigma_tot_grid, sigma_el_grid, sigma_cap_grid
    )

    Sigma_tot_0 = n_areal * sigma_tot_0 / thickness_mm
    Sigma_el_0  = n_areal * sigma_el_0  / thickness_mm

    dr = illum_radius_mm / nr
    dz = thickness_mm / nz
    dmu1 = 2.0 / nmu1
    dphi1 = 2.0 * math.pi / nphi1
    dmu2 = 2.0 / nmu2
    dphi2 = 2.0 * math.pi / nphi2
    dmu3 = 2.0 / nmu3

    total_y2 = 0.0
    total_y3plus = 0.0

    for ir in range(nr):
        r0_mm = (ir + 0.5) * dr
        w_r = 2.0 * r0_mm / (illum_radius_mm * illum_radius_mm) * dr

        for iz in range(nz):
            z0_mm = (iz + 0.5) * dz
            p1 = Sigma_el_0 * math.exp(-Sigma_tot_0 * z0_mm)

            for imu1 in range(nmu1):
                mu1 = -1.0 + (imu1 + 0.5) * dmu1
                w_mu1 = 0.5 * dmu1

                E1 = scattered_energy(E_in, mu1, A)
                sigma_tot_1, sigma_el_1, _ = get_nearest_xs(
                    E1, energy_grid, sigma_tot_grid, sigma_el_grid, sigma_cap_grid
                )

                Sigma_tot_1 = n_areal * sigma_tot_1 / thickness_mm
                Sigma_el_1  = n_areal * sigma_el_1  / thickness_mm

                for iphi1 in range(nphi1):
                    phi1 = (iphi1 + 0.5) * dphi1
                    w_phi1 = dphi1 / (2.0 * math.pi)

                    L1_mm = exit_length_cylinder(
                        r0_mm, z0_mm, mu1, phi1, sample_radius_mm, thickness_mm
                    )

                    if L1_mm <= 0.0:
                        continue

                    ds1 = L1_mm / ns1

                    for is1 in range(ns1):
                        s1_mm = (is1 + 0.5) * ds1
                        p2 = Sigma_el_1 * math.exp(-Sigma_tot_1 * s1_mm)

                        x1_mm, y1_mm, z1_mm = propagate_point(
                            r0_mm, z0_mm, mu1, phi1, s1_mm
                        )

                        if (
                            z1_mm < 0.0 or z1_mm > thickness_mm or
                            (x1_mm * x1_mm + y1_mm * y1_mm) > sample_radius_mm * sample_radius_mm
                        ):
                            continue

                        prefactor = p1 * p2 * w_r * dz * w_mu1 * w_phi1 * ds1

                        for imu2 in range(nmu2):
                            mu2 = -1.0 + (imu2 + 0.5) * dmu2
                            w_mu2 = 0.5 * dmu2

                            E2 = scattered_energy(E1, mu2, A)
                            sigma_tot_2, sigma_el_2, sigma_cap_2 = get_nearest_xs(
                                E2, energy_grid, sigma_tot_grid, sigma_el_grid, sigma_cap_grid
                            )

                            Sigma_tot_2 = n_areal * sigma_tot_2 / thickness_mm
                            Sigma_el_2  = n_areal * sigma_el_2  / thickness_mm
                            Sigma_cap_2 = n_areal * sigma_cap_2 / thickness_mm

                            sigma_cap_3_avg = 0.0
                            for imu3 in range(nmu3):
                                mu3 = -1.0 + (imu3 + 0.5) * dmu3
                                w_mu3 = 0.5 * dmu3
                                E3 = scattered_energy(E2, mu3, A)
                                _, _, sigma_cap_3 = get_nearest_xs(
                                    E3, energy_grid, sigma_tot_grid, sigma_el_grid, sigma_cap_grid
                                )
                                sigma_cap_3_avg += w_mu3 * sigma_cap_3
                            Sigma_cap_3_avg = n_areal * sigma_cap_3_avg / thickness_mm

                            for iphi2 in range(nphi2):
                                phi2 = (iphi2 + 0.5) * dphi2
                                w_phi2 = dphi2 / (2.0 * math.pi)

                                L2_mm = exit_length_cylinder_cartesian(
                                    x1_mm, y1_mm, z1_mm,
                                    mu2, phi2,
                                    sample_radius_mm, thickness_mm
                                )

                                if Sigma_tot_2 > 0.0:
                                    survive2 = 1.0 - math.exp(-Sigma_tot_2 * L2_mm)
                                    p_y2 = (Sigma_cap_2 / Sigma_tot_2) * survive2
                                    p_scat_next = (Sigma_el_2 / Sigma_tot_2) * survive2
                                else:
                                    p_y2 = 0.0
                                    p_scat_next = 0.0

                                p_y3plus = p_scat_next * (
                                    1.0 - math.exp(-Sigma_cap_3_avg * L2_mm)
                                )

                                weight = prefactor * w_mu2 * w_phi2
                                total_y2 += weight * p_y2
                                total_y3plus += weight * p_y3plus

    return total_y2, total_y3plus



def _compute_chunk(args):
    (
        start_idx, end_idx, energy_sel, A, energy, sigma_tot, sigma_el, sigma_cap,
        n_areal, thickness_mm, diameter_mm, beam_diameter_mm, y1_grid, y2_grid
    ) = args

    nloc = end_idx - start_idx
    y1_chunk = np.empty(nloc, dtype=np.float64)
    y2_chunk = np.empty(nloc, dtype=np.float64)
    y3plus_chunk = np.empty(nloc, dtype=np.float64)

    e0 = energy_sel[start_idx]
    _ = calculate_y1_single_energy(
        e0, A, energy, sigma_tot, sigma_el, sigma_cap,
        n_areal, thickness_mm, diameter_mm, beam_diameter_mm,
        y1_grid[0], y1_grid[1], y1_grid[2], y1_grid[3]
    )
    _ = calculate_y2_and_y3plus_single_energy(
        e0, A, energy, sigma_tot, sigma_el, sigma_cap,
        n_areal, thickness_mm, diameter_mm, beam_diameter_mm,
        y2_grid[0], y2_grid[1], y2_grid[2], y2_grid[3], y2_grid[4], y2_grid[5], y2_grid[6]
    )

    for j in range(nloc):
        E_in = energy_sel[start_idx + j]
        y1_chunk[j] = calculate_y1_single_energy(
            E_in, A, energy, sigma_tot, sigma_el, sigma_cap,
            n_areal, thickness_mm, diameter_mm, beam_diameter_mm,
            y1_grid[0], y1_grid[1], y1_grid[2], y1_grid[3]
        )
        y2_chunk[j], y3plus_chunk[j] = calculate_y2_and_y3plus_single_energy(
            E_in, A, energy, sigma_tot, sigma_el, sigma_cap,
            n_areal, thickness_mm, diameter_mm, beam_diameter_mm,
            y2_grid[0], y2_grid[1], y2_grid[2], y2_grid[3], y2_grid[4], y2_grid[5], y2_grid[6]
        )

    return start_idx, end_idx, y1_chunk, y2_chunk, y3plus_chunk


def _build_chunks(npts, chunk_size):
    chunks = []
    start = 0
    while start < npts:
        end = min(start + chunk_size, npts)
        chunks.append((start, end))
        start = end
    return chunks

def read_xs_file(filename):
    """
    Read a 4-column cross section file.
    """
    try:
        data = np.loadtxt(filename, comments="#")
    except Exception as exc:
        raise RuntimeError(f"Could not read input file '{filename}': {exc}")

    if data.ndim != 2 or data.shape[1] < 4:
        raise RuntimeError(
            f"Input file '{filename}' must contain at least 4 columns: "
            "E[eV], sigma_tot[b], sigma_el[b], sigma_cap[b]"
        )

    energy = data[:, 0]
    sigma_tot = data[:, 1]
    sigma_el = data[:, 2]
    sigma_cap = data[:, 3]

    return energy, sigma_tot, sigma_el, sigma_cap


def calculate_capture_yield(n_areal, sigma_tot, sigma_cap):
    """
    Calculate direct capture yield Y0.
    """
    sigma_tot = np.asarray(sigma_tot, dtype=float)
    sigma_cap = np.asarray(sigma_cap, dtype=float)

    yield_cap = np.zeros_like(sigma_tot)
    positive = sigma_tot > 0.0
    yield_cap[positive] = (
        (1.0 - np.exp(-n_areal * sigma_tot[positive]))
        * sigma_cap[positive]
        / sigma_tot[positive]
    )
    return yield_cap


def main():
    if len(sys.argv) != 7:
        print(f"Usage: {sys.argv[0]} XS_FILE AREAL_DENSITY_AT_PER_BARN THICKNESS_MM SAMPLE_DIAMETER_MM BEAM_DIAMETER_MM OUTPUT_FILE")
        print(f"Example: {sys.argv[0]} Element123_xs.txt 6.e-2 3.0 77.0 20.0 yields_output.txt")
        sys.exit(1)

    filename = sys.argv[1]
    m = re.search(r'[A-Z][a-z]?(\d+)', filename)
    A = int(m.group(1))

    try:
        n_areal = float(sys.argv[2])
    except ValueError:
        print("Error: areal density must be a floating-point number in at/b.")
        sys.exit(1)

    try:
        thickness_mm = float(sys.argv[3])
    except ValueError:
        print("Error: thickness must be a floating-point number in mm.")
        sys.exit(1)

    try:
        diameter_mm = float(sys.argv[4])
    except ValueError:
        print("Error: sample diameter must be a floating-point number in mm.")
        sys.exit(1)

    try:
        beam_diameter_mm = float(sys.argv[5])
    except ValueError:
        print("Error: beam diameter must be a floating-point number in mm.")
        sys.exit(1)

    outname = sys.argv[6]

    energy, sigma_tot, sigma_el, sigma_cap = read_xs_file(filename)
    if not np.all(np.diff(energy) >= 0):
        raise ValueError("energy grid must be sorted in ascending order")

    yield_cap = calculate_capture_yield(n_areal, sigma_tot, sigma_cap)

    print(f"Input file          : {filename}")
    print(f"Output file         : {outname}")
    print(f"Sample A            : {A}")
    print(f"Areal density       : {n_areal:.6e} at/b")
    print(f"Thickness           : {thickness_mm:.6f} mm")
    print(f"Sample diameter     : {diameter_mm:.6f} mm")
    print(f"Beam diameter       : {beam_diameter_mm:.6f} mm")
    print(f"Number of points    : {len(energy)}")
    print()

    mask = (energy >= 100.0) & (energy <= 6000.0)
    energy_sel = energy[mask]

    y1 = np.zeros_like(energy_sel)
    y2 = np.zeros_like(energy_sel)
    y3plus = np.zeros_like(energy_sel)

    npts = len(energy_sel)

    y1_grid = (6, 6, 8, 8)
    y2_grid = (4, 4, 6, 6, 6, 6, 6, 6)

    # warm-up JIT compilation in the main process before timing
    _ = calculate_y1_single_energy(
        energy_sel[0], A, energy, sigma_tot, sigma_el, sigma_cap,
        n_areal, thickness_mm, diameter_mm, beam_diameter_mm,
        y1_grid[0], y1_grid[1], y1_grid[2], y1_grid[3]
    )
    _ = calculate_y2_and_y3plus_single_energy(
        energy_sel[0], A, energy, sigma_tot, sigma_el, sigma_cap,
        n_areal, thickness_mm, diameter_mm, beam_diameter_mm,
        y2_grid[0], y2_grid[1], y2_grid[2], y2_grid[3], y2_grid[4], y2_grid[5], y2_grid[6], y2_grid[7]
    )

    t0 = time.perf_counter()

    nworkers = max(1, os.cpu_count() - 1)
    chunk_size = max(1, npts // (4 * nworkers))
    chunks = _build_chunks(npts, chunk_size)
    tasks = [
        (
            start_idx, end_idx, energy_sel, A, energy, sigma_tot, sigma_el, sigma_cap,
            n_areal, thickness_mm, diameter_mm, beam_diameter_mm, y1_grid, y2_grid
        )
        for start_idx, end_idx in chunks
    ]

    done = 0
    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        futures = [executor.submit(_compute_chunk, task) for task in tasks]

        for future in as_completed(futures):
            start_idx, end_idx, y1_chunk, y2_chunk, y3plus_chunk = future.result()
            y1[start_idx:end_idx] = y1_chunk
            y2[start_idx:end_idx] = y2_chunk
            y3plus[start_idx:end_idx] = y3plus_chunk

            done += (end_idx - start_idx)
            elapsed = time.perf_counter() - t0
            avg = elapsed / done
            remaining = avg * (npts - done)

            print(
                f"\r{done}/{npts}  "
                f"elapsed = {elapsed:.1f} s  "
                f"ETA = {remaining:.1f} s",
                end="",
                flush=True
            )
    print()
    t1 = time.perf_counter()
    print(f"Total calculation time: {t1 - t0:.3f} s")
    print(f"Average per energy: {(t1 - t0)/len(energy_sel):.3f} s")

    export_data = np.column_stack((
        energy_sel,
        yield_cap[mask],
        y1,
        y2,
        y3plus
    ))

    header_lines = [
        f"xs_file = {filename}",
        f"A = {A}",
        f"n_areal_at_per_b = {n_areal:.8e}",
        f"thickness_mm = {thickness_mm:.8e}",
        f"sample_diameter_mm = {diameter_mm:.8e}",
        f"beam_diameter_mm = {beam_diameter_mm:.8e}",
        f"energy_min_eV = {energy_sel.min():.8e}",
        f"energy_max_eV = {energy_sel.max():.8e}",
        f"nworkers = {nworkers} (Numba chunked parallel)",
        f"chunk_size = {chunk_size}",
        f"y1_grid = {y1_grid}",
        f"y2_grid = {y2_grid}",
        f"total_calculation_time_s = {t1 - t0:.6f}",
        "columns: energy_eV Y0 Y1 Y2 Y3plus"
    ]
    header = "\n".join(header_lines)

    np.savetxt(outname, export_data, header=header, fmt="%.6e")

    print(f"Saved yields to: {outname}")


if __name__ == "__main__":
    main()
