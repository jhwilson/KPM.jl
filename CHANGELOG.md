# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] — unreleased

### Changed

- `kpm_1d_current` now computes the complex `Tr[J T_n]/D`; it previously used
  the real part of `⟨psi|J† T_n|psi⟩`, which vanishes for bond currents.
- `optical_cond1` and `optical_cond2` now use the correct Chebyshev weight and
  factor-of-two convention, the Kubo–Bastin index orientation for unequal
  operators, analytically continued Green coefficients, and adaptive
  quadrature with an explicit error contract. The `Omega` keyword was removed
  and `lambda` was added; `d_optical_cond1` and `d_optical_cond2` are host-side
  and no longer accept `delta`.
- `cpge` now returns the bare three-current `chi_alpha_beta_gamma` without an
  `Omega` or `1/(omega1*omega2)` prefactor. Calls with `omega2 = -omega1` are
  supported through `delta` or `lambda` regularization, with corrected index
  orientation and Chebyshev factor-of-two convention.
- kpm_3d/kpm_3d! now take the three operators in trace order:
  μ[n,m,p] = Tr[Jα T_m Jβ T_n Jγ T_p]/D (previously Jβ and Jγ were exchanged).
- `d_dc_cond` and `dos` support arbitrary derivative order through
  ForwardDiff; Zygote is no longer a dependency.
- The finite-`beta` `chern_marker` documentation now distinguishes the
  smoothed Fermi-operator marker from finite-temperature Hall conductivity.
- `d_optical_cond1` and `d_optical_cond2` now accept the finite resolved integrand at `omega = 0`.
- `cpge` and `d_cpge` now reject unregularized coincident Green edges at individual, sum, or opposite-edge frequency shifts.

### Added

- `CurrentMoments`, `current_moments`, and the typed `optical_cond` API, which
  accepts physical energies and returns two-dimensional conductivity in
  `e^2/h`.
