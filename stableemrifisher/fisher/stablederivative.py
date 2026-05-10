import numpy as np
from few.waveform import GenerateEMRIWaveform
from few.utils.constants import Gpc, MRSUN_SI
from tqdm import tqdm
from few.utils.mappings.common import chi_to_chit  # 1PAT1R modification

from ..deriv_utils.deriv_angles import viewing_angle_partials, fplus_fcross_derivs

import time


class StableEMRIDerivative(GenerateEMRIWaveform):
    """
    inherits from the GenerateEMRIWaveform class of FEW, adds functions for derivative calculation and a fresh __call__ method.
    """

    def __init__(
        self,
        waveform_class,
        *args,
        frame: str = "detector",
        return_list=False,
        flip_output=False,
        **kwargs,
    ):
        super().__init__(
            waveform_class=waveform_class,
            *args,
            frame=frame,
            return_list=return_list,
            flip_output=flip_output,
            **kwargs,
        )  # initialize GenerateEMRIWaveform
        self.cache = None  # initialize waveform cache
        # 1PAT1R modification: cached once — descriptor is fixed at construction time.
        self._is_1PA = getattr(self.waveform_generator, "descriptor", "") == "circular"

    def __getattr__(self, name):
        # get_attributes from self.waveform_generator if not found in GenerateEMRIWaveform
        return getattr(self.waveform_generator, name)

    def __call__(
        self,
        *args,  # dummy to fool FastLISAResponse
        parameters: dict,
        param_to_vary: str,
        delta: float,
        order: int,
        kind: str,
        **kwargs,
    ):
        """
        generate the waveform derivative for the given parameter
        """

        self.delta = delta
        self.order = order
        self.kind = kind

        if self.kind not in ["central", "forward", "backward"]:
            raise ValueError('kind must be one of "central", "forward", or "backward".')
        if self.order not in [2, 4, 6, 8]:
            raise ValueError("order must be one of 2, 4, 6, or 8.")

        try:
            T = kwargs["T"]
        except KeyError:
            T = 1.0

        try:
            dt = kwargs["dt"]
        except KeyError:
            dt = 10.0

        try:
            batch_size = kwargs["batch_size"]
        except KeyError:
            batch_size = -1

        try:
            show_progress = kwargs["show_progress"]
        except KeyError:
            show_progress = False

        # construct deltas for derivative
        self.deltas = self._deltas(self.delta, self.order, self.kind)

        # how we proceed depends on the parameter we are differentiating with respect to
        # 1PAT1R modification: chi2 is carried in kwargs (current_waveform_kwargs),
        # not in the positional parameters dict, so exempt it from this check.
        if param_to_vary not in parameters and param_to_vary != "chi2":  # 1PAT1R modification
            raise ValueError(
                f"Parameter '{param_to_vary}' not in parameters dictionary."
            )

        # obtain source frame angles
        theta_source, phi_source = self._get_viewing_angles(
            parameters["qS"], parameters["phiS"], parameters["qK"], parameters["phiK"]
        )

        keys_exclude = ["T", "dt", "batch_size", "show_progress"]
        kwargs_remaining = {
            key: value for key, value in kwargs.items() if key not in keys_exclude
        }
        # 1PAT1R modification: chi2 and evolve_primary are trajectory/amplitude
        # kwargs that must not be forwarded to create_waveform. Strip them here
        # into a separate dict used only where needed.
        _waveform_keys_exclude = {"chi2", "evolve_primary"}
        kwargs_waveform = {k: v for k, v in kwargs_remaining.items() if k not in _waveform_keys_exclude}

        # get waveform
        # 1PAT1R modification: include chi2 and evolve_primary from kwargs in the
        # cache key so that changes between calls correctly invalidate the cache.
        _cache_chi2 = kwargs_remaining.get("chi2", None)  # 1PAT1R modification
        _cache_evolve = kwargs_remaining.get("evolve_primary", None)  # 1PAT1R modification
        _cache_key_matches = (  # 1PAT1R modification
            self.cache is not None  # 1PAT1R modification
            and parameters == self.cache["parameters"]  # 1PAT1R modification
            and _cache_chi2 == self.cache.get("chi2", None)  # 1PAT1R modification
            and _cache_evolve == self.cache.get("evolve_primary", None)  # 1PAT1R modification
        )  # 1PAT1R modification
        if not _cache_key_matches:  # 1PAT1R modification

            t, y = self._trajectory_from_parameters(parameters, T, **kwargs_remaining)

            self.cache = {
                "t": t,
                "y": y,
                "parameters": parameters,
                "chi2": _cache_chi2,           # 1PAT1R modification: part of cache key
                "evolve_primary": _cache_evolve,  # 1PAT1R modification: part of cache key
                "coefficients": self.inspiral_generator.integrator_spline_coeff,
                # 1PAT1R modification: FEW-dev exposes a single spline coefficient
                # array (integrator_spline_coeff, shape (N-1, nparams, 8)) rather
                # than a separate integrator_spline_phase_coeff attribute.  Phase
                # components sit at parameter indices 3 (Phi_phi) and 5 (Phi_r)
                # for both the eccentric (nparams=6) and circular 1PAT1R
                # (nparams=8) models.  We extract those two columns here so the
                # downstream dPhi_dx projection (which expects a 2-column array
                # at indices [0, 2] = Phi_phi, Phi_r) remains unchanged.
                "phase_coefficients": self.xp.asarray(
                    self.inspiral_generator.integrator_spline_coeff
                )[:, [3, 5], :],  # 1PAT1R modification: was integrator_spline_phase_coeff[:, [0,2], :]
                # 1PAT1R modification: FEW-dev names this attribute integrator_t_cache,
                # not integrator_spline_t.
                "phase_coefficients_t": self.inspiral_generator.integrator_t_cache,  # 1PAT1R modification
            }

            amps_here = self._amplitudes_from_trajectory(
                parameters,
                t=t,
                y=y,
                theta_source=float(theta_source),
                phi_source=phi_source,
                cache=True,
                **kwargs_remaining,
            )

            # create waveform at injection
            waveform_source = self._create_waveform_in_batches(
                self.cache["t"],
                amps_here,  # actually teuk_amps * Ylms_in
                self.cache["dummy_ylms"],  # a bunch of ones
                self.cache["phase_coefficients_t"],
                self.cache["phase_coefficients"],
                self.cache["ls_all"],
                self.cache["ms_all"],
                # self.cache['ks_all'], #no inclination in FEW 2.0 :(
                self.cache["ns_all"],
                dt=dt,
                T=T,
                batch_size=batch_size,
                show_progress=show_progress,
                **kwargs_waveform,
            )

            self.cache["waveform_source"] = waveform_source

        self.cache["Npad"] = 0

        # now calculate the derivatives with respect to the chosen parameters

        # distance
        if param_to_vary == "dist":
            waveform_derivative_source = (
                -self.cache["waveform_source"] / parameters["dist"]
            )

        # phases
        elif param_to_vary in ["Phi_phi0", "Phi_theta0", "Phi_r0"]:
            # factor of -1j*m is applied to each amplitude
            modified_amps = self._modify_amplitudes_for_initial_phase_derivative(
                param_to_vary
            )

            # create waveform derivative
            waveform_derivative_source = self._create_waveform_in_batches(
                self.cache["t"],
                modified_amps,
                self.cache["dummy_ylms"],
                self.cache["phase_coefficients_t"],
                self.cache["phase_coefficients"],
                self.cache["ls_all"],
                self.cache["ms_all"],
                # self.cache['ks_all'],
                self.cache["ns_all"],
                dt=dt,
                T=T,
                batch_size=batch_size,
                show_progress=show_progress,
                **kwargs_waveform,
            )

        # sky angles (SSB)
        elif param_to_vary in ["qS", "phiS", "qK", "phiK"]:
            # finite differencing of the ylms w.r.t. theta, then chain rule partial h / partial theta * partial theta / partial angle
            modified_amps = self._modify_amplitudes_for_angle_derivative(
                parameters,
                param_to_vary,
                theta_source=theta_source,
                phi_source=phi_source,
            )

            waveform_derivative_source = self._create_waveform_in_batches(
                self.cache["t"],
                modified_amps,
                self.cache["dummy_ylms"],
                self.cache["phase_coefficients_t"],
                self.cache["phase_coefficients"],
                self.cache["ls_all"],
                self.cache["ms_all"],
                # self.cache['ks_all'],
                self.cache["ns_all"],
                dt=dt,
                T=T,
                batch_size=batch_size,
                show_progress=show_progress,
                **kwargs_waveform,
            )

        # 1PAT1R modification: chi2 (secondary spin) enters only the amplitude,
        # not the trajectory phase. The inspiral trajectory is independent of
        # chi2 at the ODE level (chi2 is stored in Trajectory1PAT1R.args and
        # affects the flux, but _trajectory_from_parameters already ran with the
        # correct chi2 for the injection point). We therefore skip the trajectory
        # perturbation loop entirely and finite-difference only the amplitudes
        # at the cached (unperturbed) trajectory.
        #
        # We use the dense spline-interpolated trajectory (evaluated at the
        # waveform time grid self.cache["t"]) rather than the sparse ODE output
        # self.cache["y"], because _amplitudes_from_trajectory expects a trajectory
        # with the same number of time steps as the waveform.
        elif param_to_vary == "chi2":  # 1PAT1R modification
            base_chi2 = kwargs_remaining.get("chi2", 0.0)  # 1PAT1R modification
            t_interp = self.cache["t"]  # 1PAT1R modification

            # Evaluate the spline at the dense waveform time grid to get y_dense,
            # shape (nparams, N_t), matching what _amplitudes_from_trajectory expects.
            use_gpu = self.waveform_generator.backend.uses_cupy  # 1PAT1R modification
            if use_gpu:  # 1PAT1R modification
                t_interp_np = t_interp.get()  # 1PAT1R modification
            else:  # 1PAT1R modification
                t_interp_np = t_interp  # 1PAT1R modification
            y_dense = self.xp.asarray(  # 1PAT1R modification
                self.inspiral_generator.inspiral_generator.eval_integrator_spline(  # 1PAT1R modification
                    t_interp_np  # 1PAT1R modification
                ).T  # 1PAT1R modification
            )  # shape (nparams, N_t)  # 1PAT1R modification

            amps_steps = self.xp.zeros(  # 1PAT1R modification
                (len(self.deltas), len(t_interp), len(self.cache["ls_all"])),  # 1PAT1R modification
                dtype=self.xp.complex128,  # 1PAT1R modification
            )  # 1PAT1R modification

            for k, delt in enumerate(self.deltas):  # 1PAT1R modification
                kwargs_in = {**kwargs_remaining, "chi2": base_chi2 + delt}  # 1PAT1R modification
                amps_steps[k] = self._amplitudes_from_trajectory(  # 1PAT1R modification
                    parameters,  # 1PAT1R modification
                    t_interp,  # 1PAT1R modification
                    y_dense,  # 1PAT1R modification: dense trajectory, not sparse ODE output
                    theta_source=float(theta_source),  # 1PAT1R modification
                    phi_source=phi_source,  # 1PAT1R modification
                    cache=False,  # 1PAT1R modification
                    **kwargs_in,  # 1PAT1R modification
                )  # 1PAT1R modification

            dAmp_dchi2 = self._stencil(amps_steps, self.delta, self.order, self.kind)  # 1PAT1R modification

            waveform_derivative_source = self._create_waveform_in_batches(  # 1PAT1R modification
                t_interp,  # 1PAT1R modification
                dAmp_dchi2,  # 1PAT1R modification
                self.cache["dummy_ylms"],  # 1PAT1R modification
                self.cache["phase_coefficients_t"],  # 1PAT1R modification
                self.cache["phase_coefficients"],  # 1PAT1R modification
                self.cache["ls_all"],  # 1PAT1R modification
                self.cache["ms_all"],  # 1PAT1R modification
                self.cache["ns_all"],  # 1PAT1R modification
                dt=dt,  # 1PAT1R modification
                T=T,  # 1PAT1R modification
                batch_size=batch_size,  # 1PAT1R modification
                show_progress=show_progress,  # 1PAT1R modification
                **kwargs_waveform,  # 1PAT1R modification
            )  # 1PAT1R modification

        # traj params
        else:
            # if you've reached this point, a derivative w.r.t one of the trajectory parameters is requested.
            # trajectory must be modified

            use_gpu = self.waveform_generator.backend.uses_cupy

            # get trajectories
            y_interps = self.xp.full(
                (len(self.deltas), self.cache["t"].size, len(self.cache["y"])),
                self.xp.nan,
            )  # trajectory for each of the finite difference deltas
            t_interp = self.cache["t"].copy()
            if use_gpu:
                t_interp_np = t_interp.get()
            else:
                t_interp_np = t_interp

            for k, delt in enumerate(self.deltas):
                parameters_in = parameters.copy()
                parameters_in[param_to_vary] += delt  # perturb by finite-difference
                t, y = self._trajectory_from_parameters(parameters_in, T, **kwargs_remaining)
                # re-interpolate onto the time-step grid for the injection trajectory

                if self.xp.around(t_interp[-1], 5) > self.xp.around(
                    t[-1], 5
                ):  # check plunge
                    print("plunging! t_interp: ", t_interp[-1], "t_traj: ", t[-1])

                    mask_notplunging = (
                        t_interp < t[-1]
                    )  # for all t_interp < t[-1], the perturbed trajectory is still not plunging
                    if use_gpu:
                        t_interp_np = t_interp[mask_notplunging].get()
                    else:
                        t_interp_np = t_interp[mask_notplunging]

                y_interp = self.xp.asarray(
                    self.inspiral_generator.inspiral_generator.eval_integrator_spline(
                        t_interp_np
                    ).T
                )  # any unfilled elements due to plunging trajectories assume nans.
                y_interps[k, : len(t_interp_np)] = y_interp.T

            # In the case of plunge, some trajectories will be shorter than others. They appear as NaN's in the y_interps array
            nans = self.xp.isnan(y_interps)
            if nans.any():
                max_ind = int(self.xp.where(nans.sum(2).sum(0) > 0)[0].min())
            else:
                max_ind = len(t_interp_np)

            # modify size of the trajectories and phases accordingly
            t_interp = self.cache["t"][:max_ind]
            y_interps = y_interps[:, :max_ind, :]
            phases_steps = y_interps[:, :, 3:6]  # Phi_phi, Phi_theta, Phi_r
            phase_coefficients = self.xp.asarray(
                self.cache["phase_coefficients"][: max_ind - 1, :]
            )
            phase_t = self.cache["phase_coefficients_t"][:max_ind]

            # finite differencing the phases
            dPhi_fund_dx = self._stencil(
                phases_steps, self.delta, self.order, self.kind
            )
            # project the fundamental phases up to the full mode index space
            dPhi_dx = (
                dPhi_fund_dx[:, 0, None] * self.cache["ms_all"][None, :]
                +
                # dPhi_fund_dx[:,1,None] * self.cache['ks_all'][None,:] + #no inclination in FEW 2.0
                dPhi_fund_dx[:, 2, None] * self.cache["ns_all"][None, :]
            )

            # get amplitude derivative
            amps_steps = self.xp.zeros(
                (len(self.deltas), max_ind, len(self.cache["ls_all"])),
                dtype=self.xp.complex128,
            )

            for k, delt in enumerate(self.deltas):

                parameters_in = parameters.copy()
                parameters_in[param_to_vary] += delt  # perturb by finite-difference
                amps_here = self._amplitudes_from_trajectory(
                    parameters_in,
                    t_interp,
                    y_interps[k].T,
                    theta_source=float(theta_source),
                    phi_source=phi_source,
                    cache=False,
                    **kwargs_remaining,
                )  # remember, this function multiplies by Ylmns!
                amps_steps[k] = amps_here

            # finite differencing the amplitudes
            dAmp_dx = self._stencil(
                amps_steps, self.delta, self.order, self.kind
            )  # actually d teuk_amps/dx * Ylmns

            # defining effective amplitudes = dAdx - i A dPhidx
            wave_amps = self.cache["teuk_modes_with_ylms"][:max_ind]
            effective_amps = dAmp_dx - 1j * wave_amps * dPhi_dx

            if param_to_vary in [
                "m1",
                "m2",
            ]:  # additional term due to chain rule of dist_dimensionless

                if (
                    self.frame == "detector"
                ):  # assuming waveform_generator is always in source frame.

                    mu = (
                        parameters["m1"]
                        * parameters["m2"]
                        / (parameters["m1"] + parameters["m2"])
                    )
                    dist_dimensionless = (parameters["dist"] * Gpc) / (mu * MRSUN_SI)
                    M = parameters["m1"] + parameters["m2"]

                    if param_to_vary == "m1":
                        dmu_dm = parameters["m2"] ** 2 / (M**2)
                    else:
                        dmu_dm = parameters["m1"] ** 2 / (M**2)

                    effective_amps += wave_amps / mu * dmu_dm

            # create derivative
            waveform_derivative_source = self._create_waveform_in_batches(
                t_interp,
                effective_amps,  # actually effective amplitudes * Ylmns
                self.cache["dummy_ylms"],  # just a bunch of ones
                phase_t,
                phase_coefficients,
                self.cache["ls_all"],
                self.cache["ms_all"],
                # self.cache['ks_all'],
                self.cache["ns_all"],
                dt=dt,
                T=T,
                batch_size=batch_size,
                show_progress=show_progress,
                **kwargs_waveform,
            )

            # pad with zeroes if required to get back to the waveform length
            if max_ind < self.cache["t"].size:
                self.cache["Npad"] = (
                    self.cache["waveform_source"].size - waveform_derivative_source.size
                )  # length of zero padding also cached
                waveform_derivative_source = self.xp.concatenate(
                    (
                        waveform_derivative_source,
                        self.xp.zeros(
                            self.cache["Npad"], dtype=waveform_derivative_source.dtype
                        ),
                    ),
                    axis=0,
                )

        # waveform derivative obtained in the source frame. Now we must transform to detector (SSB) frame
        # and apply antenna patterns (and its derivatives in case of qS, phiS, qK, phiK)

        if self.waveform_generator.frame == "source":
            waveform_source_flipped = -1 * self.cache["waveform_source"]
            waveform_derivative_source *= -1

        # decompose to plus and cross
        (waveform_derivative_source_plus, waveform_derivative_source_cross) = (
            waveform_derivative_source.real,
            -waveform_derivative_source.imag,
        )

        if self.frame == "source":
            if self.return_list is False:
                return (
                    waveform_derivative_source_plus
                    - 1j * waveform_derivative_source_cross
                )
            else:
                return [
                    waveform_derivative_source_plus,
                    waveform_derivative_source_cross,
                ]

        # detector frame requested; apply antenna pattern
        (waveform_derivative_detector_plus, waveform_derivative_detector_cross) = (
            self._to_SSB_frame(
                hp=waveform_derivative_source_plus,
                hc=waveform_derivative_source_cross,
                qS=parameters["qS"],
                phiS=parameters["phiS"],
                qK=parameters["qK"],
                phiK=parameters["phiK"],
            )
        )

        # if derivative is with respect to one of the angles, also need derivatives with antenna pattern
        if (param_to_vary in ["qS", "phiS", "qK", "phiK"]) & (
            parameters["phiS"] != parameters["phiK"]
        ):

            Fderivs = fplus_fcross_derivs(
                parameters["qS"],
                parameters["phiS"],
                parameters["qK"],
                parameters["phiK"],
                with_respect_to=param_to_vary,
            )
            dFplusIdx = Fderivs[f"dFplusI/d{param_to_vary}"]
            dFplusIIdx = Fderivs[f"dFplusII/d{param_to_vary}"]
            dFcrossIdx = Fderivs[f"dFcrossI/d{param_to_vary}"]
            dFcrossIIdx = Fderivs[f"dFcrossII/d{param_to_vary}"]

            waveform_derivative_detector_plus += (
                dFplusIdx * waveform_source_flipped.real
                + dFcrossIdx * (-waveform_source_flipped.imag)
            )

            waveform_derivative_detector_cross += (
                dFplusIIdx * waveform_source_flipped.real
                + dFcrossIIdx * (-waveform_source_flipped.imag)
            )

        if self.return_list is False:
            return (
                waveform_derivative_detector_plus
                - 1j * waveform_derivative_detector_cross
            )
        else:
            return [
                waveform_derivative_detector_plus,
                waveform_derivative_detector_cross,
            ]

    def clear_cache(self):
        self.cache = None  # reset cache

    def _trajectory_from_parameters(self, parameters, T, **kwargs):
        """
        calculate the inspiral trajectory over time T (years) for a given set of parameters.

        Args:
            parameters (dict): dictionary of parameters with the param name as key and its value as value.
            T (float): time (in years) for the inspiral trajectory
            **kwargs: additional keyword arguments; ``evolve_primary`` (bool) is
                read here and forwarded as ``additional_args[1]`` to
                ``Trajectory1PAT1R`` when present.
        Returns:
            t (np.ndarray): time steps (in seconds) of the trajectory
            y (np.ndarray): evolving parameters of the trajectory along the time grid
        """

        # Keys that are standard positional args to inspiral_generator or are
        # handled explicitly below (chi2) — everything else becomes additional_args.
        _STANDARD_KEYS = {
            "m1", "m2", "a", "p0", "e0", "xI0",
            "Phi_phi0", "Phi_theta0", "Phi_r0",
            "dist", "qS", "phiS", "qK", "phiK",
            "chi2",  # 1PAT1R modification: forwarded explicitly below
        }
        add_parameters = [v for k, v in parameters.items() if k not in _STANDARD_KEYS]

        # 1PAT1R modification: build additional_args = [chi2, evolve_primary?, ...]
        # chi2 is positional additional_args[0]; evolve_primary, if supplied via
        # kwargs, is additional_args[1] (Trajectory1PAT1R defaults it to True).
        chi2_val = parameters.get("chi2", None)
        if chi2_val is not None:
            evolve_primary = kwargs.get("evolve_primary", None)
            prefix = [chi2_val, evolve_primary] if evolve_primary is not None else [chi2_val]
            add_parameters = prefix + add_parameters

        traj = self.inspiral_generator(
            parameters["m1"],
            parameters["m2"],
            parameters["a"],
            parameters["p0"],
            parameters.get("e0", 0.0),   # 1PAT1R modification: default to circular
            parameters.get("xI0", 1.0),  # 1PAT1R modification: default to equatorial prograde
            *add_parameters,              # any extra trajectory parameters (e.g. chi2)
            Phi_phi0=parameters["Phi_phi0"],
            Phi_theta0=parameters.get("Phi_theta0", 0.0),  # 1PAT1R modification: not used in circular
            Phi_r0=parameters.get("Phi_r0", 0.0),          # 1PAT1R modification: not used in circular
            T=T,
            **self.inspiral_kwargs,
        )  # generate the trajectory

        # convert for gpus
        traj = self.xp.asarray([self.xp.asarray(traj[i]) for i in range(len(traj))])

        t = traj[0]
        y = traj[1:]

        return t, y

    def _amplitudes_from_trajectory(
        self, parameters, t, y, theta_source, phi_source, cache=False, **kwargs
    ):
        """
        calculate the amplitudes (and ylms) from the trajectory.

        Args:
            parameters (dict): dictionary of trajectory parameters
            t (np.ndarray): array of time steps for trajectory
            y (np.ndarray): array of evolving parameters in the trajectory at time steps
            theta_source (np.float): polar angle in source frame
            phi_source (np.float): azimuthal angle in source frame
            cache (bool): whether to cache info (True) or not (False)
        Returns:
            Teukolsky amplitudes times Ylms
        """

        # if detector frame, scale by distance in the amplitude module as well.
        if (
            self.frame == "detector"
        ):  # assuming waveform generator is always in source frame
            mu = (
                parameters["m1"]
                * parameters["m2"]
                / (parameters["m1"] + parameters["m2"])
            )
            dist_dimensionless = (parameters["dist"] * Gpc) / (mu * MRSUN_SI)
        else:
            dist_dimensionless = 1.0

        # 1PAT1R modification: AmpInterp1PAT1R requires nu, chit2, chit, and
        # delta_m1 in addition to (a, p, e, xI). self._is_1PA is set at
        # construction time from the waveform descriptor.
        if self._is_1PA:
            # Trajectory1PAT1R state vector layout (y rows after t):
            #   y[0]=p, y[1]=e, y[2]=xI, y[3]=Phi_phi, y[4]=Phi_theta,
            #   y[5]=Phi_r, y[6]=deltaM, y[7]=delta_chit1
            nu = (  # 1PAT1R modification
                parameters["m1"] * parameters["m2"]
                / (parameters["m1"] + parameters["m2"]) ** 2
            )
            chi2_val = kwargs.get("chi2", parameters.get("chi2", 0.0))  # 1PAT1R modification: kwargs takes priority so chi2 derivative branch can perturb it
            chi1_val = parameters["a"]  # 1PAT1R modification
            chit1_0, chit2 = chi_to_chit(chi1_val, chi2_val, nu)  # 1PAT1R modification
            delta_chit1 = y[7]  # 1PAT1R modification
            delta_m1 = y[6]  # 1PAT1R modification
            chit = chit1_0 + delta_chit1  # 1PAT1R modification: total evolving reduced primary spin
            teuk_modes = self.xp.asarray(
                self.amplitude_generator(
                    parameters["a"],
                    *y[:3],             # p, e, xI along trajectory
                    nu=nu,              # 1PAT1R modification
                    chit2=chit2,        # 1PAT1R modification
                    chit=chit,          # 1PAT1R modification
                    delta_m1=delta_m1,  # 1PAT1R modification
                    zero_PA_amps_only=kwargs.get("zero_PA_amps_only", False),  # 1PAT1R modification
                )
            )
        else:
            # standard (non-1PA) amplitude call
            teuk_modes = self.xp.asarray(
                self.amplitude_generator(parameters["a"], *y[:3])
            )

        # ylms
        ylms = self.ylm_gen(self.unique_l, self.unique_m, theta_source, phi_source).copy()[
            self.inverse_lm
        ]

        if cache:
            # perform mode selection in the first call (with cache=True)
            mode_selection = None

            fund_freq_args = (
                parameters["m1"],
                parameters["m2"],
                parameters["a"],
                y[0],
                y[1],
                y[2],
                t,
            )

            modeinds = [self.l_arr, self.m_arr, self.n_arr]

            # 1PAT1R modification: strip keys that mode_selector does not accept.
            _mode_sel_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in {"chi2", "evolve_primary", "zero_PA_amps_only"}
            }
            (
                teuk_modes_in,
                ylms_in,
                self.ls,
                self.ms,
                self.ns,
            ) = self.mode_selector(
                teuk_modes,
                ylms,
                modeinds,
                fund_freq_args=fund_freq_args,
                mode_selection=mode_selection,  # None
                **_mode_sel_kwargs,  # 1PAT1R modification
            )

            # we don't use mode symmetry
            m0mask = self.ms != 0
            teuk_modes_in = self.xp.concatenate(
                (
                    teuk_modes_in,
                    (-1) ** (self.ls[m0mask]) * self.xp.conj(teuk_modes_in[:, m0mask]),
                ),
                axis=1,
            )

            ylms_in = self.xp.concatenate(
                (ylms_in[: self.ls.size], ylms_in[self.ls.size :][m0mask]), axis=0
            )

            self.cache["ls_all"] = self.xp.concatenate(
                (self.ls, self.ls[m0mask]), axis=0
            )
            self.cache["ms_all"] = self.xp.concatenate(
                (self.ms, -self.ms[m0mask]), axis=0
            )

            self.cache["ns_all"] = self.xp.concatenate(
                (self.ns, -self.ns[m0mask]), axis=0
            )

            self.cache["ls"] = self.ls.copy()
            self.cache["ms"] = self.ms.copy()
            self.cache["ns"] = self.ns.copy()

            self.cache["mode_selection"] = [
                (l, m, n)
                for l, m, n in zip(self.cache["ls"], self.cache["ms"], self.cache["ns"])
            ]

            mode_map = {
                (int(l), int(m), int(n)): idx
                for idx, (l, m, n) in enumerate(zip(self.l_arr, self.m_arr, self.n_arr))
            }

            # recover the indices for the cached (ls,ms,ns) for subsequent calls.
            self.cache["keep_inds"] = [
                mode_map[(int(l), int(m), int(n))]
                for l, m, n in zip(self.ls, self.ms, self.ns)
            ]

            self.cache["teuk_modes"] = teuk_modes_in / dist_dimensionless
            self.cache["teuk_modes_with_ylms"] = self.cache["teuk_modes"] * ylms_in
            self.cache["ylms_in"] = ylms_in
            self.cache["dummy_ylms"] = self.xp.ones(
                2 * teuk_modes_in.shape[1], dtype=self.xp.complex128
            )
            self.cache["dummy_ylms"][teuk_modes_in.shape[1] :] = 0.0
            self.cache["m0mask"] = m0mask

            return self.cache["teuk_modes_with_ylms"]

        else:

            teuk_modes_in = teuk_modes[
                :, self.cache["keep_inds"]
            ]  # same modes as in the first run, the amplitudes are just different

            teuk_modes_in = self.xp.concatenate(
                (
                    teuk_modes_in,
                    (-1) ** (self.ls[self.cache["m0mask"]])
                    * self.xp.conj(teuk_modes_in[:, self.cache["m0mask"]]),
                ),
                axis=1,
            )  # get the negative m modes as well

            ylms_in = self.cache[
                "ylms_in"
            ].copy()  # already calculated in the first run, so just copy it

            return teuk_modes_in * ylms_in / dist_dimensionless

    def _create_waveform_in_batches(
        self,
        t,
        amps,
        ylms,
        phase_t,
        phase_coeffs,
        ls,
        ms,
        ns,
        dt,
        T,
        batch_size=-1,
        show_progress=False,
        **kwargs,
    ):
        """
        A wrapper for self.create_waveform that handles batching over the time axis.
        """
        # split into batches
        if batch_size == -1 or self.allow_batching is False:
            inds_split_all = [self.xp.arange(len(t))]
        else:
            split_inds = []
            i = 0
            while i < len(t):
                i += batch_size
                if i >= len(t):
                    break
                split_inds.append(i)

            inds_split_all = self.xp.split(self.xp.arange(len(t)), split_inds)

        iterator = enumerate(inds_split_all)
        iterator = (
            tqdm(iterator, desc="time batch", total=len(inds_split_all))
            if show_progress
            else iterator
        )

        for i, inds_in in iterator:
            t_batch = t[inds_in]
            amps_batch = amps[inds_in]

            waveform_batch = self.create_waveform(
                t_batch,
                amps_batch,
                ylms,
                phase_t,
                phase_coeffs,
                l_arr=ls,
                m_arr=ms,
                n_arr=ns,
                dt=dt,
                T=T,
                **kwargs,
            )

            if i == 0:
                waveform = waveform_batch
            else:
                waveform = self.xp.concatenate((waveform, waveform_batch))

        return waveform

    def _modify_amplitudes_for_initial_phase_derivative(self, param_to_vary):
        """
        calculates modified amplitudes for phase derivatives.
        """

        if param_to_vary == "Phi_phi0":
            factor = -1j * self.cache["ms_all"]
        elif param_to_vary == "Phi_theta0":
            raise NotImplementedError  # no inclination in FEW 2.0
        elif param_to_vary == "Phi_r0":
            factor = -1j * self.cache["ns_all"]

        modified_amps = self.cache["teuk_modes_with_ylms"] * factor[None, :]
        return modified_amps

    def _modify_amplitudes_for_angle_derivative(
        self, parameters, param_to_vary, theta_source, phi_source
    ):
        """
        calculates modified amplitudes for angle derivatives (qS, phiS, qK, phiK)
        """

        ylm_temp = self.xp.zeros(
            (len(self.deltas), self.cache["ls_all"].size), dtype=self.xp.complex128
        )

        for k, delt in enumerate(self.deltas):

            theta_source_perturb = theta_source + float(delt)
            phi_source_perturb = phi_source
            ylm_temp[k] = self.ylm_gen(
                self.cache["ls_all"],
                self.cache["ms_all"],
                theta_source_perturb,
                phi_source_perturb,
            )

        dYlm_dtheta = self._stencil(ylm_temp, self.delta, self.order, self.kind)

        if param_to_vary == "qS":
            key_dtheta_dx = "del theta_src / del qS"
        elif param_to_vary == "phiS":
            key_dtheta_dx = "del theta_src / del phi_S"
        elif param_to_vary == "qK":
            key_dtheta_dx = "del theta_src / del qK"
        elif param_to_vary == "phiK":
            key_dtheta_dx = "del theta_src / del phi_K"

        dtheta_dx = viewing_angle_partials(
            parameters["qS"], parameters["phiS"], parameters["qK"], parameters["phiK"]
        )[key_dtheta_dx]

        modified_amps = self.cache["teuk_modes"] * dYlm_dtheta[None, :] * dtheta_dx

        return modified_amps

    def _deltas(self, delta, order, kind):
        """
        return the np.ndarray of parameter deltas for a given delta
        """

        if kind == "central":
            half = order // 2
            positions = list(range(-half, 0)) + list(range(1, half + 1))

        elif kind == "forward":
            positions = list(range(order + 1))

        elif kind == "backward":
            positions = list(range(-order, 1))

        return np.array(positions) * delta

    def _available_stencils(self):
        """
        Accessed from Fornberg 1988: https://doi.org/10.1090%2FS0025-5718-1988-0935077-0
        """
        return {
            "central": {
                2: self.xp.asarray([-1 / 2, 1 / 2]),
                4: self.xp.asarray([1 / 12, -2 / 3, 2 / 3, -1 / 12]),
                6: self.xp.asarray([-1 / 60, 3 / 20, -3 / 4, 3 / 4, -3 / 20, 1 / 60]),
                8: self.xp.asarray(
                    [1 / 280, -4 / 105, 1 / 5, -4 / 5, 4 / 5, -1 / 5, 4 / 105, -1 / 280]
                ),
            },
            "forward": {
                2: self.xp.asarray([-3 / 2, 2, -1 / 2]),
                4: self.xp.asarray([-25 / 12, 4, -3, 4 / 3, -1 / 4]),
                6: self.xp.asarray(
                    [-49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6]
                ),
                8: self.xp.asarray(
                    [
                        -761 / 280,
                        8,
                        -14,
                        56 / 3,
                        -35 / 2,
                        56 / 5,
                        -14 / 3,
                        8 / 7,
                        -1 / 8,
                    ]
                ),
            },
            "backward": {
                2: self.xp.asarray([1 / 2, -2, 3 / 2]),
                4: self.xp.asarray([1 / 4, -4 / 3, 3, -4, 25 / 12]),
                6: self.xp.asarray(
                    [1 / 6, -6 / 5, 15 / 4, -20 / 3, 15 / 2, -6, 49 / 20]
                ),
                8: self.xp.asarray(
                    [1 / 8, -8 / 7, 14 / 3, -56 / 5, 35 / 2, -56 / 3, 14, -8, 761 / 280]
                ),
            },
        }

    def _stencil(self, func_steps, delta, order, kind):
        """
        return the stencil for finite-differences
        """

        return (
            self.xp.tensordot(
                self._available_stencils()[kind][order], func_steps, axes=(0, 0)
            )
            / delta
        )