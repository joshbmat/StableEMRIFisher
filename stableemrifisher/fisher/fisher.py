"""Stable EMRI Fisher-matrix utilities.

This module provides the `StableEMRIFisher` class to compute signal-to-noise
ratio (SNR), select numerically stable finite-difference step sizes ("stable
deltas"), and build Fisher information matrices for Extreme Mass Ratio
Inspirals (EMRIs) using the FEW toolkit. It supports optional LISA response
wrapping, basic plunge checks, optional GPU acceleration via CuPy, and
convenience plotting/saving utilities.
"""

import os
import sys
import time
import logging
from typing import Optional, Union, Dict, List, Tuple, Any, Callable, Type

import numpy as np
import h5py
import matplotlib.pyplot as plt

try:
    import cupy as cp

    ArrayType = Union[np.ndarray, cp.ndarray]
except ImportError:
    cp = None
    ArrayType = np.ndarray

from few.utils.constants import YRSID_SI
from few.waveform import GenerateEMRIWaveform
from stableemrifisher.fisher.derivatives import derivative, handle_a_flip
from stableemrifisher.fisher.stablederivative import StableEMRIDerivative
from stableemrifisher.utils import inner_product, SNRcalc, generate_PSD
from stableemrifisher.noise import sensitivity_LWA, write_psd_file, load_psd_from_file
from stableemrifisher.plot import CovEllipsePlot, StabilityPlot

logger = logging.getLogger("stableemrifisher")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel("INFO")
logger.info("startup")


class StableEMRIFisher:
    """Compute stable Fisher matrices for EMRI signals."""

    # 1PAT1R modification: canonical 14-arg positional order for
    # GenerateEMRIWaveform.__call__ and fixed values for the parameters
    # that are absent from the circular-model Fisher parameter set.
    _CIRC_FIXED = {"e0": 0.0, "xI0": 1.0, "Phi_theta0": 0.0, "Phi_r0": 0.0}
    _FULL_ORDER = [
        "m1", "m2", "a", "p0", "e0", "xI0",
        "dist", "qS", "phiS", "qK", "phiK",
        "Phi_phi0", "Phi_theta0", "Phi_r0",
    ]

    def __init__(
        self,
        *,
        waveform_class: Type,
        waveform_class_kwargs: Optional[Dict[str, Any]] = None,
        waveform_generator: Type = GenerateEMRIWaveform,
        waveform_generator_kwargs: Optional[Dict[str, Any]] = None,
        ResponseWrapper: Optional[Type] = None,
        ResponseWrapper_kwargs: Optional[Dict[str, Any]] = None,
        noise_model: Optional[Callable] = None,
        noise_kwargs: Optional[Dict[str, Any]] = None,
        channels: Optional[List[str]] = None,
        deriv_type: str = "stable",
        stats_for_nerds: bool = False,
        use_gpu: bool = False,
        dt: float = 10.0,
        T: float = 1.0,
        der_order: int = 2,
        Ndelta: int = 8,
        CovEllipse: bool = False,
        stability_plot: bool = False,
        save_derivatives: bool = False,
        filename: Optional[str] = None,
        plunge_check: bool = True,
        return_derivatives: bool = False,
        waveform_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.waveform = None
        self.wave_params = {}
        self.traj_params = {}
        self.wave_params_list = []
        self.SNR2 = None
        self.PSD_funcs = None

        self.use_gpu = use_gpu
        if self.use_gpu and cp is None:
            logger.warning("CuPy not found; disabling GPU acceleration.")
            self.use_gpu = False

        if stats_for_nerds:
            logger.setLevel("DEBUG")

        if waveform_class_kwargs is None:
            waveform_class_kwargs = {}

        if waveform_generator_kwargs is None:
            waveform_generator_kwargs = {**waveform_class_kwargs}
        else:
            waveform_generator_kwargs = {
                **waveform_class_kwargs,
                **waveform_generator_kwargs,
            }

        if ResponseWrapper_kwargs is None:
            ResponseWrapper_kwargs = {}
        elif "waveform_gen" in ResponseWrapper_kwargs:
            logger.warning(
                "ResponseWrapper_kwargs should not "
                "contain 'waveform_gen'. It will be set automatically."
            )
            ResponseWrapper_kwargs.pop("waveform_gen")

        waveform_generator = waveform_generator(
            waveform_class=waveform_class,
            **waveform_generator_kwargs,
        )
        self.waveform_generator_kwargs = waveform_generator_kwargs

        self.traj_module = waveform_generator.waveform_generator.inspiral_generator
        self.traj_module_func = waveform_generator.waveform_generator.inspiral_kwargs[
            "func"
        ]

        _wg_class_name = waveform_generator.waveform_generator.__class__.__name__
        if _wg_class_name == "Pn5AAKWaveform" and deriv_type == "stable":
            logger.warning(
                "5PNAAK waveform model is incompatible with deriv_type 'stable'. "
                "Switching to deriv_type 'direct'."
            )
            deriv_type = "direct"

        # 1PAT1R modification: flag used throughout to enable circular-orbit-specific logic.
        self._is_circular = (  # 1PAT1R modification
            getattr(waveform_generator.waveform_generator, "descriptor", "").lower() == "circular"  # 1PAT1R modification
        )

        self.deriv_type = deriv_type
        if self.deriv_type == "stable":
            waveform_derivative = StableEMRIDerivative(
                waveform_class=waveform_class,
                **waveform_generator_kwargs,
            )
            self.waveform_derivative_kwargs = {}
            self._deltas = waveform_derivative._deltas
            self._stencil = waveform_derivative._stencil
        elif self.deriv_type == "direct":
            waveform_derivative = derivative
            self.waveform_derivative_kwargs = {"use_gpu": self.use_gpu}
        else:
            raise ValueError("deriv_type must be 'stable' or 'direct'.")

        if ResponseWrapper is not None:
            self.waveform_generator = ResponseWrapper(
                waveform_generator, **ResponseWrapper_kwargs
            )
            if self.deriv_type == "direct":
                self.derivative = waveform_derivative
                self.waveform_derivative_kwargs.update(
                    {"waveform_generator": self.waveform_generator}
                )
            else:
                response_for_derivative = ResponseWrapper(
                    waveform_derivative, **ResponseWrapper_kwargs
                )
                self.derivative = response_for_derivative
            self.has_ResponseWrapper = True
            self.T = ResponseWrapper_kwargs["Tobs"]
            self.dt = ResponseWrapper_kwargs["dt"]
        else:
            self.waveform_generator = waveform_generator
            self.derivative = waveform_derivative
            self.T = T
            self.dt = dt
            if self.deriv_type == "direct":
                self.waveform_derivative_kwargs.update(
                    {"waveform_generator": self.waveform_generator}
                )
            self.has_ResponseWrapper = False

        if noise_model is None and self.has_ResponseWrapper is True:
            logger.info("No noise model provided but response has been provided")
            logger.info("Generating and loading default PSD file")
            run_direc = os.getcwd()
            if ResponseWrapper_kwargs["tdi"] == "2nd generation":
                PSD_filename = "tdi2_wo_background.npy"
                kwargs_PSD = {"stochastic_params": [T * YRSID_SI]}
                write_psd_file(
                    model="scirdv1",
                    channels="AE",
                    tdi2=True,
                    include_foreground=False,
                    filename=run_direc + PSD_filename,
                    **kwargs_PSD,
                )
                logger.info("\nTDI2 A and E with stochastic background.")
            else:
                PSD_filename = "tdi1_wo_background.npy"
                kwargs_PSD = {"stochastic_params": [T * YRSID_SI]}
                write_psd_file(
                    model="scirdv1",
                    channels="AE",
                    tdi2=False,
                    include_foreground=False,
                    filename=run_direc + PSD_filename,
                    **kwargs_PSD,
                )
                logger.info("\nTDI1 A and E with stochastic background.")

            if self.use_gpu:
                self.noise_model = load_psd_from_file(run_direc + PSD_filename, xp=cp)
            else:
                self.noise_model = load_psd_from_file(run_direc + PSD_filename, xp=np)
            self.noise_kwargs = {}
            self.channels = ["A", "E"]
        elif noise_model is None and self.has_ResponseWrapper is False:
            logger.warning("No noise model or response wrapper provided.")
            logger.warning("Defaulting to the sky-averaged sensitivity curve")
            self.noise_model = sensitivity_LWA
            self.noise_kwargs = {}
            self.channels = channels if channels is not None else ["I", "II"]
        else:
            self.noise_model = noise_model
            self.noise_kwargs = (
                noise_kwargs if noise_kwargs is not None else {"TDI": "TDI1"}
            )
            self.channels = channels if channels is not None else ["A", "E"]

        self.minmax = {
            "a": [-0.15, 0.15] if self._is_circular else [0.05, 0.95],  # 1PAT1R modification: Trajectory1PAT1R valid range is [-0.2, 0.2]
            "chi2": [-0.95, 0.95],  # 1PAT1R modification
            "e0": [0.01, 0.7],
            "Phi_phi0": [0.1, 2 * np.pi * 0.9],
            "Phi_r0": [0.1, 2 * np.pi * 0.9],
            "Phi_theta0": [0.1, 2 * np.pi * 0.9],
            "qS": [0.1, np.pi * 0.9],
            "qK": [0.1, np.pi * 0.9],
            "phiS": [0.1, 2 * np.pi * 0.9],
            "phiK": [0.1, 2 * np.pi * 0.9],
        }

        self.order = der_order
        self.Ndelta = Ndelta
        self.CovEllipse = CovEllipse
        self.stability_plot = stability_plot
        self.save_derivatives = save_derivatives
        self.plunge_check = plunge_check
        self.return_derivatives = return_derivatives

        if waveform_kwargs is not None:
            self.waveform_kwargs = {**waveform_kwargs}
        else:
            self.waveform_kwargs = {}

        self.param_names: Optional[List[str]] = None
        self.npar: Optional[int] = None
        self.deltas: Optional[Dict[str, float]] = None
        self.current_waveform_kwargs: Optional[Dict[str, Any]] = None
        self.delta_range: Optional[Dict[str, Tuple[float, float]]] = None

        self.window = None
        self.fmin = None
        self.fmax = None
        self.freq_mask = None
        self.filename = filename
        self.suffix = None

    def __call__(
        self,
        wave_params: Dict[str, float],
        add_param_args: Optional[Dict[str, Any]] = None,
        waveform_kwargs: Optional[Dict[str, Any]] = None,
        window: Optional[Union[np.ndarray, Any]] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        freq_mask: Optional[np.ndarray] = None,
        param_names: Optional[List[str]] = None,
        deltas: Optional[Dict[str, float]] = None,
        der_order: Optional[int] = None,
        kind: Optional[str] = None,
        Ndelta: Optional[int] = None,
        delta_range: Optional[Dict[str, List[float]]] = None,
        CovEllipse: Optional[bool] = None,
        stability_plot: Optional[bool] = None,
        save_derivatives: Optional[bool] = None,
        return_derivatives: Optional[bool] = None,
        live_dangerously: Optional[bool] = None,
        plunge_check: Optional[bool] = None,
        filename: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        if deltas is not None and len(deltas) != len(param_names):
            logger.critical(
                "Length of deltas array should be equal to "
                "length of param_names.\nAssuming deltas = None."
            )
            deltas = None
        self.deltas = deltas

        self.order = der_order if der_order is not None else self.order
        self.Ndelta = Ndelta if Ndelta is not None else self.Ndelta
        self.kind = kind if kind is not None else "central"
        self.window = window
        self.fmin = fmin
        self.fmax = fmax
        self.freq_mask = freq_mask
        if self.freq_mask is not None:
            self.fmin = None
            self.fmax = None

        self.CovEllipse = CovEllipse if CovEllipse is not None else self.CovEllipse
        self.stability_plot = (
            stability_plot if stability_plot is not None else self.stability_plot
        )
        self.save_derivatives = (
            save_derivatives if save_derivatives is not None else self.save_derivatives
        )
        self.filename = filename
        self.suffix = suffix
        self.plunge_check = (
            plunge_check if plunge_check is not None else self.plunge_check
        )
        self.return_derivatives = (
            return_derivatives
            if return_derivatives is not None
            else self.return_derivatives
        )

        call_waveform_kwargs = dict(self.waveform_kwargs)
        if waveform_kwargs is not None:
            call_waveform_kwargs.update(waveform_kwargs)

        call_waveform_kwargs.update({"dt": self.dt, "T": self.T})

        self.current_waveform_kwargs = call_waveform_kwargs

        if delta_range is None:
            self.delta_range = {}
        else:
            self.delta_range = delta_range

        # 1PAT1R modification: copy so we never mutate the caller's dict. chi2
        # must be routed as a keyword argument to GenerateEMRIWaveform.__call__()
        # and must NOT appear as a positional arg. We keep chi2 in self.wave_params
        # for all downstream name-keyed reads (delta heuristics, bounds checks,
        # CovEllipsePlot) but exclude it from the positional wave_params_list.
        self.wave_params = dict(wave_params)  # 1PAT1R modification: copy, not reference
        chi2_val = self.wave_params.get("chi2", None)  # 1PAT1R modification
        if chi2_val is not None:  # 1PAT1R modification
            self.current_waveform_kwargs["chi2"] = chi2_val  # 1PAT1R modification

        if param_names is None:
            if self.has_ResponseWrapper:
                EMRI_ORBIT = (
                    self.waveform_generator.waveform_gen.waveform_generator.descriptor
                )
                BACKGROUND = (
                    self.waveform_generator.waveform_gen.waveform_generator.background
                )
            else:
                EMRI_ORBIT = (
                    self.waveform_generator.waveform_generator.descriptor
                )
                BACKGROUND = (
                    self.waveform_generator.waveform_generator.background
                )

            logger.info("EMRI_ORBIT: %s BACKGROUND: %s", EMRI_ORBIT, BACKGROUND)

            if EMRI_ORBIT == "eccentric equatorial" and BACKGROUND == "Kerr":
                param_names = [
                    "m1", "m2", "a", "p0", "e0",
                    "dist", "qS", "phiS", "qK", "phiK",
                    "Phi_phi0", "Phi_r0",
                ]
            elif EMRI_ORBIT == "eccentric equatorial" and BACKGROUND == "Schwarzschild":
                param_names = [
                    "m1", "m2", "p0", "e0",
                    "dist", "qS", "phiS", "qK", "phiK",
                    "Phi_phi0", "Phi_r0",
                ]
            elif EMRI_ORBIT == "eccentric inclined" and BACKGROUND == "Kerr":
                param_names = [
                    "m1", "m2", "a", "p0", "e0", "xI0",
                    "dist", "qS", "phiS", "qK", "phiK",
                    "Phi_phi0", "Phi_r0",
                ]
            elif EMRI_ORBIT == "eccentric inclined" and BACKGROUND == "Schwarzschild":
                param_names = [
                    "m1", "m2", "p0", "e0", "xI0",
                    "dist", "qS", "phiS", "qK", "phiK",
                    "Phi_phi0", "Phi_r0",
                ]
            elif EMRI_ORBIT.lower() == "circular" and BACKGROUND == "Kerr":
                # 1PAT1R modification: default parameter set for the quasi-circular
                # Kerr 1PA model. chi2 is included as a Fisher parameter; it is
                # routed as a keyword argument inside derivative() and
                # GenerateEMRIWaveform.__call__() — never as a positional arg.
                param_names = [
                    "m1", "m2", "a", "p0",
                    "dist", "qS", "phiS", "qK", "phiK",
                    "Phi_phi0", "chi2",
                ]  # 1PAT1R modification
            else:
                raise ValueError(
                    f"Cannot auto-detect param_names for waveform with "
                    f"descriptor='{EMRI_ORBIT}' and background='{BACKGROUND}'. "
                    "Please pass param_names explicitly."
                )

        self.param_names = param_names
        self.npar = len(self.param_names)

        if self._is_circular:
            # 1PAT1R modification: Trajectory1PAT1R positional signature is
            # (m1, m2, a [=chi1], p0). chi2 is passed separately as a kwarg;
            # extrinsic/phase parameters are excluded here.
            _traj_keys = ["m1", "m2", "a", "p0"]  # 1PAT1R modification
        else:
            _traj_keys = ["m1", "m2", "a", "p0", "e0", "xI0"]
        self.traj_params = {
            k: self.wave_params[k] for k in _traj_keys if k in self.wave_params
        }

        if add_param_args is not None:
            for k, v in add_param_args.items():
                self.wave_params[k] = v
                self.traj_params[k] = v

        if self._is_circular:
            # 1PAT1R modification: pad wave_params with the fixed circular-orbit
            # values so the positional 14-arg call to GenerateEMRIWaveform is
            # always complete. chi2 is excluded — it travels as a kwarg.
            _merged = {**self._CIRC_FIXED, **self.wave_params}
            self.wave_params_positional = {k: _merged[k] for k in self._FULL_ORDER if k in _merged}
        else:
            self.wave_params_positional = {k: v for k, v in self.wave_params.items() if k != "chi2"}
        self.wave_params_list = list(self.wave_params_positional.values())

        if self.plunge_check:
            final_time = self.check_if_plunging()
            self.T = final_time / YRSID_SI
            self.current_waveform_kwargs.update({"T": self.T})

        rho = self.SNRcalc_SEF(
            fmin=self.fmin,
            fmax=self.fmax,
            freq_mask=self.freq_mask,
            window=self.window,
            use_gpu=self.use_gpu,
            *self.wave_params_list,
            **self.current_waveform_kwargs,
        )

        self.SNR2 = rho**2

        logger.info("Waveform Generated. SNR: %s", rho)

        if rho <= 20.0:
            logger.critical(
                "The optimal source SNR is <= 20. "
                + "The Fisher approximation may not be valid!"
            )

        if self.deriv_type == "direct":
            self.waveform_derivative_kwargs.update(
                {
                    # 1PAT1R modification: wave_params_positional is the 14-arg
                    # positional dict (no chi2, dummy e0/x0/phases for circular).
                    # chi2 is already in current_waveform_kwargs and will be
                    # forwarded as a kwarg through derivative() -> waveform_generator.
                    "parameters": self.wave_params_positional,  # 1PAT1R modification
                    "waveform": self.waveform,
                    "order": self.order,
                    "waveform_kwargs": self.current_waveform_kwargs,
                }
            )
        else:
            self.waveform_derivative_kwargs.update(
                {
                    "parameters": self.wave_params_positional,  # 1PAT1R modification
                    "order": self.order,
                    **self.current_waveform_kwargs,
                }
            )

        if self.filename is not None:
            if not os.path.exists(self.filename):
                os.makedirs(self.filename)

        if not live_dangerously:
            if self.deltas is None:
                start = time.time()
                self.Fisher_Stability()
                end = time.time() - start
                logger.info("Time taken to compute stable deltas is %s seconds", end)
        else:
            logger.debug("You have elected for dangerous living, I like it. ")
            fudge_factor_intrinsic = (
                3
                * (self.wave_params["m2"] / self.wave_params["m1"])
                * (1 / self.SNR2) ** (1 / 2)
            )
            # 1PAT1R modification: chi2 added to the set of intrinsic parameters
            # that use a fractional step size (relative to the parameter value,
            # or absolute when the value is zero/unit-scale like spins).
            _intrinsic_params = {"m1", "m2", "a", "p0", "e0", "xI0", "chi2"}  # 1PAT1R modification
            _extrinsic_params = {
                "dist", "qS", "phiS", "qK", "phiK",
                "Phi_phi0", "Phi_theta0", "Phi_r0",
            }
            danger_delta_dict = {}
            for pname in self.param_names:
                if pname in _intrinsic_params:
                    # 1PAT1R modification: spins a and chi2 are dimensionless so
                    # use scale=1.0 rather than the parameter value itself.
                    scale = self.wave_params[pname] if pname not in {"a", "chi2"} else 1.0  # 1PAT1R modification
                    danger_delta_dict[pname] = fudge_factor_intrinsic * scale
                elif pname in _extrinsic_params:
                    danger_delta_dict[pname] = 1e-6
                else:
                    val = self.wave_params.get(pname, 1.0)
                    danger_delta_dict[pname] = fudge_factor_intrinsic * abs(val) if val != 0.0 else 1e-6

            self.deltas = danger_delta_dict
            self.save_deltas()

        start = time.time()
        Fisher = self.FisherCalc()
        end = time.time() - start
        logger.info("Time taken to compute FM is %s seconds", end)

        if self.CovEllipse:
            covariance = np.linalg.inv(Fisher)
            if self.filename is not None:
                if self.suffix is not None:
                    CovEllipsePlot(
                        covariance,
                        self.param_names,
                        self.wave_params,
                        filename=os.path.join(
                            self.filename, f"covariance_ellipses_{self.suffix}.png"
                        ),
                    )
                else:
                    CovEllipsePlot(
                        covariance,
                        self.param_names,
                        self.wave_params,
                        filename=os.path.join(self.filename, "covariance_ellipses.png"),
                    )
            else:
                CovEllipsePlot(covariance, self.param_names, self.wave_params)
                plt.show()
            return Fisher, covariance

        return Fisher

    def SNRcalc_SEF(
        self,
        *waveform_args,
        window=None,
        fmin=None,
        fmax=None,
        use_gpu=False,
        **waveform_kwargs,
    ):
        if self.use_gpu:
            xp = cp
        else:
            xp = np

        try:
            dt = waveform_kwargs["dt"]
        except KeyError as e:
            raise ValueError(f"waveform_kwargs must include {e}.") from e

        self.waveform = xp.asarray(
            self.waveform_generator(*waveform_args, **waveform_kwargs)
        )

        if not self.has_ResponseWrapper:
            self.waveform = xp.asarray([self.waveform.real, -self.waveform.imag])

        logger.info("waveform shape: %s", self.waveform.shape)
        logger.debug("wave ndim: %s", self.waveform.ndim)

        self.PSD_funcs = generate_PSD(
            waveform=self.waveform,
            dt=dt,
            noise_PSD=self.noise_model,
            channels=self.channels,
            noise_kwargs=self.noise_kwargs,
            use_gpu=self.use_gpu,
        )

        logger.info("Computing SNR for parameters: %s", waveform_args)

        return SNRcalc(
            self.waveform,
            self.PSD_funcs,
            dt=dt,
            window=window,
            fmin=fmin,
            fmax=fmax,
            use_gpu=use_gpu,
        )

    def check_if_plunging(self):
        """Check for plunge and return an adjusted evolution time (seconds)."""
        traj_vals = list(handle_a_flip(self.traj_params).values())

        if self._is_circular:
            # 1PAT1R modification: insert fixed e0=0/xI0=1 before additional_args
            # = [chi2, evolve_primary] to match EMRIInspiral positional signature.
            chi2 = self.current_waveform_kwargs.get("chi2", 0.0)
            evolve_primary = self.current_waveform_kwargs.get("evolve_primary", True)
            t_traj = self.traj_module(
                *traj_vals, 0.0, 1.0, chi2, evolve_primary,
                Phi_phi0=self.wave_params["Phi_phi0"],
                T=self.T, dt=self.dt,
            )[0]
        else:
            t_traj = self.traj_module(
                *traj_vals,
                Phi_phi0=self.wave_params["Phi_phi0"],
                Phi_theta0=self.wave_params["Phi_theta0"],
                Phi_r0=self.wave_params["Phi_r0"],
                T=self.T,
                dt=self.dt,
            )[0]

        if t_traj[-1] < self.T * YRSID_SI - 1.0:
            logger.warning("Body is plunging! Expect instabilities.")
            final_time = t_traj[-1] - 6 * 60 * 60
            logger.warning(
                "Removed last 6 hours of inspiral. New evolution time: %s years",
                final_time / YRSID_SI,
            )
        else:
            logger.info("Body is not plunging, Fisher should be stable.")
            final_time = self.T * YRSID_SI
        return final_time

    def Fisher_Stability(self):
        """Search per-parameter finite-difference steps that stabilize Gamma_ii."""
        if not self.use_gpu:
            xp = np
        else:
            xp = cp
        logger.info("calculating stable deltas...")

        # 1PAT1R modification: warn if chi1 is outside the valid range for
        # Trajectory1PAT1R, which only supports |chi1| <= 0.2.
        if self._is_circular and abs(self.wave_params.get("a", 0.0)) > 0.2:  # 1PAT1R modification
            logger.warning(  # 1PAT1R modification
                "chi1 = %s is outside the valid range [-0.2, 0.2] for "  # 1PAT1R modification
                "Trajectory1PAT1R. The trajectory will raise a ValueError.",  # 1PAT1R modification
                self.wave_params.get("a"),  # 1PAT1R modification
            )  # 1PAT1R modification

        Ndelta = self.Ndelta
        deltas = {}
        relerr_min = {}

        for param_name in self.param_names:

            try:
                delta_init = self.delta_range[param_name]

            except KeyError:

                if self.wave_params.get(param_name, None) == 0.0:
                    # 1PAT1R modification: when chi2=0 use an absolute step-size
                    # grid rather than a fractional one so the search is not degenerate.
                    if param_name == "chi2":  # 1PAT1R modification
                        delta_init = np.geomspace(1e-2, 1e-7, Ndelta)  # 1PAT1R modification
                    else:
                        delta_init = np.geomspace(1e-4, 1e-9, Ndelta)

                elif param_name in {"m1", "m2"}:
                    delta_init = np.geomspace(
                        1e-4 * self.wave_params[param_name],
                        1e-9 * self.wave_params[param_name],
                        Ndelta,
                    )
                elif param_name in {"a", "p0", "e0", "xI0", "chi2"}:
                    # 1PAT1R modification: chi2 shares the same fractional delta
                    # grid as other dimensionless orbital/spin parameters.
                    delta_init = np.geomspace(
                        1e-4 * self.wave_params[param_name],
                        1e-9 * self.wave_params[param_name],
                        Ndelta,
                    )
                else:
                    delta_init = np.geomspace(
                        1e-1 * self.wave_params[param_name],
                        1e-6 * self.wave_params[param_name],
                        Ndelta,
                    )

            Gamma = []

            relerr_flag = False
            for delta_k in delta_init:

                if param_name in self.minmax:
                    if self.wave_params.get(param_name, 0.5) <= self.minmax[param_name][0]:
                        kind = "forward"
                    elif self.wave_params.get(param_name, 0.5) > self.minmax[param_name][1]:
                        kind = "backward"
                    else:
                        kind = self.kind
                else:
                    kind = self.kind

                if param_name == "dist":
                    del_k = xp.asarray(
                        self.derivative(
                            *self.wave_params_list,
                            param_to_vary=param_name,
                            delta=delta_k,
                            kind=kind,
                            **self.waveform_derivative_kwargs,
                        )
                    )
                    relerr_flag = True
                    deltas["dist"] = 0.0
                    relerr_min["dist"] = 0.0
                    break

                if (param_name in ["Phi_phi0", "Phi_theta0", "Phi_r0"]) & (
                    self.deriv_type == "stable"
                ):
                    del_k = xp.asarray(
                        self.derivative(
                            *self.wave_params_list,
                            param_to_vary=param_name,
                            delta=delta_k,
                            kind=kind,
                            **self.waveform_derivative_kwargs,
                        )
                    )
                    relerr_flag = True
                    deltas[param_name] = 0.0
                    relerr_min[param_name] = 0.0
                    break

                if (
                    (param_name in ["qS", "phiS", "qK", "phiK"])
                    & (self.deriv_type == "stable")
                    & (self.has_ResponseWrapper)
                ):
                    if len(delta_init) == 1:
                        relerr_flag = True
                        deltas[param_name] = delta_k
                        break
                    deltas_grid = self._deltas(delta_k, self.order, kind=kind)
                    Rh_temp = xp.zeros(
                        (len(deltas_grid), len(self.waveform), len(self.waveform[0])),
                        dtype=xp.complex128,
                    )
                    for dd, delt in enumerate(deltas_grid):
                        parameters_in = self.waveform_derivative_kwargs[
                            "parameters"
                        ].copy()
                        parameters_in[param_name] += float(delt)
                        parameters_in_list = list(parameters_in.values())
                        Rh_temp[dd] = xp.asarray(
                            self.waveform_generator(
                                *parameters_in_list, **self.current_waveform_kwargs
                            )
                        )
                    del_k = self._stencil(
                        Rh_temp, delta=delta_k, order=self.order, kind=kind
                    )

                else:
                    deltas[param_name] = delta_k
                    del_k = xp.asarray(
                        self.derivative(
                            *self.wave_params_list,
                            param_to_vary=param_name,
                            delta=delta_k,
                            kind=kind,
                            **self.waveform_derivative_kwargs,
                        )
                    )

                    if len(delta_init) == 1:
                        relerr_flag = True

                if del_k.ndim == 1:
                    del_k = xp.asarray([del_k.real, -del_k.imag])

                Gammai = inner_product(
                    del_k,
                    del_k,
                    self.PSD_funcs,
                    self.dt,
                    window=self.window,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    freq_mask=self.freq_mask,
                    use_gpu=self.use_gpu,
                )
                logger.debug("Gamma_ii for %s: %s", param_name, Gammai)
                if np.isnan(Gammai):
                    Gamma.append(0.0)
                    logger.warning(
                        "NaN type encountered during "
                        "Fisher calculation! Replacing with 0.0."
                    )
                else:
                    Gamma.append(Gammai)

            if not relerr_flag:
                if self.use_gpu:
                    Gamma = xp.asnumpy(xp.array(Gamma))
                else:
                    Gamma = xp.array(Gamma)

                if (Gamma[1:] == 0.0).all():
                    relerr = list(np.ones(len(Gamma) - 1))
                else:
                    relerr = []
                    for m in range(1, len(Gamma)):
                        if Gamma[m - 1] == 0.0:
                            relerr.append(1.0)
                        else:
                            relerr.append(np.abs(Gamma[m] - Gamma[m - 1]) / Gamma[m])

                logger.debug(relerr)
                relerr_min_i = relerr.index(min(relerr))
                logger.debug(relerr_min_i)

                if relerr[relerr_min_i] >= 0.01:
                    logger.warning(
                        "minimum relative error is greater "
                        "than 1%% for %s. Fisher may be unstable!",
                        param_name,
                    )

                deltas_min_i = relerr_min_i + 1
                deltas[param_name] = delta_init[deltas_min_i].item()
                relerr_min[param_name] = relerr[relerr_min_i]

                if self.stability_plot:
                    if self.filename is not None:
                        if self.suffix is not None:
                            StabilityPlot(
                                delta_init,
                                Gamma,
                                stable_index=deltas_min_i,
                                param_name=param_name,
                                filename=os.path.join(
                                    self.filename,
                                    f"stability_{self.suffix}_{param_name}.png",
                                ),
                            )
                        else:
                            StabilityPlot(
                                delta_init,
                                Gamma,
                                stable_index=deltas_min_i,
                                param_name=param_name,
                                filename=os.path.join(
                                    self.filename,
                                    f"stability_{param_name}.png",
                                ),
                            )
                    else:
                        StabilityPlot(
                            delta_init,
                            Gamma,
                            stable_index=deltas_min_i,
                            param_name=param_name,
                        )

        logger.debug("stable deltas: %s", deltas)
        self.deltas = deltas
        self.save_deltas()

    def save_deltas(self):
        if self.filename is not None:
            if self.suffix is not None:
                with open(
                    f"{self.filename}/stable_deltas_{self.suffix}.txt",
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as file:
                    file.write(str(self.deltas))
            else:
                with open(
                    f"{self.filename}/stable_deltas.txt",
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as file:
                    file.write(str(self.deltas))

    def FisherCalc(self):
        """Assemble the Fisher matrix using numerically differentiated waveforms."""
        if self.use_gpu:
            xp = cp
        else:
            xp = np

        logger.info("calculating Fisher matrix...")

        Fisher = np.zeros((self.npar, self.npar), dtype=np.float64)
        dtv = []
        for i, param_name in enumerate(self.param_names):

            if param_name in self.minmax:
                if self.wave_params.get(param_name, 0.5) <= self.minmax[param_name][0]:
                    kind = "forward"
                elif self.wave_params.get(param_name, 0.5) > self.minmax[param_name][1]:
                    kind = "backward"
                else:
                    kind = self.kind
            else:
                kind = self.kind

            if (
                (param_name in ["qS", "phiS", "qK", "phiK"])
                & (self.deriv_type == "stable")
                & (self.has_ResponseWrapper)
            ):
                deltas_grid = self._deltas(
                    self.deltas[param_name], self.order, kind=kind
                )
                Rh_temp = xp.zeros(
                    (len(deltas_grid), len(self.waveform), len(self.waveform[0])),
                    dtype=self.waveform.dtype,
                )
                for dd, delt in enumerate(deltas_grid):
                    parameters_in = self.waveform_derivative_kwargs["parameters"].copy()
                    parameters_in[param_name] += float(delt)
                    parameters_in_list = list(parameters_in.values())
                    Rh_temp[dd] = xp.asarray(
                        self.waveform_generator(
                            *parameters_in_list, **self.current_waveform_kwargs
                        )
                    )
                dtv_i = self._stencil(
                    Rh_temp,
                    delta=self.deltas[param_name],
                    order=self.order,
                    kind=kind,
                )

            else:
                dtv_i = xp.asarray(
                    self.derivative(
                        *self.wave_params_list,
                        param_to_vary=param_name,
                        delta=self.deltas[param_name],
                        kind=kind,
                        **self.waveform_derivative_kwargs,
                    )
                )

            if dtv_i.ndim == 1:
                dtv_i = xp.asarray([dtv_i.real, -dtv_i.imag])

            dtv.append(dtv_i)

        logger.info("Finished derivatives")

        if self.save_derivatives:
            dtv_save = xp.asarray(dtv)
            if self.use_gpu:
                dtv_save = xp.asnumpy(dtv_save)
            if self.filename is not None:
                if self.suffix is not None:
                    with h5py.File(
                        f"{self.filename}/Fisher_{self.suffix}.h5", "w"
                    ) as f:
                        f.create_dataset("derivatives", data=dtv_save)
                else:
                    with h5py.File(f"{self.filename}/Fisher.h5", "w") as f:
                        f.create_dataset("derivatives", data=dtv_save)

        for i in range(self.npar):
            for j in range(i, self.npar):
                if self.use_gpu:
                    Fisher[i, j] = np.float64(
                        xp.asnumpy(
                            inner_product(
                                dtv[i],
                                dtv[j],
                                self.PSD_funcs,
                                self.dt,
                                window=self.window,
                                fmin=self.fmin,
                                fmax=self.fmax,
                                freq_mask=self.freq_mask,
                                use_gpu=self.use_gpu,
                            ).real
                        )
                    )
                else:
                    Fisher[i, j] = np.float64(
                        (
                            inner_product(
                                dtv[i],
                                dtv[j],
                                self.PSD_funcs,
                                self.dt,
                                window=self.window,
                                fmin=self.fmin,
                                fmax=self.fmax,
                                freq_mask=self.freq_mask,
                                use_gpu=self.use_gpu,
                            ).real
                        )
                    )

                Fisher[j, i] = Fisher[i, j]

        diag_elements = np.diag(Fisher)

        if 0 in diag_elements:
            logger.critical("Nasty. We have a degeneracy. Can't measure a parameter")
            degen_index = np.argwhere(diag_elements == 0)[0][0]
            Fisher[degen_index, degen_index] = 1.0

        if (np.linalg.eigvals(Fisher) < 0.0).any():
            logger.critical(
                "Calculated Fisher is not positive "
                "semi-definite. "
                "Try lowering inspiral error tolerance "
                "or increasing the derivative order."
            )
        else:
            logger.info("Calculated Fisher is *atleast* positive-definite.")

        if self.filename is None:
            pass
        else:
            if self.save_derivatives:
                mode = "a"
            else:
                mode = "w"
            if self.suffix is not None:
                with h5py.File(f"{self.filename}/Fisher_{self.suffix}.h5", mode) as f:
                    f.create_dataset("Fisher", data=Fisher)
            else:
                with h5py.File(f"{self.filename}/Fisher.h5", mode) as f:
                    f.create_dataset("Fisher", data=Fisher)

        if self.return_derivatives is True:
            return dtv, Fisher
        return Fisher