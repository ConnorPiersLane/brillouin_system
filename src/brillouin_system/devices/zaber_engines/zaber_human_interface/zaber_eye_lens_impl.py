import inspect
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from zaber_motion import Library, Units
from zaber_motion.ascii import Connection
from zaber_motion.ascii.axis import Axis

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_position_log import ZaberPositionLog


@dataclass(frozen=True, slots=True)
class PVTPoint:
    position_um: float
    velocity_um_s: float
    dt_s: float


@dataclass(frozen=True, slots=True)
class PVTScanModel:
    start_um: float
    signed_distance_um: float
    signed_velocity_um_s: float
    dwell_s: float
    accel_s: float
    cruise_s: float
    decel_s: float
    settle_s: float
    t_launch_perf: float
    launch_window_s: float
    tau_bias_s: float = 0.0

    @property
    def direction(self) -> float:
        return 1.0 if self.signed_distance_um >= 0 else -1.0

    @property
    def abs_velocity_um_s(self) -> float:
        return abs(self.signed_velocity_um_s)

    @property
    def abs_distance_um(self) -> float:
        return abs(self.signed_distance_um)

    @property
    def t_motion_start_perf(self) -> float:
        return self.t_launch_perf + self.dwell_s + self.tau_bias_s

    @property
    def t_cruise_start_perf(self) -> float:
        return self.t_motion_start_perf + self.accel_s

    @property
    def t_motion_end_perf(self) -> float:
        return self.t_motion_start_perf + self.accel_s + self.cruise_s + self.decel_s + self.settle_s

    def _a(self) -> float:
        if self.accel_s <= 0:
            return 0.0
        return self.abs_velocity_um_s / self.accel_s

    def _decel(self) -> float:
        if self.decel_s <= 0:
            return 0.0
        return self.abs_velocity_um_s / self.decel_s

    def position_at_rel(self, t_rel_s: float) -> float:
        """Return modeled position in um at time relative to motion start."""
        d = self.direction
        v = self.abs_velocity_um_s
        a = self._a()
        ad = self._decel()
        t = float(t_rel_s)

        if t <= 0:
            return self.start_um

        # accel phase
        if self.accel_s > 0 and t < self.accel_s:
            ds = 0.5 * a * t * t
            return self.start_um + d * ds

        t_after_accel = max(0.0, t - self.accel_s)
        d_accel = 0.5 * v * self.accel_s
        z_after_accel = self.start_um + d * d_accel

        # cruise phase
        if t_after_accel < self.cruise_s:
            ds = d_accel + v * t_after_accel
            return self.start_um + d * ds

        t_after_cruise = max(0.0, t_after_accel - self.cruise_s)
        d_cruise = v * self.cruise_s
        z_after_cruise = self.start_um + d * (d_accel + d_cruise)

        # decel phase
        if self.decel_s > 0 and t_after_cruise < self.decel_s:
            td = t_after_cruise
            ds = d_accel + d_cruise + v * td - 0.5 * ad * td * td
            return self.start_um + d * ds

        # settle / clamp at end position
        return self.start_um + self.signed_distance_um

    def position_at_perf(self, t_perf: float) -> float:
        return self.position_at_rel(float(t_perf) - self.t_motion_start_perf)


@dataclass(frozen=True, slots=True)
class PVTExecution:
    points: tuple[PVTPoint, ...]
    model: PVTScanModel
    launch_t0_perf: float
    launch_t1_perf: float
    pvt_id: int = 1

    @property
    def launch_t_mid_perf(self) -> float:
        return 0.5 * (self.launch_t0_perf + self.launch_t1_perf)

    @property
    def launch_uncertainty_s(self) -> float:
        return self.launch_t1_perf - self.launch_t0_perf


class ZaberEyeLens:
    def __init__(self, port: str = "COM5", axis_index: int = 1):
        Library.enable_device_db_store()
        self.connection = Connection.open_serial_port(port)
        devices = self.connection.detect_devices()
        if not devices:
            raise RuntimeError("No Zaber devices found.")
        self.device = devices[0]
        self.axis: Axis = self.device.get_axis(axis_index)

        # Guard state
        self._slew_guard_active = False
        self._slew_guard_thread = None
        self._slew_guard_start_pos = None
        self._slew_guard_max_dist = None  # in um

        self._log_active = False
        self._log_thread = None
        self._log_lock = threading.Lock()
        self._log_t = []
        self._log_z = []

        self._active_pvt: Optional[PVTExecution] = None

        self.home()
        self.move_init()

    def home(self):
        self.axis.home()
        self.axis.wait_until_idle()

    def move_init(self):
        self.move_abs(0)

    def move_abs(self, position_um: float):
        self.axis.move_absolute(float(position_um), Units.LENGTH_MICROMETRES)
        self.axis.wait_until_idle()

    def move_rel(self, delta_um: float):
        self.axis.move_relative(float(delta_um), Units.LENGTH_MICROMETRES)
        self.axis.wait_until_idle()

    def get_position(self) -> float:
        return float(self.axis.get_position(Units.LENGTH_MICROMETRES))

    def start_slewing(self, speed_um_per_s: float):
        self.axis.move_velocity(
            float(speed_um_per_s),
            Units.VELOCITY_MICROMETRES_PER_SECOND,
        )

    def stop_slewing(self):
        self._slew_guard_active = False
        self.axis.stop()
        self.axis.wait_until_idle()

    def start_slewing_guarded(self, speed_um_per_s: float, max_distance_um: float):
        if max_distance_um <= 0:
            return

        self._slew_guard_start_pos = self.get_position()
        self._slew_guard_max_dist = float(abs(max_distance_um))
        self._slew_guard_active = True
        self.start_slewing(speed_um_per_s)

        def _guard_loop():
            try:
                while self._slew_guard_active:
                    pos = self.get_position()
                    travelled = abs(pos - self._slew_guard_start_pos)
                    if travelled >= self._slew_guard_max_dist:
                        self.axis.stop()
                        self.axis.wait_until_idle()
                        break
                    time.sleep(0.01)
            finally:
                self._slew_guard_active = False

        t = threading.Thread(target=_guard_loop, daemon=True)
        self._slew_guard_thread = t
        t.start()

    def start_position_log(self, *, poll_s: float = 0.016, alpha: float = 0.5) -> None:
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be 0<=alpha<=1 but is {alpha}")
        if self._log_active:
            raise RuntimeError("Position logging already running")
        if poll_s <= 0.0:
            raise ValueError(f"poll_s must be > 0 but is {poll_s}")

        with self._log_lock:
            self._log_t = []
            self._log_z = []

        self._log_active = True

        def _loop():
            next_t = time.perf_counter()
            try:
                while self._log_active:
                    now = time.perf_counter()
                    if now < next_t:
                        time.sleep(next_t - now)
                        continue
                    next_t += poll_s

                    t0 = time.perf_counter()
                    z = self.get_position()
                    t1 = time.perf_counter()
                    t = t0 + alpha * (t1 - t0)

                    with self._log_lock:
                        self._log_t.append(t)
                        self._log_z.append(z)
            finally:
                self._log_active = False

        th = threading.Thread(target=_loop, name="ZaberPosLog", daemon=True)
        self._log_thread = th
        th.start()

    def stop_position_log(self, *, join_timeout_s: float = 2.0) -> ZaberPositionLog:
        if not self._log_active:
            return ZaberPositionLog(
                t_perf=np.empty(0, dtype=np.float64),
                z_um=np.empty(0, dtype=np.float64),
            )

        self._log_active = False
        th = self._log_thread
        if th is not None:
            th.join(timeout=float(join_timeout_s))
            if th.is_alive():
                raise TimeoutError("Zaber position log thread did not stop")

        with self._log_lock:
            t = np.asarray(self._log_t, dtype=np.float64)
            z = np.asarray(self._log_z, dtype=np.float64)

        if t.size >= 2:
            keep = np.concatenate(([True], np.diff(t) > 0))
            t = t[keep]
            z = z[keep]
        return ZaberPositionLog(t_perf=t, z_um=z)

    @contextmanager
    def slewing_with_log(self, speed_um_per_s: float, max_distance_um: float, *, poll_s: float = 0.016, alpha: float = 0.5):
        self.start_position_log(poll_s=poll_s, alpha=alpha)
        try:
            self.start_slewing_guarded(speed_um_per_s, max_distance_um)
            yield
        finally:
            try:
                self.stop_slewing()
            except Exception:
                pass
            _ = self.stop_position_log()

    # ------------------------------------------------------------------
    # PVT helpers
    # ------------------------------------------------------------------
    def build_linear_scan_points(
        self,
        *,
        start_um: Optional[float] = None,
        speed_um_s: float,
        distance_um: float,
        dwell_s: float = 0.050,
        accel_s: float = 0.050,
        decel_s: Optional[float] = None,
        settle_s: float = 0.0,
    ) -> tuple[PVTPoint, ...]:
        """
        Build a simple 1-axis PVT trajectory:
        dwell -> accelerate -> cruise -> decelerate -> settle.

        The resulting points are device-side timing points. Use the returned
        points together with start_linear_scan_pvt() to link the trajectory to
        perf_counter() via the launch-time bracket.
        """
        if speed_um_s == 0:
            raise ValueError("speed_um_s must be non-zero")
        if distance_um == 0:
            raise ValueError("distance_um must be non-zero")
        if dwell_s < 0 or accel_s < 0 or settle_s < 0:
            raise ValueError("dwell_s, accel_s and settle_s must be >= 0")

        if start_um is None:
            start_um = self.get_position()
        start_um = float(start_um)

        signed_distance_um = float(distance_um)
        direction = 1.0 if signed_distance_um >= 0 else -1.0
        v_abs = abs(float(speed_um_s))
        v = direction * v_abs
        decel_s = float(accel_s if decel_s is None else decel_s)
        if decel_s < 0:
            raise ValueError("decel_s must be >= 0")

        d_acc = 0.5 * v_abs * accel_s if accel_s > 0 else 0.0
        d_dec = 0.5 * v_abs * decel_s if decel_s > 0 else 0.0
        d_flat = abs(signed_distance_um) - d_acc - d_dec
        if d_flat < -1e-9:
            raise ValueError(
                "distance_um is too short for the requested accel/decel profile; "
                "reduce accel_s/decel_s or increase distance_um"
            )
        d_flat = max(0.0, d_flat)
        t_flat = d_flat / v_abs if v_abs > 0 else 0.0

        points: list[PVTPoint] = []
        z = start_um

        if dwell_s > 0:
            points.append(PVTPoint(position_um=z, velocity_um_s=0.0, dt_s=float(dwell_s)))

        if accel_s > 0:
            z += direction * d_acc
            points.append(PVTPoint(position_um=z, velocity_um_s=v, dt_s=float(accel_s)))

        if t_flat > 0:
            z += direction * d_flat
            points.append(PVTPoint(position_um=z, velocity_um_s=v, dt_s=float(t_flat)))

        if decel_s > 0:
            z = start_um + signed_distance_um
            points.append(PVTPoint(position_um=z, velocity_um_s=0.0, dt_s=float(decel_s)))
        else:
            z = start_um + signed_distance_um
            if not points or abs(points[-1].position_um - z) > 1e-12 or abs(points[-1].velocity_um_s) > 1e-12:
                points.append(PVTPoint(position_um=z, velocity_um_s=v, dt_s=0.0))

        if settle_s > 0:
            points.append(PVTPoint(position_um=z, velocity_um_s=0.0, dt_s=float(settle_s)))

        return tuple(points)

    def build_linear_scan_model(
        self,
        *,
        start_um: float,
        speed_um_s: float,
        distance_um: float,
        t_launch_perf: float,
        launch_window_s: float,
        dwell_s: float = 0.050,
        accel_s: float = 0.050,
        decel_s: Optional[float] = None,
        settle_s: float = 0.0,
        tau_bias_s: float = 0.0,
    ) -> PVTScanModel:
        if speed_um_s == 0:
            raise ValueError("speed_um_s must be non-zero")
        if distance_um == 0:
            raise ValueError("distance_um must be non-zero")

        decel_s = float(accel_s if decel_s is None else decel_s)
        if dwell_s < 0 or accel_s < 0 or decel_s < 0 or settle_s < 0:
            raise ValueError("all timing values must be >= 0")

        v_abs = abs(float(speed_um_s))
        d_acc = 0.5 * v_abs * float(accel_s)
        d_dec = 0.5 * v_abs * float(decel_s)
        d_flat = abs(float(distance_um)) - d_acc - d_dec
        if d_flat < -1e-9:
            raise ValueError(
                "distance_um is too short for the requested accel/decel profile; "
                "reduce accel_s/decel_s or increase distance_um"
            )
        d_flat = max(0.0, d_flat)
        t_flat = d_flat / v_abs
        return PVTScanModel(
            start_um=float(start_um),
            signed_distance_um=float(distance_um),
            signed_velocity_um_s=float(np.sign(distance_um) * v_abs),
            dwell_s=float(dwell_s),
            accel_s=float(accel_s),
            cruise_s=float(t_flat),
            decel_s=float(decel_s),
            settle_s=float(settle_s),
            t_launch_perf=float(t_launch_perf),
            launch_window_s=float(launch_window_s),
            tau_bias_s=float(tau_bias_s),
        )




    def start_linear_scan_pvt(
            self,
            *,
            speed_um_s: float,
            distance_um: float,
            start_um: Optional[float] = None,
            dwell_s: float = 0.050,
            accel_s: float = 0.050,
            decel_s: Optional[float] = None,
            settle_s: float = 0.0,
            pvt_id: int = 1,
            clear_first: bool = True,
            tau_bias_s: float = 0.0,
    ) -> PVTExecution:

        if start_um is None:
            start_um = self.get_position()
        self.move_abs(start_um)

        points = self.build_linear_scan_points(
            start_um=start_um,
            speed_um_s=speed_um_s,
            distance_um=distance_um,
            dwell_s=dwell_s,
            accel_s=accel_s,
            decel_s=decel_s,
            settle_s=settle_s,
        )

        self.queue_pvt_points(points, pvt_id=pvt_id, clear_first=clear_first)

        t0 = time.perf_counter()
        self.start_pvt(pvt_id=pvt_id)
        t1 = time.perf_counter()

        model = self.build_linear_scan_model(
            start_um=start_um,
            speed_um_s=speed_um_s,
            distance_um=distance_um,
            t_launch_perf=0.5 * (t0 + t1),
            launch_window_s=t1 - t0,
            dwell_s=dwell_s,
            accel_s=accel_s,
            decel_s=decel_s,
            settle_s=settle_s,
            tau_bias_s=tau_bias_s,
        )
        execution = PVTExecution(
            points=points,
            model=model,
            launch_t0_perf=t0,
            launch_t1_perf=t1,
            pvt_id=int(pvt_id),
        )
        self._active_pvt = execution
        return execution


    def active_pvt(self) -> Optional[PVTExecution]:
        return self._active_pvt

    def position_from_active_pvt(self, t_perf: float) -> float:
        if self._active_pvt is None:
            raise RuntimeError("No active PVT execution is registered")
        return self._active_pvt.model.position_at_perf(t_perf)

    def wait_for_pvt(self, *, timeout_s: Optional[float] = None) -> Optional[PVTExecution]:
        if timeout_s is None:
            self.axis.wait_until_idle()
            return self._active_pvt

        deadline = time.perf_counter() + float(timeout_s)
        while time.perf_counter() < deadline:
            if self.axis.is_idle():
                return self._active_pvt
            time.sleep(0.002)
        raise TimeoutError("PVT motion did not finish before timeout")

    def stop_pvt(self):
        self.stop_slewing()
        self._active_pvt = None

    # ------------------------------------------------------------------
    # Runtime-robust PVT API wrappers
    # ------------------------------------------------------------------
    def queue_pvt_points(self, points: Iterable[PVTPoint], *, pvt_id: int = 1, clear_first: bool = True) -> Any:
        seq = self._resolve_pvt_sequence(pvt_id)
        if clear_first:
            self._clear_pvt_sequence(seq)
        for pt in points:
            self._append_pvt_point(seq, pt)
        return seq

    def start_pvt(self, *, pvt_id: int = 1) -> Any:
        seq = self._resolve_pvt_sequence(pvt_id)
        return self._call_first_available(
            seq,
            [
                ("call", (), {}),
                ("start", (), {}),
                ("run", (), {}),
                ("execute", (), {}),
                ("start_live", (), {}),
                ("begin", (), {}),
            ],
            what=f"start PVT sequence {pvt_id}",
        )

    def _resolve_pvt_sequence(self, pvt_id: int = 1) -> Any:
        candidates = [
            getattr(self.device, "pvt", None),
            getattr(self.axis, "pvt", None),
            getattr(getattr(self.axis, "device", None), "pvt", None),
        ]
        for cand in candidates:
            if cand is None:
                continue

            # If this already behaves like a sequence, use it directly.
            if self._looks_like_pvt_sequence(cand):
                return cand

            # Bound method or callable manager.
            try:
                maybe = cand(pvt_id) if callable(cand) else None
                if maybe is not None:
                    return maybe
            except TypeError:
                pass
            except Exception:
                pass

            # Common manager method names.
            for name in ("get_sequence", "sequence", "get", "get_pvt_sequence", "open_sequence"):
                fn = getattr(cand, name, None)
                if callable(fn):
                    try:
                        return fn(pvt_id)
                    except Exception:
                        continue

        raise RuntimeError(
            "Could not resolve a Zaber PVT sequence object from the current motion library API. "
            "Check your installed zaber-motion version and adapt _resolve_pvt_sequence()."
        )

    @staticmethod
    def _looks_like_pvt_sequence(obj: Any) -> bool:
        names = set(dir(obj))
        return bool({"append", "add_point", "start", "run", "execute", "clear", "erase"} & names)

    def _clear_pvt_sequence(self, seq: Any) -> None:
        try:
            self._call_first_available(
                seq,
                [
                    ("clear", (), {}),
                    ("erase", (), {}),
                    ("reset", (), {}),
                    ("clear_points", (), {}),
                ],
                what="clear PVT sequence",
            )
        except RuntimeError:
            # Some APIs auto-overwrite / do not expose clear. That's acceptable.
            pass

    def _append_pvt_point(self, seq: Any, pt: PVTPoint) -> Any:
        time_unit = getattr(Units, "TIME_SECONDS", None)

        attempts = [
            ("append", (pt.position_um, pt.velocity_um_s, pt.dt_s), {}),
            ("append_point", (pt.position_um, pt.velocity_um_s, pt.dt_s), {}),
            ("add", (pt.position_um, pt.velocity_um_s, pt.dt_s), {}),
            ("add_point", (pt.position_um, pt.velocity_um_s, pt.dt_s), {}),
            (
                "append",
                (
                    pt.position_um,
                    Units.LENGTH_MICROMETRES,
                    pt.velocity_um_s,
                    Units.VELOCITY_MICROMETRES_PER_SECOND,
                    pt.dt_s,
                    time_unit,
                ),
                {},
            ),
            (
                "append_point",
                (
                    pt.position_um,
                    Units.LENGTH_MICROMETRES,
                    pt.velocity_um_s,
                    Units.VELOCITY_MICROMETRES_PER_SECOND,
                    pt.dt_s,
                    time_unit,
                ),
                {},
            ),
            (
                "add_point",
                (),
                {
                    "position": pt.position_um,
                    "position_unit": Units.LENGTH_MICROMETRES,
                    "velocity": pt.velocity_um_s,
                    "velocity_unit": Units.VELOCITY_MICROMETRES_PER_SECOND,
                    "time": pt.dt_s,
                    "time_unit": time_unit,
                },
            ),
        ]

        # Drop attempts containing None units.
        attempts = [a for a in attempts if None not in a[1] and None not in a[2].values()]
        return self._call_first_available(seq, attempts, what="append PVT point")

    @staticmethod
    def _call_first_available(obj: Any, attempts: list[tuple[str, tuple[Any, ...], dict[str, Any]]], *, what: str) -> Any:
        errors: list[str] = []
        for name, args, kwargs in attempts:
            if name == "call":
                fn = obj if callable(obj) else None
            else:
                fn = getattr(obj, name, None)
            if not callable(fn):
                continue
            try:
                return fn(*args, **kwargs)
            except TypeError as exc:
                errors.append(f"{name}{args!r}{kwargs!r}: {exc}")
                continue
            except Exception as exc:
                # Runtime errors here are likely meaningful device/API errors; preserve them.
                raise RuntimeError(f"Failed to {what} via '{name}': {exc}") from exc
        details = "\n".join(errors[-4:]) if errors else "no compatible methods found"
        raise RuntimeError(f"Unable to {what}; attempts failed: {details}")

    def close(self):
        self.connection.close()

