"""Furuta pendulum in MuJoCo with the same interface as ``envs.Furuta``.

State (phi, phi_dot, alpha, alpha_dot) with alpha = 0 upright. Input = motor voltage;
the DC-motor model tau = kt (V - km phi_dot) / Rm is applied as joint torque every
physics substep. Features / deltas / Jacobian assembly are inherited from ``Furuta``,
so every TabPFN component works unchanged; only ``step`` and ``dynamics`` differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout

from tabpfn_control.envs import Furuta, wrap_angle




# Clickable plates on the table. Each is its own body so the viewer's picker returns a distinct id;
# none has a joint, so they add no degrees of freedom and the dynamics are untouched.
BUTTONS = [("switch", "switch plants", "0.92 0.41 0.20"),
           ("flip", "reverse motor", "0.80 0.24 0.20"),
           ("randomise", "random pendulum", "0.92 0.60 0.16"),
           ("reset", "drop to hanging", "0.35 0.55 0.85"),
           ("dither", "dither on/off", "0.45 0.70 0.45"),
           ("pushl", "push left", "0.62 0.62 0.66"),
           ("pushr", "push right", "0.62 0.62 0.66")]


def furuta_xml(Lr: float, Lp: float, mr: float, mp: float, Dr: float, Dp: float, timestep: float = 0.002,
               buttons: bool = False, m_tip: float = 0.0) -> str:
    """MJCF for a rotary inverted pendulum with rod arm (length Lr, mass mr) and rod pendulum (Lp, mp).

    Pendulum hinge about the arm's radial axis; qpos[1] = 0 is upright. Motor gain/bias set from Python.
    Everything below the rig - table, legs, floor, lights, and the optional button plates - is scenery:
    no mass, no contacts, no degrees of freedom, so it cannot touch a single number in the results.
    """
    btn = "\n".join(
        f'''    <body name="btn_{n}" pos="{-0.375 + 0.125 * i:.3f} -0.30 0.013">
      <geom name="btn_{n}" type="box" size="0.057 0.030 0.009" rgba="{c} 1" material="plastic"
            contype="0" conaffinity="0" mass="0"/>
    </body>''' for i, (n, _, c) in enumerate(BUTTONS)) if buttons else ""
    top = Lp + 0.12                                        # pendulum pivot height above the table
    bob_r = 0.020 * (max(m_tip, 1e-12) / 0.05) ** (1 / 3) if m_tip > 0 else 0.020
    return f"""<mujoco model="furuta">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <headlight ambient="0.20 0.21 0.24" diffuse="0.22 0.22 0.24" specular="0.05 0.05 0.05"/>
    <quality shadowsize="8192" offsamples="16"/>
    <global offwidth="1600" offheight="1200"/>
    <map znear="0.02" zfar="40" force="0.004" shadowclip="1.5" fogstart="0.7" fogend="1.5"/>
    <scale forcewidth="0.035" contactwidth="0.05" contactheight="0.02" framelength="0.18" framewidth="0.015"/>
    <rgba force="0.95 0.45 0.10 1" com="0.9 0.9 0.9 0.6" fog="0.066 0.086 0.128 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.13 0.17 0.25" rgb2="0.012 0.016 0.028" width="800" height="800"/>
    <texture name="floortex" type="2d" builtin="checker" rgb1="0.10 0.115 0.14" rgb2="0.10 0.115 0.14" mark="edge" markrgb="0.22 0.25 0.31" width="512" height="512"/>
    <material name="floor" texture="floortex" texrepeat="24 24" reflectance="0.22" shininess="0.6" specular="0.3"/>
    <texture name="toptex" type="2d" builtin="flat" rgb1="0.23 0.245 0.275" rgb2="0.23 0.245 0.275" width="256" height="256"/>
    <material name="benchtop" texture="toptex" reflectance="0.10" shininess="0.35" specular="0.25"/>
    <material name="steel" rgba="0.70 0.72 0.76 1" reflectance="0.35" shininess="0.9" specular="0.8"/>
    <material name="dark" rgba="0.10 0.11 0.13 1" reflectance="0.15" shininess="0.6" specular="0.4"/>
    <material name="rod" reflectance="0.1" shininess="0.9" specular="0.7"/>
    <material name="gloss" reflectance="0.2" shininess="1.0" specular="1.0"/>
    <material name="led" emission="0.75" shininess="1.0" specular="0.5"/>
    <material name="plastic" reflectance="0.05" shininess="0.4" specular="0.3"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.7 -0.8 1.9" dir="-0.35 0.42 -1" directional="false" castshadow="true"
           diffuse="0.75 0.74 0.70" specular="0.30 0.30 0.30"/>
    <light name="fill" pos="-1.0 0.9 1.3" dir="0.55 -0.5 -1" directional="false" castshadow="false"
           diffuse="0.22 0.24 0.30"/>
    <light name="rim" pos="0.2 1.4 1.5" dir="-0.1 -0.75 -0.55" directional="false" castshadow="false"
           diffuse="0.55 0.60 0.72" specular="0.6 0.6 0.7"/>
    <geom name="floor" type="plane" size="6 6 0.05" pos="0 0 -0.62" material="floor" contype="0" conaffinity="0"/>
    <geom name="tabletop" type="box" size="0.54 0.42 0.014" pos="0 -0.06 -0.014" material="benchtop" contype="0" conaffinity="0"/>
    <geom name="taberim" type="box" size="0.55 0.43 0.004" pos="0 -0.06 -0.030" material="dark" contype="0" conaffinity="0"/>
    <geom name="leg1" type="cylinder" size="0.018 0.29" pos="-0.48 -0.38 -0.323" material="dark" contype="0" conaffinity="0"/>
    <geom name="leg2" type="cylinder" size="0.018 0.29" pos="0.48 -0.38 -0.323" material="dark" contype="0" conaffinity="0"/>
    <geom name="leg3" type="cylinder" size="0.018 0.29" pos="-0.48 0.30 -0.323" material="dark" contype="0" conaffinity="0"/>
    <geom name="leg4" type="cylinder" size="0.018 0.29" pos="0.48 0.30 -0.323" material="dark" contype="0" conaffinity="0"/>
    <geom name="plinth" type="cylinder" size="0.085 0.009" pos="0 0 0.009" material="dark" contype="0" conaffinity="0"/>
    <geom name="base" type="cylinder" size="0.032 {Lp / 2 + 0.04}" pos="0 0 {Lp / 2 + 0.05}" material="steel" contype="0" conaffinity="0"/>
    <geom name="motorcase" type="cylinder" size="0.055 0.032" pos="0 0 {top - 0.05:.4f}" material="dark" contype="0" conaffinity="0"/>
    <geom name="motorcap" type="cylinder" size="0.057 0.004" pos="0 0 {top - 0.014:.4f}" material="steel" contype="0" conaffinity="0"/>
    <geom name="polarity" type="box" size="0.0045 0.011 0.016" pos="0.0555 0 {top - 0.046:.4f}" rgba="0.30 0.72 0.38 1" material="led" contype="0" conaffinity="0"/>
{btn}
    <body name="arm" pos="0 0 {top}">
      <joint name="arm" type="hinge" axis="0 0 1" damping="{Dr}"/>
      <geom name="armrod" type="capsule" fromto="0 0 0 {Lr} 0 0" size="0.006" mass="{mr}" material="steel" contype="0" conaffinity="0"/>
      <geom name="armhub" type="cylinder" size="0.017 0.010" pos="0 0 0" material="dark" mass="0" contype="0" conaffinity="0"/>
      <body name="pendulum" pos="{Lr} 0 0">
        <joint name="pend" type="hinge" axis="1 0 0" damping="{Dp}"/>
        <geom name="pendrod" type="capsule" fromto="0 0 0 0 0 {Lp}" size="0.005" mass="{mp}" rgba="0.30 0.55 0.95 1" material="rod" contype="0" conaffinity="0"/>
        <geom name="hinge" type="cylinder" size="0.011 0.014" fromto="-0.014 0 0 0.014 0 0" material="steel" mass="0" contype="0" conaffinity="0"/>
        <geom name="bob" type="sphere" size="{bob_r:.5f}" pos="0 0 {Lp}" mass="{m_tip}" rgba="0.98 0.66 0.16 1" material="gloss" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="motor" joint="arm" gear="1" ctrlrange="-10 10" dyntype="none" gaintype="fixed" biastype="affine" gainprm="0.05" biasprm="0 0 -0.01"/>
  </actuator>
</mujoco>"""


@dataclass
class MujocoFuruta(Furuta):
    """MuJoCo-backed Furuta. ``dt`` is the control period; physics runs at the XML timestep."""

    name: str = "furuta-mujoco"
    m_tip: float = 0.0             # point mass bolted to the end of the pendulum (0 = bare rod)
    arm_mass: float | None = None  # if set, scales the arm body's mass/inertia after loading
    perturb_torque: float = 0.0  # external torque on the pendulum joint for the next step (demo pushes)
    buttons: bool = False        # add the clickable scenery plates (demo only; they carry no dynamics)
    model: object = field(default=None, repr=False)
    data: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.model = mujoco.MjModel.from_xml_string(
            furuta_xml(self.Lr, self.Lp, self.mr, self.mp, self.Dr, self.Dp, buttons=self.buttons,
                       m_tip=self.m_tip))
        arm_id = self.model.body("arm").id
        if self.arm_mass is not None:
            scale = self.arm_mass / float(self.model.body_mass[arm_id])
            self.model.body_mass[arm_id] *= scale
            self.model.body_inertia[arm_id] *= scale
        act = self.model.actuator("motor").id
        self.model.actuator_gainprm[act, 0] = self.kt / self.Rm
        self.model.actuator_biasprm[act, :3] = [0.0, 0.0, -self.kt * self.km / self.Rm]
        self.model.actuator_ctrlrange[act] = [-self.max_voltage, self.max_voltage]
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(round(self.dt / self.model.opt.timestep))
        self._j_arm = self.model.joint("arm").id
        self._j_pend = self.model.joint("pend").id
        self._q_arm = self.model.jnt_qposadr[self._j_arm]
        self._q_pend = self.model.jnt_qposadr[self._j_pend]
        self._v_arm = self.model.jnt_dofadr[self._j_arm]
        self._v_pend = self.model.jnt_dofadr[self._j_pend]

    # -------------------------------------------------------------- state io
    def _read_state(self) -> np.ndarray:
        q, v = self.data.qpos, self.data.qvel
        return np.array([wrap_angle(q[self._q_arm]), v[self._v_arm], wrap_angle(q[self._q_pend]), v[self._v_pend]])

    def set_state(self, s: np.ndarray) -> None:
        s = np.asarray(s, dtype=float)
        self.data.qpos[self._q_arm], self.data.qvel[self._v_arm] = s[0], s[1]
        self.data.qpos[self._q_pend], self.data.qvel[self._v_pend] = s[2], s[3]
        mujoco.mj_forward(self.model, self.data)
        self.state = self._read_state()

    def reset(self, rng: np.random.Generator | None = None, hanging: bool = True) -> np.ndarray:
        rng = rng or np.random.default_rng()
        a0 = np.pi if hanging else 0.0
        mujoco.mj_resetData(self.model, self.data)
        self.set_state([0.0, 0.0, a0 + rng.normal(0, 0.05), rng.normal(0, 0.05)])
        return self.state.copy()

    # -------------------------------------------------------------- stepping
    def _apply(self, V: float) -> None:
        self.data.ctrl[0] = float(np.clip(V, -self.max_voltage, self.max_voltage))  # native actuator = DC motor
        self.data.qfrc_applied[self._v_pend] = self.perturb_torque

    def step(self, u: float | np.ndarray, capture=None, capture_every: int = 0) -> np.ndarray:
        """Advance one control period. `capture(qpos)`, if given, is called every `capture_every`
        physics substeps, which is how a recording samples motion faster than the controller runs:
        the control period is 50 ms but the integrator is at 2 ms, so there is plenty of real motion
        between decisions and no reason for a video to show only twenty frames a second."""
        V = float(np.asarray(u, dtype=float).reshape(()))
        for i in range(self.n_substeps):
            self._apply(V)
            mujoco.mj_step(self.model, self.data)
            if capture is not None and capture_every and (i + 1) % capture_every == 0:
                capture(self.data.qpos.copy())
        self.perturb_torque = 0.0
        self.data.qvel[self._v_arm] = np.clip(self.data.qvel[self._v_arm], -self.max_speed, self.max_speed)
        self.data.qvel[self._v_pend] = np.clip(self.data.qvel[self._v_pend], -self.max_speed, self.max_speed)
        self.state = self._read_state()
        return self.state.copy()

    def dynamics(self, s: np.ndarray, V: np.ndarray, damping: bool = True) -> np.ndarray:
        """Vectorised one-step map via mujoco.rollout (C, multithreaded): states (N,4), V (N,) -> (N,4)."""
        s = np.atleast_2d(np.asarray(s, dtype=float)); N = len(s)
        V = np.clip(np.broadcast_to(np.asarray(V, dtype=float), (N,)), -self.max_voltage, self.max_voltage)
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        nstate = mujoco.mj_stateSize(self.model, spec)
        init = np.zeros((N, nstate))
        init[:, 1 + self._q_arm] = s[:, 0]; init[:, 1 + self._q_pend] = s[:, 2]          # [time, qpos(2), qvel(2), act]
        init[:, 1 + self.model.nq + self._v_arm] = s[:, 1]; init[:, 1 + self.model.nq + self._v_pend] = s[:, 3]
        ctrl = np.repeat(V[:, None, None], self.n_substeps, axis=1)
        saved = np.zeros(nstate); mujoco.mj_getState(self.model, self.data, saved, spec)  # rollout uses data as workspace
        state, _ = mj_rollout.rollout(self.model, self.data, init, ctrl, nstep=self.n_substeps)
        mujoco.mj_setState(self.model, self.data, saved, spec); mujoco.mj_forward(self.model, self.data)
        last = state[:, -1, :]
        q = last[:, 1:1 + self.model.nq]; v = last[:, 1 + self.model.nq:1 + self.model.nq + self.model.nv]
        out = np.stack([wrap_angle(q[:, self._q_arm]), np.clip(v[:, self._v_arm], -self.max_speed, self.max_speed),
                        wrap_angle(q[:, self._q_pend]), np.clip(v[:, self._v_pend], -self.max_speed, self.max_speed)], axis=-1)
        return out if N > 1 else out[0]

    def energy(self, s: np.ndarray | None = None) -> float:
        if s is not None:
            self.set_state(s)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_energyPos(self.model, self.data); mujoco.mj_energyVel(self.model, self.data)
        return float(self.data.energy[0] + self.data.energy[1])

    def linearize_upright_discrete(self, eps: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
        """Discrete-time (Ad, Bd) of the control-period map at the upright, by central finite differences
        on the simulator (the reference for the TabPFN-Jacobian LQR)."""
        s0 = np.zeros(4)
        f0 = self.dynamics(s0, np.array(0.0))
        Ad = np.zeros((4, 4)); Bd = np.zeros((4, 1))
        for i in range(4):
            e = np.zeros(4); e[i] = eps
            Ad[:, i] = (self.dynamics(s0 + e, np.array(0.0)) - self.dynamics(s0 - e, np.array(0.0))) / (2 * eps)
        Bd[:, 0] = (self.dynamics(s0, np.array(eps)) - self.dynamics(s0, np.array(-eps))) / (2 * eps)
        return Ad, Bd

    def set_plant(self, pend_mass_scale: float = 1.0, motor_scale: float = 1.0, pend_damping_scale: float = 1.0,
                  arm_damping_scale: float = 1.0, arm_friction: float | None = None,
                  pend_friction: float | None = None) -> dict:
        """Change the physical plant at run time (for in-context adaptation experiments).

        Scales are applied to the CURRENT values, so calling it twice compounds. Returns what changed.
        """
        pend = self.model.body("pendulum").id
        act = self.model.actuator("motor").id
        self.model.body_mass[pend] *= pend_mass_scale
        self.model.body_inertia[pend] *= pend_mass_scale
        self.kt *= motor_scale
        self.km *= motor_scale
        self.model.actuator_gainprm[act, 0] = self.kt / self.Rm
        self.model.actuator_biasprm[act, 2] = -self.kt * self.km / self.Rm
        self.model.dof_damping[self._v_pend] *= pend_damping_scale
        self.model.dof_damping[self._v_arm] *= arm_damping_scale
        if arm_friction is not None:      # dry friction (stiction): small voltages stop moving the arm
            self.model.dof_frictionloss[self._v_arm] = arm_friction
        if pend_friction is not None:
            self.model.dof_frictionloss[self._v_pend] = pend_friction
        mujoco.mj_forward(self.model, self.data)
        self.restyle()
        return {"pend_mass": float(self.model.body_mass[pend]), "kt": self.kt,
                "pend_damping": float(self.model.dof_damping[self._v_pend]),
                "arm_damping": float(self.model.dof_damping[self._v_arm]),
                "arm_friction": float(self.model.dof_frictionloss[self._v_arm]),
                "pend_friction": float(self.model.dof_frictionloss[self._v_pend])}

    # geoms and bodies whose pose or inertia depends on the pendulum's length or tip mass
    _GEOM_DEPS = ("pendrod", "bob", "base", "motorcase", "motorcap", "polarity")
    _BODY_DEPS = ("arm", "pendulum")

    def set_geometry(self, Lp: float | None = None, m_tip: float | None = None) -> dict:
        """Re-dimension the pendulum on a model that is already loaded, and in a viewer already
        drawing it. Length and tip mass are compile-time quantities in MJCF, so the honest way to
        change them at run time is to compile the geometry you want and copy the numbers across:
        the live model then holds exactly what the compiler would have produced, inertia included,
        rather than a hand-derived approximation of it.
        """
        self.Lp = float(self.Lp if Lp is None else Lp)
        self.m_tip = float(self.m_tip if m_tip is None else m_tip)
        ref = mujoco.MjModel.from_xml_string(
            furuta_xml(self.Lr, self.Lp, self.mr, self.mp, self.Dr, self.Dp, buttons=self.buttons,
                       m_tip=self.m_tip))
        for n in self._GEOM_DEPS:
            i = self.model.geom(n).id
            self.model.geom_pos[i], self.model.geom_quat[i] = ref.geom_pos[i], ref.geom_quat[i]
            self.model.geom_size[i], self.model.geom_rbound[i] = ref.geom_size[i], ref.geom_rbound[i]
        for n in self._BODY_DEPS:
            i = self.model.body(n).id
            for arr in ("body_pos", "body_mass", "body_ipos", "body_iquat", "body_inertia",
                        "body_subtreemass"):
                getattr(self.model, arr)[i] = getattr(ref, arr)[i]
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return {"Lp": self.Lp, "m_tip": self.m_tip}

    def restyle(self) -> None:
        """Make the current plant legible at a glance. Appearance only: every geom touched here is
        massless or contact-free, so nothing it changes can reach the dynamics."""
        m = self.model
        reversed_motor = self.kt < 0
        # the LED on the motor housing is the polarity tell-tale; its material glows
        m.geom_rgba[m.geom("polarity").id] = ([1.00, 0.22, 0.20, 1.0] if reversed_motor
                                              else [0.25, 0.95, 0.45, 1.0])
        m.geom_rgba[m.geom("motorcase").id] = ([0.30, 0.08, 0.08, 1.0] if reversed_motor
                                               else [0.10, 0.11, 0.13, 1.0])
        # the rod darkens with extra damping (the bob's size is set by the geometry, not here)
        rel_damp = float(m.dof_damping[self._v_pend]) / max(self.Dp, 1e-9)
        t = float(np.clip((rel_damp - 1.0) / 2.0, 0.0, 1.0))
        m.geom_rgba[m.geom("pendrod").id] = [0.30 - 0.14 * t, 0.55 - 0.25 * t, 0.95 - 0.30 * t, 1.0]
        mujoco.mj_forward(self.model, self.data)

    def push(self, torque: float) -> None:
        """Apply an external torque on the pendulum joint during the next control step (demo perturbation)."""
        self.perturb_torque = float(torque)
