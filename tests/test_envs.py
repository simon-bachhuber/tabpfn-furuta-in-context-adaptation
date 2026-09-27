import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from tabpfn_control.envs import Furuta, make_env, wrap_angle  # noqa: E402
from tabpfn_control.history import HistoryEncoder  # noqa: E402
from tabpfn_control.lqr import discretize, dlqr  # noqa: E402


def test_hanging_stable_upright_unstable():
    env = make_env("furuta-mujoco")
    env.set_state([0, 0, np.pi - 0.05, 0])
    for _ in range(100):
        s = env.step(0.0)
    assert abs(wrap_angle(s[2])) > np.pi - 0.3
    env.set_state([0, 0, 0.05, 0])
    for _ in range(60):
        s = env.step(0.0)
    assert abs(s[2]) > 0.5


def test_energy_conserved_without_damping_or_motor():
    from tabpfn_control.mujoco_env import MujocoFuruta
    env = MujocoFuruta(km=0.0)
    env.model.dof_damping[:] = 0.0
    env.set_state([0.3, 2.0, 2.5, -3.0])
    E0 = env.energy()
    for _ in range(60):
        env.step(0.0)
    assert abs(env.energy() - E0) < 2e-3 * max(1.0, abs(E0))


def test_batched_dynamics_matches_step_and_restores_state():
    env = make_env("furuta-mujoco")
    rng = np.random.default_rng(0)
    S = np.stack([rng.uniform(-3, 3, 20), rng.uniform(-5, 5, 20), rng.uniform(-3, 3, 20), rng.uniform(-5, 5, 20)], -1)
    U = rng.uniform(-10, 10, 20)
    B = env.dynamics(S, U)
    for i in range(20):
        env.set_state(S[i])
        assert np.allclose(env.step(U[i]), B[i])
    s0 = env.reset(np.random.default_rng(0))
    ref = env.dynamics(s0, np.array(3.0))
    env.dynamics(S, U)                      # a planner-sized batch must not disturb the live state
    assert np.allclose(env.state, s0)
    assert np.allclose(env.step(3.0), ref)


def test_numpy_model_matches_mujoco_up_to_mirror():
    """The hand-derived Lagrangian model equals MuJoCo at matched parameters (mirrored angle)."""
    from tabpfn_control.mujoco_env import MujocoFuruta
    Lr, Lp, mp, mr_mj = 0.15, 0.30, 0.05, 0.04
    ref = Furuta(Lr=Lr, Lp=Lp, mp=mp, mr=12 * (mr_mj * Lr**2 / 3) / Lr**2, kt=0.1, km=0.1, Dr=0.003, Dp=0.001)
    env = MujocoFuruta(Lr=Lr, Lp=Lp, mp=mp, mr=mr_mj, kt=0.1, km=0.1, Dr=0.003, Dp=0.001)
    mirror = lambda s: np.array([s[0], s[1], -s[2], -s[3]])  # noqa: E731
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = np.array([rng.uniform(-3, 3), rng.uniform(-5, 5), rng.uniform(-3, 3), rng.uniform(-5, 5)])
        V = rng.uniform(-10, 10)
        n = ref.accelerations(s, np.array(V))
        env.set_state(mirror(s)); env.data.ctrl[0] = V; mujoco.mj_forward(env.model, env.data)
        assert abs(n[0] - env.data.qacc[0]) < 0.05 * abs(n[0]) + 1.0
        assert abs(n[1] + env.data.qacc[1]) < 0.05 * abs(n[1]) + 1.0


def test_finite_difference_linearisation_is_unstable_and_stabilisable():
    env = make_env("furuta-mujoco")
    Ad, Bd = env.linearize_upright_discrete()
    assert np.max(np.abs(np.linalg.eigvals(Ad))) > 1.0           # upright is unstable
    K = dlqr(Ad, Bd, np.diag([1, 0.1, 10, 0.1]), np.array([[0.1]]))
    assert np.all(np.abs(np.linalg.eigvals(Ad - Bd @ K)) < 1)    # and stabilisable
    assert discretize(*env.linearize_upright(), env.dt)[0].shape == (4, 4)


@pytest.mark.parametrize("kind,n", [("ols", 20), ("gain", 50), ("resp", 5), ("lags", 5), ("none", 0)])
def test_history_encoder_offline_matches_online(kind, n):
    """The table builder and the control loop must produce identical history features."""
    env = make_env("furuta-mujoco")
    enc = HistoryEncoder(env, kind, k=max(n, 1), window=max(n, 1))
    rng = np.random.default_rng(0)
    S, U, S2 = [], [], []
    s = env.reset(rng, hanging=True)
    for _ in range(60):
        u = float(rng.uniform(-10, 10)); s2 = env.step(u)
        S.append(s.copy()); U.append(u); S2.append(s2.copy()); s = s2
    S, U, S2 = np.array(S), np.array(U), np.array(S2)
    H, ok = enc.encode_burst(S, U, S2)
    assert H.shape[1] == enc.n_features and ok[-1]
    online = enc.encode([(S[i], U[i], S2[i]) for i in range(len(U) - 1)])
    assert np.allclose(H[-1], online)


def test_identified_gain_tracks_the_true_motor_polarity():
    """The statistic the policy reads must actually carry the plant's control direction."""
    from tabpfn_control.policy import POLARITY_COL
    env = make_env("furuta-mujoco")
    enc = HistoryEncoder(env, "ols", window=40)
    signs = []
    for motor in (1.0, -1.0):
        e = make_env("furuta-mujoco"); e.set_plant(motor_scale=motor)
        rng = np.random.default_rng(0)
        past, s = [], e.reset(rng, hanging=False)
        for _ in range(40):
            u = float(rng.uniform(-8, 8)); s2 = e.step(u)
            past.append((s.copy(), u, s2.copy())); s = s2
            if abs(wrap_angle(s[2])) > 1.2:
                s = e.reset(rng, hanging=False)
        signs.append(np.sign(enc.encode(past)[POLARITY_COL]))
    assert signs[0] == -signs[1] and signs[0] != 0


def test_runtime_geometry_equals_a_freshly_compiled_model():
    """Changing length or tip mass on a live model must be exact, not an approximation of the
    compiler's inertia. The viewer holds one model for its whole session, so this is the only way a
    new pendulum can appear in it."""
    from tabpfn_control.mujoco_env import MujocoFuruta
    rng = np.random.default_rng(0)
    S = np.stack([rng.uniform(-3, 3, 80), rng.uniform(-6, 6, 80),
                  rng.uniform(-3, 3, 80), rng.uniform(-8, 8, 80)], -1)
    U = rng.uniform(-10, 10, 80)
    for Lp, mt in [(0.20, 0.010), (0.40, 0.080), (0.31, 0.045)]:
        fresh = MujocoFuruta(Lp=Lp, mp=0.02, m_tip=mt)
        live = MujocoFuruta(mp=0.02)
        live.set_geometry(Lp=Lp, m_tip=mt)
        assert np.array_equal(fresh.dynamics(S, U), live.dynamics(S, U))


def test_tip_mass_moves_the_mass_outward():
    """A mass at the end is not the same as more rod. It moves the centre of mass towards the tip
    and raises the inertia about the hinge, which is what makes the swing-up harder.

    Deliberately not a period test: the arm is free to counter-rotate, so this is a coupled system
    and not a simple pendulum, and the naive period comparison gives the opposite of the textbook
    answer. The feasibility sweep behind LP_RANGE/MTIP_RANGE is what establishes the task stays
    solvable; this only checks the mass went where it was asked to go.
    """
    from tabpfn_control.mujoco_env import MujocoFuruta
    bare, tipped = MujocoFuruta(mp=0.02, Lp=0.3), MujocoFuruta(mp=0.02, Lp=0.3, m_tip=0.06)
    i = bare.model.body("pendulum").id
    assert np.isclose(bare.model.body_ipos[i][2], 0.15, atol=1e-6)       # a rod balances at half
    assert tipped.model.body_ipos[i][2] > 0.25                          # the tip drags it outward
    assert tipped.model.body_inertia[i][0] > 3 * bare.model.body_inertia[i][0]
    assert np.isclose(tipped.model.body_mass[i], 0.08, atol=1e-9)


def test_the_energy_feature_cannot_leak_the_plant():
    """pump_feats must read fixed reference constants, never the pendulum it is running on."""
    from tabpfn_control.mujoco_env import MujocoFuruta
    from tabpfn_control.policy import policy_features, pump_ref_for
    ref = pump_ref_for(0.02, 0.30, 0.045)
    rng = np.random.default_rng(0)
    S = np.stack([rng.uniform(-3, 3, 40), rng.uniform(-6, 6, 40),
                  rng.uniform(-3, 3, 40), rng.uniform(-8, 8, 40)], -1)
    short = MujocoFuruta(Lp=0.20, mp=0.02, m_tip=0.01)
    long_ = MujocoFuruta(Lp=0.40, mp=0.02, m_tip=0.08)
    assert np.array_equal(policy_features(short, S, ref=ref), policy_features(long_, S, ref=ref))
