from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sumo"))

from sumo_sim import SumoPursuit, TargetPolicy


class _Vehicle:
    def __init__(self) -> None:
        self.resumed = []
        self.replaced = []
        self.routes = []

    def isStopped(self, vid):
        return True

    def resume(self, vid):
        self.resumed.append(vid)

    def getStops(self, vid, limit):
        return (object(), object())

    def replaceStop(self, vid, index, edge):
        self.replaced.append((vid, index, edge))

    def setRoute(self, vid, edges):
        self.routes.append((vid, list(edges)))


class _SpeedVehicle:
    def __init__(self) -> None:
        self.speeds = []

    def getIDList(self):
        return ("police_0", "police_1", "fugitive")

    def setSpeed(self, vid, speed):
        self.speeds.append((vid, speed))


def _sim() -> SumoPursuit:
    sim = SumoPursuit.__new__(SumoPursuit)
    sim.fid = "fugitive"
    sim.legacy_fugitive_stops = False
    sim.known_sids = set()
    sim._loops = set()
    sim.conn = SimpleNamespace(vehicle=_Vehicle())
    sim.seg = {
        1: SimpleNamespace(start_id=10, end_id=11, length_m=30.0),
        2: SimpleNamespace(start_id=11, end_id=12, length_m=40.0),
        3: SimpleNamespace(start_id=12, end_id=13, length_m=50.0),
        4: SimpleNamespace(start_id=12, end_id=11, length_m=80.0),  # immediate U-turn
        5: SimpleNamespace(start_id=13, end_id=14, length_m=60.0),
    }
    outgoing = {11: [2], 12: [4, 3], 13: [5]}
    sim.network_out = lambda node: outgoing.get(node, [])
    return sim


def test_moving_fugitive_has_no_terminal_stop_and_gets_lookahead() -> None:
    sim = _sim()

    sim._set_fugitive_route(["e1", "e2"])

    vehicle = sim.conn.vehicle
    assert vehicle.resumed == ["fugitive"]
    assert vehicle.replaced == [("fugitive", 0, ""), ("fugitive", 0, "")]
    assert vehicle.routes == [("fugitive", ["e1", "e2", "e3", "e5"])]


def test_provisional_route_prefers_forward_edge_over_uturn() -> None:
    sim = _sim()

    assert sim._extend_fugitive_route(["e1", "e2"], min_edges=3) == ["e1", "e2", "e3"]


def test_generic_policy_keeps_its_recommended_edge_instead_of_becoming_pure_pursuit() -> None:
    policy = TargetPolicy.__new__(TargetPolicy)
    policy.n = 2
    policy.stash = {}
    policy.latest = (
        SimpleNamespace(segment_id=17, next_intersection_id=8),
        SimpleNamespace(segment_id=None, next_intersection_id=9),
    )
    policy.team = object()

    assert policy.goals() == [("edge", 17), ("hold", 9)]


def test_direct_speed_preserves_safety_bits_and_ignores_road_limit() -> None:
    sim = SumoPursuit.__new__(SumoPursuit)
    sim.direct_speed = True

    assert sim._speed_mode(True) == 71
    assert sim._speed_mode(False) == 95


def test_direct_speed_commands_case_targets_without_overriding_yielding_officer() -> None:
    sim = SumoPursuit.__new__(SumoPursuit)
    vehicle = _SpeedVehicle()
    sim.conn = SimpleNamespace(vehicle=vehicle)
    sim.direct_speed = True
    sim.police_ids = ["police_0", "police_1"]
    sim.fid = "fugitive"
    sim._passing = {"police_1"}
    sim.case = SimpleNamespace(vp=22.0, vf=26.0)

    sim._apply_cruise_speeds()

    assert vehicle.speeds == [("police_0", 22.0), ("fugitive", 26.0)]
