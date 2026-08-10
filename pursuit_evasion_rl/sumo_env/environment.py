"""SUMO-backed pursuit environment: six police vehicles, one fugitive, real traffic.

Replaces the abstract seven-vehicle simulator with a microscopic one.  Every
vehicle -- including a few hundred ordinary drivers -- obeys car-following,
lane-changing and traffic signals, so congestion is something that happens to
the pursuit rather than a number the environment asserts.

Decision semantics are kept deliberately identical to the abstract
environment so the policy side (masked MAPPO over asynchronous decisions)
transfers unchanged: an officer only chooses when it reaches a junction, its
choice is one of that junction's legal outgoing edges, and the choice is
executed by retargeting the vehicle rather than by teleporting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pursuit_evasion_rl.sumo_env.barriers import (
    Barrier, BarrierConfig, apply_barriers, choose_barriers,
)
from pursuit_evasion_rl.sumo_env.net_builder import SumoNetwork, sumo_binary, sumo_home
from pursuit_evasion_rl.sumo_env.traffic import (
    TrafficConfig,
    dispersed_spawn_edges,
    write_background_routes,
)

POLICE_COUNT = 6
POLICE_IDS: tuple[str, ...] = tuple(f"P{index}" for index in range(POLICE_COUNT))
FUGITIVE_ID = "F0"
STAY_ACTION = 5
ACTION_SIZE = 6


class EpisodeOutcome:
    """Terminal outcomes, mirroring the abstract environment's vocabulary."""

    CAPTURE = "capture"
    ESCAPE = "escape"
    TIMEOUT = "timeout"
    #: Episode discarded: the simulator removed a tracked vehicle.
    VOID = "void"


@dataclass(frozen=True, slots=True)
class SumoEpisodeConfig:
    """Episode-level physics and termination settings."""

    step_length_s: float = 1.0
    max_steps: int = 450
    capture_radius_m: float = 25.0
    police_speed_mps: float = 16.0
    fugitive_speed_mps: float = 9.0
    #: Fugitive escapes by reaching an edge with no outgoing connections that
    #: stays inside the extract -- i.e. the bbox boundary.
    escape_on_boundary: bool = True
    gui: bool = False
    #: Milliseconds SUMO-GUI pauses between steps.  Without this the window
    #: replays a 400-step episode faster than anyone can watch it.
    gui_delay_ms: int = 120
    #: Cars are ~5 m on a ~3 km extract, i.e. sub-pixel at whole-network zoom,
    #: so SUMO-GUI shows an empty map unless vehicles are drawn oversized.
    gui_vehicle_scale: float = 14.0
    #: Camera follows the fugitive; without it the viewport sits on the whole
    #: city and the pursuit happens somewhere off in the corner.
    gui_follow_fugitive: bool = True
    #: SUMO zoom level; ~1200 frames a few blocks around the tracked car.
    gui_zoom: float = 1200.0
    seed: int = 0
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    #: 에피소드마다 무작위로 놓이는 도로 차단(공사·사고·경찰 차단선).
    #: 모든 도로가 늘 열려 있는 지도에서는 우회 판단을 배울 일이 없다.
    barriers: BarrierConfig = field(default_factory=BarrierConfig)

    def __post_init__(self) -> None:
        if self.step_length_s <= 0 or self.max_steps <= 0:
            raise ValueError("step_length_s and max_steps must be positive")
        if self.capture_radius_m <= 0:
            raise ValueError("capture_radius_m must be positive")


@dataclass(frozen=True, slots=True)
class VehicleView:
    """Read-only snapshot of one tracked vehicle."""

    vehicle_id: str
    edge_id: str
    lane_id: str
    position_xy: tuple[float, float]
    speed_mps: float
    at_junction: bool
    route_edges: tuple[str, ...]

    @property
    def street_name(self) -> str:
        return self.edge_id


@dataclass(frozen=True, slots=True)
class SumoEpisodeState:
    """Everything a policy or a dashboard needs about one simulation step."""

    step: int
    sim_time_s: float
    police: tuple[VehicleView, ...]
    fugitive: VehicleView | None
    background_count: int
    outcome: str | None
    events: tuple[str, ...]


class SumoPursuitEnv:
    """Gymnasium-shaped pursuit environment driven by SUMO through TraCI.

    Only one instance may be connected at a time per TraCI label; the class
    manages its own connection and always closes it, including on failure, so
    a crashed episode cannot leave an orphaned ``sumo`` process holding a port.
    """

    #: Fraction of the map trimmed from each side when choosing where the
    #: fugitive starts, so it cannot reach a bbox boundary immediately.
    INTERIOR_MARGIN = 0.28

    def __init__(
        self,
        network: SumoNetwork,
        config: SumoEpisodeConfig | None = None,
        *,
        work_dir: str | Path = "cache/sumo/runs",
        label: str = "police-runner",
    ) -> None:
        sumo_home()  # fail fast with a clear message if SUMO is missing
        self.network = network
        self.config = config or SumoEpisodeConfig()
        self.work_dir = Path(work_dir)
        self.label = label
        self._connected = False
        self._step = 0
        self._outcome: str | None = None
        self._events: list[str] = []
        self._net = None
        self._route_path: Path | None = None
        self._congested: tuple[str, ...] = ()
        self._targets: dict[str, str] = {}
        self._barriers: tuple[Barrier, ...] = ()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def sumolib_net(self):
        if self._net is None:
            import sumolib

            self._net = sumolib.net.readNet(str(self.network.net_path))
        return self._net

    @property
    def outcome(self) -> str | None:
        return self._outcome

    @property
    def connected(self) -> bool:
        return self._connected

    def reset(self, *, seed: int | None = None) -> SumoEpisodeState:
        """Start a fresh episode: background traffic first, pursuit vehicles after."""
        import traci

        self.close()
        episode_seed = self.config.seed if seed is None else int(seed)
        traffic_config = _replace_seed(self.config.traffic, episode_seed)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        route_path = self.work_dir / f"background_{episode_seed}.rou.xml"
        self._route_path, self._congested = write_background_routes(
            self.sumolib_net, route_path, traffic_config
        )

        binary = sumo_binary("sumo-gui" if self.config.gui else "sumo")
        command = [
            binary,
            "-n", str(self.network.net_path),
            "-r", str(self._route_path),
            "--step-length", str(self.config.step_length_s),
            "--seed", str(episode_seed % 65536),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--ignore-route-errors", "true",
            "--time-to-teleport", "-1",       # never teleport a jammed vehicle
            "--collision.action", "warn",
            "--default.carfollowmodel", "Krauss",
        ]
        if self.config.gui:
            command += [
                "--start", "true",
                "--quit-on-end", "false",          # leave the result on screen
                "--delay", str(self.config.gui_delay_ms),
                "--gui-settings-file", str(self._write_gui_settings()),
                "--window-size", "1280,860",
            ]
        traci.start(command, label=self.label)
        self._connection = traci.getConnection(self.label)
        self._connected = True
        self._step = 0
        self._outcome = None
        self._events = []
        self._targets = {}
        self._barriers = ()

        for _ in range(max(0, traffic_config.warmup_steps)):
            self._connection.simulationStep()

        self._insert_pursuit_vehicles(episode_seed)
        self._connection.simulationStep()
        self._install_barriers(episode_seed)
        self._configure_gui()
        self._events.append("episode_started")
        return self.episode_state()

    def close(self) -> None:
        if not self._connected:
            return
        try:
            self._connection.close()
        except Exception:
            pass
        finally:
            self._connected = False

    def __enter__(self) -> "SumoPursuitEnv":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    #: Name of the view scheme this environment writes and then selects.
    GUI_SCHEME = "police-runner"

    def _write_gui_settings(self) -> Path:
        """Write the SUMO-GUI view configuration used for demos.

        ``vehicleExaggeration`` does the real work.  At the zoom needed to see
        a whole city extract a true-to-size 5 m car covers less than a pixel,
        which is why an unconfigured SUMO-GUI looks like an empty road map.
        Colouring by the given vehicle colour keeps the assignments this
        environment makes -- blue officers, red fugitive, default background
        traffic -- instead of SUMO's own speed-based palette.
        """
        path = self.work_dir / "gui-settings.xml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"""<viewsettings>
  <scheme name="{self.GUI_SCHEME}">
    <opengl dither="0" fps="0" drawBoundaries="0"/>
    <background backgroundColor="18,20,24" showGrid="0"
                gridXSize="100.00" gridYSize="100.00"/>
    <edges laneEdgeMode="0" scaleMode="0" laneShowBorders="1"
           showLinkDecals="1" showRails="1" edgeName_show="0"
           streetName_show="0" edgeValue_show="0">
      <colorScheme name="uniform">
        <entry color="90,95,105"/>
      </colorScheme>
      <scalingScheme name="default">
        <entry color="1.0"/>
      </scalingScheme>
    </edges>
    <vehicles vehicleMode="0" vehicleQuality="2"
              vehicle_minGap_show="0" vehicleName_show="0"
              vehicleText_show="0" minVehicleSize="6.00"
              vehicleExaggeration="{self.config.gui_vehicle_scale:.2f}"
              showBlinker="1" drawMinGap="0">
      <colorScheme name="given vehicle/type/route color"/>
      <scalingScheme name="uniform">
        <entry color="1.0"/>
      </scalingScheme>
    </vehicles>
    <junctions junctionMode="0" drawShape="1" drawCrossingsAndWalkingareas="0"
               junctionName_show="0" internalJunctionName_show="0">
      <colorScheme name="uniform">
        <entry color="60,64,72"/>
      </colorScheme>
    </junctions>
    <additionals addMode="0" addName_show="0"/>
    <pois poiName_show="0" poiTextParam_show="0"/>
    <polys polyName_show="0" polyType_show="0"/>
    <legend showSizeLegend="1" showColorLegend="0" showVehicleColorLegend="0"/>
  </scheme>
  <delay value="{self.config.gui_delay_ms}"/>
</viewsettings>
""",
            encoding="utf-8",
        )
        return path

    def _configure_gui(self) -> None:
        """Select the demo scheme, zoom in, and lock the camera on the pursuit."""
        if not self.config.gui:
            return
        view = "View #0"
        connection = self._connection
        try:
            connection.gui.setSchema(view, self.GUI_SCHEME)
        except Exception:
            # The scheme file may not have been picked up; the default view is
            # still usable, so this is a downgrade rather than a failure.
            self._events.append("gui_scheme_unavailable")
        try:
            if self.config.gui_follow_fugitive:
                connection.gui.trackVehicle(view, FUGITIVE_ID)
            connection.gui.setZoom(view, self.config.gui_zoom)
        except Exception as error:
            self._events.append(f"gui_camera_skipped:{type(error).__name__}")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _insert_pursuit_vehicles(self, seed: int) -> None:
        """Place the fugitive and six officers on well-separated edges."""
        police_edges = dispersed_spawn_edges(
            self.sumolib_net, count=POLICE_COUNT, seed=seed + 977, min_separation_m=400.0
        )
        fugitive_edge = self._interior_spawn_edge(seed + 313, exclude=set(police_edges))
        connection = self._connection
        connection.route.add("fugitive_route", [fugitive_edge])
        connection.vehicle.add(
            FUGITIVE_ID, "fugitive_route", typeID="DEFAULT_VEHTYPE",
            departSpeed="0", departLane="best",
        )
        connection.vehicle.setMaxSpeed(FUGITIVE_ID, self.config.fugitive_speed_mps)
        connection.vehicle.setColor(FUGITIVE_ID, (220, 60, 50, 255))
        for index, (officer, edge_id) in enumerate(zip(POLICE_IDS, police_edges)):
            route_id = f"route_{officer}"
            connection.route.add(route_id, [edge_id])
            connection.vehicle.add(
                officer, route_id, typeID="DEFAULT_VEHTYPE",
                departSpeed="0", departLane="best",
            )
            connection.vehicle.setMaxSpeed(officer, self.config.police_speed_mps)
            connection.vehicle.setColor(officer, (40, 110, 220, 255))

    def _interior_spawn_edge(self, seed: int, exclude: set[str]) -> str:
        """An edge well inside the extract for the fugitive to start on.

        Spawning next to the bbox edge let the fugitive reach a boundary in
        seconds, which scores as an escape before any pursuit happened -- the
        same defect the abstract environment had until its placement filter
        required interior, movable nodes.  Requiring the central portion of
        the map, and outgoing connections, removes that trivial escape.
        """
        import random

        net = self.sumolib_net
        xs, ys = [], []
        for edge in net.getEdges():
            if edge.getID().startswith(":"):
                continue
            for x, y in edge.getShape():
                xs.append(x)
                ys.append(y)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        margin_x = (max_x - min_x) * self.INTERIOR_MARGIN
        margin_y = (max_y - min_y) * self.INTERIOR_MARGIN

        candidates = []
        for edge in net.getEdges():
            edge_id = edge.getID()
            if edge_id.startswith(":") or edge_id in exclude:
                continue
            if not edge.getOutgoing() or edge.getLength() < 40.0:
                continue
            shape = edge.getShape()
            cx = sum(point[0] for point in shape) / len(shape)
            cy = sum(point[1] for point in shape) / len(shape)
            if (min_x + margin_x) <= cx <= (max_x - margin_x) and (
                min_y + margin_y
            ) <= cy <= (max_y - margin_y):
                candidates.append(edge_id)
        if not candidates:
            raise RuntimeError("network has no interior edge for the fugitive")
        return random.Random(seed).choice(sorted(candidates))

    @property
    def barriers(self) -> tuple[Barrier, ...]:
        """이번 에피소드에 설치된 차단 구간."""
        return self._barriers

    def _install_barriers(self, seed: int) -> None:
        """도주자 위치를 피해 차단 구간을 고르고 시뮬레이션에 반영한다."""
        if not self.config.barriers.enabled:
            return
        fugitive = self._vehicle_view(FUGITIVE_ID)
        chosen = choose_barriers(
            self.sumolib_net, self.config.barriers, seed=seed + 4231,
            fugitive_xy=fugitive.position_xy if fugitive else None,
        )
        applied = set(apply_barriers(self._connection, chosen))
        self._barriers = tuple(item for item in chosen if item.edge_id in applied)
        if self._barriers:
            self._events.append(f"barriers_installed:{len(self._barriers)}")

    # ------------------------------------------------------------------
    # Observation of raw state
    # ------------------------------------------------------------------
    def _vehicle_view(self, vehicle_id: str) -> VehicleView | None:
        connection = self._connection
        if vehicle_id not in connection.vehicle.getIDList():
            return None
        lane_id = connection.vehicle.getLaneID(vehicle_id)
        edge_id = connection.vehicle.getRoadID(vehicle_id)
        return VehicleView(
            vehicle_id=vehicle_id,
            edge_id=edge_id,
            lane_id=lane_id,
            position_xy=tuple(connection.vehicle.getPosition(vehicle_id)),
            speed_mps=float(connection.vehicle.getSpeed(vehicle_id)),
            at_junction=edge_id.startswith(":"),
            route_edges=tuple(connection.vehicle.getRoute(vehicle_id)),
        )

    def episode_state(self) -> SumoEpisodeState:
        police = tuple(
            view for view in (self._vehicle_view(officer) for officer in POLICE_IDS) if view
        )
        fugitive = self._vehicle_view(FUGITIVE_ID)
        background = max(
            0, len(self._connection.vehicle.getIDList()) - len(police) - (1 if fugitive else 0)
        )
        return SumoEpisodeState(
            step=self._step,
            sim_time_s=self._step * self.config.step_length_s,
            police=police,
            fugitive=fugitive,
            background_count=background,
            outcome=self._outcome,
            events=tuple(self._events[-24:]),
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def legal_targets(self, vehicle_id: str) -> tuple[str, ...]:
        """Outgoing edges reachable from the vehicle's current edge.

        This is the SUMO analogue of the abstract environment's action mask:
        the slot order is stable (sorted edge id) so a stored mask means the
        same thing at rollout and at update time.
        """
        view = self._vehicle_view(vehicle_id)
        if view is None:
            return ()
        edge_id = view.edge_id
        if edge_id.startswith(":"):
            # Inside a junction: keep the committed route, no fresh decision.
            return ()
        edge = self.sumolib_net.getEdge(edge_id)
        outgoing = sorted({item.getID() for item in edge.getOutgoing()})
        return tuple(outgoing[: ACTION_SIZE - 1])

    def action_mask(self, vehicle_id: str) -> list[bool]:
        """Boolean mask over ``ACTION_SIZE`` slots; STAY is always legal."""
        mask = [False] * ACTION_SIZE
        for index, _ in enumerate(self.legal_targets(vehicle_id)):
            mask[index] = True
        mask[STAY_ACTION] = True
        return mask

    def _apply_action(self, vehicle_id: str, action: int) -> None:
        if action == STAY_ACTION:
            return
        targets = self.legal_targets(vehicle_id)
        if not targets or action < 0 or action >= len(targets):
            return
        try:
            self._connection.vehicle.changeTarget(vehicle_id, targets[action])
            self._events.append(f"{vehicle_id}:target:{targets[action]}")
        except Exception:
            # An unreachable target is a legal thing for a policy to ask for on
            # a directed network; keep the previous route rather than failing.
            self._events.append(f"{vehicle_id}:target_rejected")

    def reachable(self, from_edge: str, to_edge: str) -> bool:
        """Whether a driving route exists; SUMO rejects unreachable targets."""
        if from_edge.startswith(":") or to_edge.startswith(":"):
            return False
        if from_edge == to_edge:
            return True
        try:
            path, _cost = self.sumolib_net.getShortestPath(
                self.sumolib_net.getEdge(from_edge), self.sumolib_net.getEdge(to_edge)
            )
        except Exception:
            return False
        return bool(path)

    def set_target(self, vehicle_id: str, edge_id: str) -> bool:
        """Dispatch a vehicle to an edge and let SUMO route it there.

        These are the dispatch semantics an operations room actually uses --
        "go block that junction" -- rather than turn-by-turn steering, and it
        lets SUMO reroute around whatever congestion appears en route.
        """
        view = self._vehicle_view(vehicle_id)
        if view is None or not self.reachable(view.edge_id, edge_id):
            return False
        try:
            self._connection.vehicle.changeTarget(vehicle_id, edge_id)
            self._targets[vehicle_id] = edge_id
            return True
        except Exception:
            return False

    def current_target(self, vehicle_id: str) -> str | None:
        return self._targets.get(vehicle_id)

    def step_targets(self, targets: Mapping[str, str]) -> SumoEpisodeState:
        """Dispatch targets, advance one step, evaluate terminals."""
        if not self._connected:
            raise RuntimeError("call reset() before step()")
        if self._outcome is not None:
            return self.episode_state()
        for vehicle_id, edge_id in targets.items():
            if edge_id and self.set_target(vehicle_id, edge_id):
                self._events.append(f"{vehicle_id}:dispatch:{edge_id}")
        self._keep_routed()
        self._connection.simulationStep()
        self._step += 1
        self._evaluate_terminal()
        return self.episode_state()

    def step(self, actions: Mapping[str, int]) -> SumoEpisodeState:
        """Apply officer actions, advance one simulation step, evaluate terminals."""
        if not self._connected:
            raise RuntimeError("call reset() before step()")
        if self._outcome is not None:
            return self.episode_state()
        for officer in POLICE_IDS:
            action = actions.get(officer)
            if action is not None:
                self._apply_action(officer, int(action))
        fugitive_action = actions.get(FUGITIVE_ID)
        if fugitive_action is not None:
            self._apply_action(FUGITIVE_ID, int(fugitive_action))

        self._connection.simulationStep()
        self._step += 1
        self._evaluate_terminal()
        return self.episode_state()

    def _keep_routed(self) -> None:
        """Extend any tracked vehicle that is about to run out of route.

        SUMO retires a vehicle that reaches the end of its route, reporting it
        as "arrived".  For background traffic that is correct; for the pursuit
        vehicles it would silently delete an agent mid-episode and -- because
        a missing fugitive looks exactly like one that drove off the map --
        score a phantom escape.  Keeping them routed makes the terminal
        conditions mean what they say.
        """
        connection = self._connection
        for vehicle_id in (FUGITIVE_ID, *POLICE_IDS):
            view = self._vehicle_view(vehicle_id)
            if view is None or view.edge_id.startswith(":"):
                continue
            try:
                index = connection.vehicle.getRouteIndex(vehicle_id)
            except Exception:
                continue
            if len(view.route_edges) - max(0, index) > 1:
                continue
            edge = self.sumolib_net.getEdge(view.edge_id)
            for successor in sorted(edge.getOutgoing(), key=lambda item: -item.getLength()):
                if self.set_target(vehicle_id, successor.getID()):
                    self._events.append(f"{vehicle_id}:route_extended")
                    break

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def min_separation_m(self) -> float | None:
        state = self.episode_state()
        if state.fugitive is None or not state.police:
            return None
        return min(
            math.dist(officer.position_xy, state.fugitive.position_xy)
            for officer in state.police
        )

    def _evaluate_terminal(self) -> None:
        state = self.episode_state()
        if state.fugitive is None:
            # Reaching a boundary edge is an escape and is detected below while
            # the vehicle still exists.  A fugitive that is simply gone means
            # SUMO retired it (route exhausted, collision, teleport), which is
            # an environment fault; recording it as an escape would inflate the
            # evader's score with episodes nobody played.
            self._outcome = EpisodeOutcome.VOID
            self._events.append("fugitive_vanished")
            return
        separation = self.min_separation_m()
        if separation is not None and separation <= self.config.capture_radius_m:
            self._outcome = EpisodeOutcome.CAPTURE
            self._events.append(f"capture:{separation:.1f}m")
            return
        if self.config.escape_on_boundary and self._on_boundary(state.fugitive):
            self._outcome = EpisodeOutcome.ESCAPE
            self._events.append("fugitive_reached_boundary")
            return
        if self._step >= self.config.max_steps:
            self._outcome = EpisodeOutcome.TIMEOUT
            self._events.append("timeout")

    def _on_boundary(self, view: VehicleView) -> bool:
        if view.edge_id.startswith(":"):
            return False
        try:
            edge = self.sumolib_net.getEdge(view.edge_id)
        except Exception:
            return False
        return not edge.getOutgoing()


def _replace_seed(config: TrafficConfig, seed: int) -> TrafficConfig:
    return TrafficConfig(
        vehicle_count=config.vehicle_count,
        insertion_window_s=config.insertion_window_s,
        warmup_steps=config.warmup_steps,
        congested_fraction=config.congested_fraction,
        congestion_multiplier=config.congestion_multiplier,
        min_trip_edges=config.min_trip_edges,
        seed=seed,
    )


__all__ = (
    "ACTION_SIZE",
    "EpisodeOutcome",
    "FUGITIVE_ID",
    "POLICE_COUNT",
    "POLICE_IDS",
    "STAY_ACTION",
    "SumoEpisodeConfig",
    "SumoEpisodeState",
    "SumoPursuitEnv",
    "VehicleView",
)