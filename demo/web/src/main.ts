import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./style.css";

// ---------------------------------------------------------------------------------------------
// types
type LatLon = [number, number];
interface MapInfo { name: string; label: string; roads: LatLon[][]; exits: LatLon[]; bounds: [LatLon, LatLon] }
// a barricade is a line across the road ([LatLon, LatLon]); older recordings carry just its point
type BarShape = LatLon | LatLon[];
interface Frame { t: number; p: LatLon[]; f: LatLon; role: number[]; tgt: (LatLon | null)[]; reg: string; open: number; ox: LatLon[]; bg?: LatLon[]; bk?: number[]; bars?: BarShape[]; kit?: boolean[]; sig?: string[] }
interface Setup { police: LatLon[]; fugitive: LatLon; policy: string; engine: string; profile: string; traffic: number | null; barricades?: BarShape[] }
// static street furniture of the simulated network: signal stop lines (coloured from Frame.sig) and crosswalks
interface NetInfo { stops: { tls: number; idx: number[]; line: LatLon[] }[]; cross: { w: number; pts: LatLon[] }[] }
interface Run { id: number; mapInfo: MapInfo; setup: Setup | null; frames: Frame[]; outcome: string | null; seconds: number | null; error: string | null; label: string; done: boolean; recorded?: boolean; net?: NetInfo }

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const OUTCOME: Record<string, string> = { success: "검거", escape: "탈출", timeout: "시간초과" };
const ENGINE: Record<string, string> = { sumo: "SUMO", point: "빠른 시뮬레이터" };

// ---------------------------------------------------------------------------------------------
// Server address: a user override is kept in this browser.  Quick-tunnel addresses expire, so a
// previously stored trycloudflare URL must not pin returning visitors to a dead backend after a redeploy.
const CURRENT_QUICK_TUNNEL = "https://fairfield-inherited-buyer-facts.trycloudflare.com";
function readStored(): string | null { try { return localStorage.getItem("apiBase"); } catch { return null; } }
function readSessionToken(): string { try { return sessionStorage.getItem("apiToken") ?? ""; } catch { return ""; } }
const storedApiBase = readStored();
const staleQuickTunnel = !!storedApiBase && storedApiBase.includes(".trycloudflare.com")
  && storedApiBase !== CURRENT_QUICK_TUNNEL;
let apiBase = (staleQuickTunnel ? CURRENT_QUICK_TUNNEL
  : storedApiBase ?? CURRENT_QUICK_TUNNEL).replace(/\/$/, "");
if (staleQuickTunnel) {
  try { localStorage.setItem("apiBase", apiBase); } catch { /* storage unavailable: keep in memory */ }
}
let apiToken = readSessionToken();
const api = (path: string) => `${apiBase}${path}`;
function apiHeaders(base: HeadersInit = {}) {
  const h = new Headers(base);
  if (apiToken) h.set("X-Demo-Token", apiToken);
  return h;
}
function apiError(context: string, err: unknown) {
  const msg = err instanceof Error ? err.message : String(err);
  const auth = msg.includes("401") ? " 접근 토큰이 맞는지 확인해 주세요." : "";
  const where = apiBase || "현재 페이지와 같은 주소";
  return `${context} (${msg}). 서버 주소: ${where}.${auth}`;
}

// ---------------------------------------------------------------------------------------------
// map
const dark = matchMedia("(prefers-color-scheme: dark)");
const map = L.map("map", { zoomControl: true, preferCanvas: true, attributionControl: true });
const canvas = L.canvas({ padding: 0.3 });
// OpenStreetMap tiles (dark mode recolours them with a CSS filter on the tile pane); the simulator's own
// road network is drawn on top, so the map stays readable even when tiles cannot load.
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);
const roadLayer = L.layerGroup().addTo(map);
function drawRoads() {
  roadLayer.clearLayers();
  if (!info) return;
  const col = css("--road");
  for (const r of info.roads) L.polyline(r, { renderer: canvas, color: col, weight: 2, opacity: 0.55, interactive: false }).addTo(roadLayer);
}
function showMap(m: MapInfo) {
  info = m;
  $("mapLabel").textContent = m.label;
  map.fitBounds(L.latLngBounds(m.bounds[0], m.bounds[1]), { padding: [10, 10] });
  drawRoads();
}
function fitAction(fr: Frame) {
  map.fitBounds(L.latLngBounds([...fr.p, fr.f]).pad(0.6), { maxZoom: 17 });
}
dark.addEventListener("change", () => { drawRoads(); redrawFrame(true); });
map.setView([36.351, 127.385], 15);

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const placeLayer = L.layerGroup().addTo(map);
const regionLayer = L.layerGroup().addTo(map);
const bgLayer = L.layerGroup().addTo(map);
const trailLayer = L.layerGroup().addTo(map);
const targetLayer = L.layerGroup().addTo(map);
const exitLayer = L.layerGroup().addTo(map);
const crossLayer = L.layerGroup().addTo(map);
const sigLayer = L.layerGroup().addTo(map);
const barLayer = L.layerGroup().addTo(map);
const vehLayer = L.layerGroup().addTo(map);

const vehIcon = (cls: string, text = "") => L.divIcon({ className: "", html: `<div class="veh ${cls}">${text}</div>`, iconSize: [22, 22], iconAnchor: [11, 11] });

// barricades are drawn across the road: a striped bar rotated to the line it closes, at least 24 px long
const BAR_HALF_M = 4;                     // half-length of a placed barricade before the server reports its real size
function metresPerPixel(at: L.LatLng): number {
  const p = map.latLngToLayerPoint(at);
  return map.distance(map.layerPointToLatLng(p), map.layerPointToLatLng(p.add([10, 0]))) / 10;
}
// the road direction (screen space) nearest to a point on the simulator's roads, or null
function roadDirection(ll: L.LatLng): L.Point | null {
  if (!info) return null;
  const p = map.latLngToLayerPoint(ll);
  let best: L.Point | null = null, bd = Infinity;
  for (const r of info.roads) {
    for (let i = 1; i < r.length; i++) {
      const a = map.latLngToLayerPoint(r[i - 1]), b = map.latLngToLayerPoint(r[i]);
      const d = L.LineUtil.closestPointOnSegment(p, a, b).distanceTo(p);
      if (d < bd && a.distanceTo(b) > 0) { bd = d; best = b.subtract(a); }
    }
  }
  return best ? best.divideBy(best.distanceTo(L.point(0, 0))) : null;
}
// a placed (snapped) point -> the line across its road
function crossLine(ll: L.LatLng): LatLon[] {
  const dir = roadDirection(ll);
  if (!dir) return [[ll.lat, ll.lng], [ll.lat, ll.lng]];
  const half = BAR_HALF_M / metresPerPixel(ll);
  const p = map.latLngToLayerPoint(ll), n = L.point(-dir.y * half, dir.x * half);
  const a = map.layerPointToLatLng(p.subtract(n)), b = map.layerPointToLatLng(p.add(n));
  return [[a.lat, a.lng], [b.lat, b.lng]];
}
function barMarker(shape: BarShape, seen: boolean, options: L.MarkerOptions = {}): L.Marker {
  const line = typeof shape[0] === "number" ? crossLine(L.latLng(shape as LatLon)) : shape as LatLon[];
  const a = map.latLngToLayerPoint(line[0]), b = map.latLngToLayerPoint(line[1]);
  const len = Math.max(24, a.distanceTo(b));
  const deg = a.distanceTo(b) > 0 ? Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI : 0;
  const centre = map.layerPointToLatLng(a.add(b).divideBy(2));
  const icon = L.divIcon({ className: "", html: `<div class="barr${seen ? " seen" : ""}" style="width:${len.toFixed(1)}px;transform:rotate(${deg.toFixed(1)}deg)"></div>`,
    iconSize: [len, 11], iconAnchor: [len / 2, 6] });
  return L.marker(centre, { icon, ...options });
}

// signals: the colour a driver on that lane sees (green if any of its movements may go)
const SIG: Record<string, string> = { green: "#03b26c", yellow: "#ffb300", red: "#f04452", off: "#8b95a1" };
function laneColour(state: string, idx: number[]): string {
  let y = false, r = false;
  for (const i of idx) {
    const ch = state.charAt(i);
    if (ch === "G" || ch === "g") return "green";
    if (ch === "y" || ch === "Y") y = true; else if (ch === "r" || ch === "u") r = true;
  }
  return y ? "yellow" : r ? "red" : "off";
}
// zoomed out: one dot per signal, coloured by the phase most of its movements are in
function dominantColour(state: string): string {
  let g = 0, y = 0, r = 0;
  for (const ch of state.toLowerCase()) { if (ch === "g") g++; else if (ch === "y") y++; else if (ch === "r") r++; }
  if (!g && !y && !r) return "off";
  if (y && y >= Math.max(g, r)) return "yellow";
  return g > r ? "green" : "red";
}
const DETAIL_ZOOM = 16;                   // stop lines and crosswalks from this zoom on, signal dots below
function drawCrossings() {
  crossLayer.clearLayers();
  const net = current?.net;
  if (!net?.cross.length || map.getZoom() < DETAIL_ZOOM) return;
  const mpp = metresPerPixel(map.getCenter());
  const dash = Math.max(2, 0.6 / mpp);
  for (const c of net.cross) {
    L.polyline(c.pts, { renderer: canvas, color: "#ffffff", weight: Math.min(16, Math.max(3, c.w / mpp)), opacity: 0.85,
      dashArray: `${dash.toFixed(1)} ${dash.toFixed(1)}`, lineCap: "butt", interactive: false }).addTo(crossLayer);
  }
}
function drawSignals(state: string[] | undefined) {
  sigLayer.clearLayers();
  const net = current?.net;
  if (!net?.stops.length || !state) return;
  if (map.getZoom() >= DETAIL_ZOOM) {
    for (const s of net.stops) {
      L.polyline(s.line, { renderer: canvas, color: SIG[laneColour(state[s.tls] ?? "", s.idx)], weight: 4, opacity: 0.95, lineCap: "butt", interactive: false }).addTo(sigLayer);
    }
    return;
  }
  const centre = new Map<number, [number, number, number]>();
  for (const s of net.stops) {
    const c = centre.get(s.tls) ?? [0, 0, 0];
    centre.set(s.tls, [c[0] + (s.line[0][0] + s.line[1][0]) / 2, c[1] + (s.line[0][1] + s.line[1][1]) / 2, c[2] + 1]);
  }
  centre.forEach(([la, lo, n], t) => {
    const col = SIG[dominantColour(state[t] ?? "")];
    L.circleMarker([la / n, lo / n], { renderer: canvas, radius: 4, color: css("--panel"), weight: 1.5, fillColor: col, fillOpacity: 1, interactive: false }).addTo(sigLayer);
  });
}
map.on("zoomend", () => {
  drawCrossings();
  if (map.hasLayer(placeLayer)) drawPlacement();
  else redrawFrame(true);
});

// ---------------------------------------------------------------------------------------------
// state
let info: MapInfo | null = null;
let replayMap = false;
type Mode = "police" | "fugitive" | "barricade";
let mode: Mode = "police";
let police: L.LatLng[] = [];
let barricades: L.LatLng[] = [];
let fugitive: L.LatLng | null = null;
const opts = { policy: "ga", engine: "sumo", profile: "ideal", traffic: "" as string, ignoreRed: false, carry: false };
const runs: Run[] = [];
let current: Run | null = null;
let cursor = 0;            // playback time in frames (float)
let playing = false;
let follow = true;         // keep playing as new frames stream in
let runSeq = 0;

// ---------------------------------------------------------------------------------------------
// segmented controls
function seg(id: string, key: keyof typeof opts, after?: () => void) {
  const el = $(id);
  el.querySelectorAll<HTMLButtonElement>("button").forEach((b) => b.addEventListener("click", () => {
    el.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    (opts as Record<string, unknown>)[key] = b.dataset.v ?? "";
    after?.();
  }));
}
seg("policySeg", "policy");
seg("engineSeg", "engine", syncOpts);
seg("profileSeg", "profile", syncOpts);
seg("trafficSeg", "traffic");
$<HTMLInputElement>("ignoreRed").addEventListener("change", (e) => { opts.ignoreRed = (e.target as HTMLInputElement).checked; });
$<HTMLInputElement>("carry").addEventListener("change", (e) => { opts.carry = (e.target as HTMLInputElement).checked; });
function syncOpts() {
  $("sumoOpts").hidden = opts.engine !== "sumo";
  $("realOpts").hidden = !(opts.engine === "sumo" && opts.profile === "real");
}
syncOpts();

function setMode(m: Mode) {
  mode = m;
  $("modePolice").setAttribute("aria-pressed", String(m === "police"));
  $("modeFugitive").setAttribute("aria-pressed", String(m === "fugitive"));
  $("modeBarricade").setAttribute("aria-pressed", String(m === "barricade"));
}
$("modePolice").addEventListener("click", () => setMode("police"));
$("modeFugitive").addEventListener("click", () => setMode("fugitive"));
$("modeBarricade").addEventListener("click", () => setMode("barricade"));

// nearest point on the simulator's roads (the server snaps again; this just shows where it lands)
// The server stands a barricade 5-95 % along its road (clear of the intersections); clamp the same way
// here so the placed barricade is shown where it will block.
const BAR_MIN_FRAC = 0.05, BAR_MAX_FRAC = 0.95;
function snapToRoad(ll: L.LatLng): L.LatLng {
  if (!info) return ll;
  const p = map.latLngToLayerPoint(ll);
  let best: { road: LatLon[]; i: number; t: number } | null = null, bd = Infinity;
  for (const r of info.roads) {
    for (let i = 1; i < r.length; i++) {
      const a = map.latLngToLayerPoint(r[i - 1]), b = map.latLngToLayerPoint(r[i]);
      const q = L.LineUtil.closestPointOnSegment(p, a, b), d = q.distanceTo(p);
      if (d < bd) { bd = d; best = { road: r, i, t: a.distanceTo(b) > 0 ? q.distanceTo(a) / a.distanceTo(b) : 0 }; }
    }
  }
  if (!best) return ll;
  // metres along the road to the snapped point, clamped, then back to a point on the road
  const pts = best.road.map((q) => L.latLng(q));
  const seg = pts.slice(1).map((q, k) => pts[k].distanceTo(q));
  const total = seg.reduce((s, x) => s + x, 0);
  let along = seg.slice(0, best.i - 1).reduce((s, x) => s + x, 0) + best.t * seg[best.i - 1];
  along = Math.min(BAR_MAX_FRAC * total, Math.max(BAR_MIN_FRAC * total, along));
  for (let k = 0; k < seg.length; k++) {
    if (along <= seg[k] || k === seg.length - 1) {
      const w = seg[k] > 0 ? Math.min(1, along / seg[k]) : 0;
      return L.latLng(pts[k].lat + w * (pts[k + 1].lat - pts[k].lat), pts[k].lng + w * (pts[k + 1].lng - pts[k].lng));
    }
    along -= seg[k];
  }
  return ll;
}

// ---------------------------------------------------------------------------------------------
// placement
function drawPlacement() {
  placeLayer.clearLayers();
  police.forEach((ll, i) => {
    const m = L.marker(ll, { icon: vehIcon("police", String(i + 1)), draggable: true, keyboard: true, title: `경찰차 ${i + 1}` });
    m.on("dragend", () => { police[i] = m.getLatLng(); });
    m.on("click", () => { police.splice(i, 1); drawPlacement(); });
    m.addTo(placeLayer);
  });
  if (fugitive) {
    const m = L.marker(fugitive, { icon: vehIcon("fugitive"), draggable: true, title: "도주차" });
    m.on("dragend", () => { fugitive = m.getLatLng(); });
    m.addTo(placeLayer);
  }
  barricades.forEach((ll, i) => {
    const m = barMarker([ll.lat, ll.lng], false, { draggable: true, keyboard: true, title: `바리케이드 ${i + 1}` });
    m.on("dragend", () => { barricades[i] = snapToRoad(m.getLatLng()); drawPlacement(); });
    m.on("click", () => { barricades.splice(i, 1); drawPlacement(); });
    m.addTo(placeLayer);
  });
  $("counts").textContent = `경찰차 ${police.length}대 · 도주차 ${fugitive ? 1 : 0}대 · 바리케이드 ${barricades.length}개`;
}
map.on("click", (e: L.LeafletMouseEvent) => {
  if (replayMap || current && !current.done) return;
  showPlacement();
  if (mode === "police") { if (police.length < 12) police.push(e.latlng); }
  else if (mode === "barricade") { if (barricades.length < 20) barricades.push(snapToRoad(e.latlng)); }
  else fugitive = e.latlng;
  drawPlacement();
});
$("clear").addEventListener("click", () => {
  if (replayMap) return;
  police = []; fugitive = null; barricades = []; showPlacement(); drawPlacement();
});
$("example").addEventListener("click", async () => {
  if (!info) return;
  if (replayMap) { setStatus("error", "녹화 시연에서는 새 배치를 만들 수 없어요. 서버 주소를 확인하고 지도를 다시 불러와 주세요."); return; }
  const n = Math.max(2, Math.min(12, Number($<HTMLInputElement>("exampleN").value) || 6));
  try {
    const r = await fetch(api("/api/example"), { method: "POST", headers: apiHeaders({ "content-type": "application/json" }),
      body: JSON.stringify({ map: info.name, n, seed: Math.floor(Math.random() * 1000) }) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json() as { police: LatLon[]; fugitive: LatLon };
    police = d.police.map((p) => L.latLng(p[0], p[1]));
    fugitive = L.latLng(d.fugitive[0], d.fugitive[1]);
    showPlacement(); drawPlacement();
  } catch (err) { setStatus("error", apiError("예시 배치를 불러오지 못했어요", err)); }
});

function showPlacement() {
  // leaving a finished run's view: back to editing the placement
  playing = false;
  follow = false;
  cancelAnimationFrame(rafId);
  $("play").textContent = "▶";
  regionLayer.clearLayers(); bgLayer.clearLayers(); trailLayer.clearLayers(); targetLayer.clearLayers(); vehLayer.clearLayers(); exitLayer.clearLayers(); barLayer.clearLayers();
  sigLayer.clearLayers(); crossLayer.clearLayers();
  $("barLegend").hidden = true; $("sigLegend").hidden = true; $("crossLegend").hidden = true;
  map.addLayer(placeLayer);
}

// ---------------------------------------------------------------------------------------------
// map data
async function loadMaps() {
  try {
    const r = await fetch(api("/api/maps"), { headers: apiHeaders() });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const list = await r.json() as { name: string; label: string }[];
    const sel = $<HTMLSelectElement>("mapSelect");
    if (!list.length) throw new Error("지도 목록이 비어 있어요");
    sel.innerHTML = list.map((m) => `<option value="${m.name}">${m.label}</option>`).join("");
    sel.onchange = () => loadMap(sel.value).catch((err) => setStatus("error", apiError("지도를 불러오지 못했어요", err)));
    await loadMap(list[0].name);
    setStatus("idle", "지도를 눌러 경찰차와 도주차를 놓아 주세요");
  } catch (err) {
    setStatus("error", `${apiError("서버에 연결하지 못했어요", err)} 녹화된 SUMO 시연은 볼 수 있어요.`);
  }
}
async function loadMap(name: string) {
  const r = await fetch(api(`/api/map/${name}`), { headers: apiHeaders() });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  replayMap = false;
  showMap(await r.json() as MapInfo);
  police = []; fugitive = null; barricades = []; showPlacement(); drawPlacement();
}

// ---------------------------------------------------------------------------------------------
// running
function setStatus(state: string, text: string) { const s = $("status"); s.dataset.state = state; s.textContent = text; }

$("fallback").addEventListener("click", async () => {
  try {
    const res = await fetch("/fallback.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const saved = await res.json() as { map: MapInfo; setup: Setup; frames: Frame[]; outcome: string; seconds: number };
    if (!saved.frames.length) throw new Error("녹화 프레임이 없어요");
    replayMap = true;
    showMap(saved.map);
    police = saved.setup.police.map((p) => L.latLng(p[0], p[1]));
    fugitive = L.latLng(saved.setup.fugitive[0], saved.setup.fugitive[1]);
    barricades = (saved.setup.barricades ?? []).map((q) => typeof q[0] === "number"
      ? L.latLng((q as LatLon)[0], (q as LatLon)[1])
      : L.latLng(((q as LatLon[])[0][0] + (q as LatLon[])[1][0]) / 2, ((q as LatLon[])[0][1] + (q as LatLon[])[1][1]) / 2));
    drawPlacement();
    const replay: Run = { id: ++runSeq, mapInfo: saved.map, setup: saved.setup, frames: saved.frames,
      outcome: saved.outcome, seconds: saved.seconds, error: null, done: true, recorded: true,
      label: "녹화 시연 · GA 역할 정책 · SUMO (신호 없음) · 경찰 6대" };
    runs.unshift(replay);
    select(replay);
    renderHistory();
    playing = true;
    tickStart();
  } catch (err) {
    setStatus("error", `녹화 시연을 열지 못했어요 (${(err as Error).message}).`);
  }
});

$("run").addEventListener("click", startRun);
async function startRun() {
  if (!info) return;
  if (replayMap) { setStatus("error", "녹화 시연은 현재 배치로 다시 계산할 수 없어요. 서버에 연결한 뒤 지도를 다시 불러와 주세요."); return; }
  if (!police.length || !fugitive) { setStatus("error", "경찰차를 1대 이상, 도주차를 1대 놓아 주세요"); return; }
  if (barricades.length && opts.engine !== "sumo") { setStatus("error", "바리케이드는 SUMO 시뮬레이터에서만 쓸 수 있어요. 시뮬레이터를 SUMO로 바꾸거나 바리케이드를 지워 주세요."); return; }
  const body = {
    map: info.name, police: police.map((p) => [p.lat, p.lng]), fugitive: [fugitive.lat, fugitive.lng],
    barricades: barricades.map((p) => [p.lat, p.lng]),
    carry_barricades: opts.engine === "sumo" && opts.carry,
    policy: opts.policy, engine: opts.engine, profile: opts.profile,
    traffic: opts.engine === "sumo" && opts.profile === "real" && opts.traffic ? Number(opts.traffic) : null,
    ignore_red: opts.engine === "sumo" && opts.profile === "real" && opts.ignoreRed,
  };
  const policyLabel = document.querySelector<HTMLButtonElement>(`#policySeg button[data-v="${opts.policy}"]`)?.textContent ?? opts.policy;
  const cond = opts.engine === "sumo" ? (opts.profile === "real" ? `실제 도로${body.traffic ? (body.traffic < 1 ? "·혼잡" : "·보통") : ""}${body.ignore_red ? "·긴급" : ""}` : "신호 없음") : "";
  const run: Run = { id: ++runSeq, mapInfo: info, setup: null, frames: [], outcome: null, seconds: null, error: null, done: false,
    label: `${policyLabel} · ${ENGINE[opts.engine]}${cond ? ` (${cond})` : ""} · 경찰 ${police.length}대${barricades.length ? ` · 바리케이드 ${barricades.length}` : ""}${body.carry_barricades ? " · 차량 바리케이드" : ""}` };
  runs.unshift(run); select(run); renderHistory();
  $<HTMLButtonElement>("run").disabled = true;
  setStatus("running", opts.engine === "sumo" && body.traffic ? "SUMO 준비 중 · 일반 교통을 먼저 채우고 있어요" : "계산 중…");
  map.removeLayer(placeLayer);
  try {
    const res = await fetch(api("/api/run"), { method: "POST", headers: apiHeaders({ "content-type": "application/json" }), body: JSON.stringify(body) });
    if (!res.ok || !res.body) {
      let detail = "";
      try { const j = await res.json() as { detail?: unknown }; if (typeof j.detail === "string") detail = ` ${j.detail}`; } catch { /* not JSON */ }
      throw new Error(`HTTP ${res.status}${detail}`);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let k: number;
      while ((k = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, k); buf = buf.slice(k + 1);
        if (line.trim()) handle(run, JSON.parse(line));
      }
    }
  } catch (err) {
    run.error = apiError("실행 요청 실패", err);
  }
  run.done = true;
  $<HTMLButtonElement>("run").disabled = false;
  if (run.error) setStatus("error", `실행 실패: ${run.error}`);
  else if (current === run && !playing && cursor >= run.frames.length - 1) finishStatus(run);
  else if (current === run) setStatus("running", "계산 완료 · 재생 중");
  renderHistory();
}

function handle(run: Run, msg: Record<string, unknown>) {
  if (msg.type === "setup") { run.setup = msg as unknown as Setup; if (current === run) { setStatus("running", "추격 진행 중"); } }
  else if (msg.type === "net") { run.net = msg as unknown as NetInfo; if (current === run) { drawCrossings(); redrawFrame(true); } }
  else if (msg.type === "frame") {
    run.frames.push(msg as unknown as Frame);
    if (current === run) {
      $<HTMLInputElement>("scrub").max = String(Math.max(0, run.frames.length - 1));
      if (run.frames.length === 1) { cursor = 0; playing = true; follow = true; fitAction(run.frames[0]); redrawFrame(true); tickStart(); }
      if (!playing && follow && cursor >= run.frames.length - 2) { playing = true; tickStart(); }
    }
  } else if (msg.type === "end") { run.outcome = msg.outcome as string; run.seconds = msg.seconds as number; }
  else if (msg.type === "error") { run.error = msg.message as string; }
}

function finishStatus(run: Run) {
  if (run.error) { setStatus("error", run.error); return; }
  if (!run.outcome) { setStatus("running", "추격 진행 중"); return; }
  const t = Math.round(run.seconds ?? 0);
  const result = run.outcome === "success" ? `${t}초 만에 검거` : run.outcome === "escape" ? `${t}초에 도주차가 지도 밖으로 빠져나감` : `${t}초 동안 검거하지 못함`;
  setStatus(run.outcome, run.recorded ? `녹화 시연 · ${result}` : result);
}

// ---------------------------------------------------------------------------------------------
// playback
let lastTs = 0;
let rafId = 0;
function tickStart() { lastTs = performance.now(); cancelAnimationFrame(rafId); rafId = requestAnimationFrame(tick); $("play").textContent = "❚❚"; }
function tick(ts: number) {
  if (!current || !playing) return;
  if (!current.frames.length) { playing = false; return; }
  const dt = (ts - lastTs) / 1000; lastTs = ts;
  const speed = Number($<HTMLSelectElement>("speed").value);
  const maxT = current.frames.length - 1;
  cursor = Math.max(0, Math.min(maxT, cursor + dt * speed));
  redrawFrame(false);
  if (cursor >= maxT) {
    playing = false; $("play").textContent = "▶";
    if (current.done) { follow = false; finishStatus(current); }
    return;                                      // resumes when more frames stream in
  }
  rafId = requestAnimationFrame(tick);
}
$("play").addEventListener("click", () => {
  if (!current || !current.frames.length) return;
  if (playing) { playing = false; follow = false; $("play").textContent = "▶"; return; }
  if (cursor >= current.frames.length - 1 && current.done) cursor = 0;
  playing = true; follow = true; tickStart();
});
const stepBy = (d: number) => { if (!current) return; playing = false; follow = false; $("play").textContent = "▶"; cursor = Math.max(0, Math.min(current.frames.length - 1, Math.round(cursor) + d)); redrawFrame(true); };
$("back").addEventListener("click", () => stepBy(-1));
$("fwd").addEventListener("click", () => stepBy(1));
$("scrub").addEventListener("input", (e) => { if (!current) return; playing = false; follow = false; $("play").textContent = "▶"; cursor = Number((e.target as HTMLInputElement).value); redrawFrame(true); });
document.addEventListener("keydown", (e) => {
  const tag = (e.target as HTMLElement).tagName;
  if (tag === "INPUT" || tag === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); $("play").click(); }
  if (e.code === "ArrowRight") stepBy(1);
  if (e.code === "ArrowLeft") stepBy(-1);
});

// ---------------------------------------------------------------------------------------------
// drawing
const bitCache = new Map<string, Uint8Array>();
function bits(b64: string) {
  let v = bitCache.get(b64);
  if (!v) { const s = atob(b64); v = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) v[i] = s.charCodeAt(i); if (bitCache.size > 3000) bitCache.clear(); bitCache.set(b64, v); }
  return v;
}
const lerp = (a: LatLon, b: LatLon, u: number): LatLon => [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];
let lastIdx = -1;
let policeMarkers: L.Marker[] = [];
let fugMarker: L.Marker | null = null;

function redrawFrame(force: boolean) {
  if (!current || !current.frames.length || !info) return;
  const fr = current.frames;
  cursor = Math.max(0, Math.min(fr.length - 1, Number.isFinite(cursor) ? cursor : 0));
  const i = Math.floor(cursor);
  const u = Math.min(1, cursor - i);
  const a = fr[i], b = fr[Math.min(fr.length - 1, i + 1)] ?? a;
  // moving vehicles: interpolated every animation frame
  if (policeMarkers.length !== a.p.length || force && !map.hasLayer(vehLayer)) {
    vehLayer.clearLayers();
    policeMarkers = a.p.map((p, k) => L.marker(p, { icon: vehIcon("police", String(k + 1)), interactive: false }).addTo(vehLayer));
    fugMarker = L.marker(a.f, { icon: vehIcon("fugitive"), interactive: false, zIndexOffset: 1000 }).addTo(vehLayer);
  }
  a.p.forEach((p, k) => policeMarkers[k].setLatLng(lerp(p, b.p[k] ?? p, u)));
  fugMarker?.setLatLng(lerp(a.f, b.f, u));
  // per-second layers: only when the second changes
  if (i !== lastIdx || force) {
    lastIdx = i;
    regionLayer.clearLayers();
    const reg = bits(a.reg), col = css("--region");
    info.roads.forEach((r, j) => { if (reg[j >> 3] & (1 << (j & 7))) L.polyline(r, { renderer: canvas, color: col, weight: 6, opacity: 0.45, interactive: false }).addTo(regionLayer); });
    exitLayer.clearLayers();
    const ec = css("--exit");
    a.ox.forEach((q) => L.circleMarker(q, { renderer: canvas, radius: 6, color: css("--panel"), weight: 2, fillColor: ec, fillOpacity: 1, interactive: false }).addTo(exitLayer));
    trailLayer.clearLayers();
    const t0 = Math.max(0, i - 30);
    L.polyline(fr.slice(t0, i + 1).map((q) => q.f), { renderer: canvas, color: css("--fugitive"), weight: 3, opacity: 0.5, interactive: false }).addTo(trailLayer);
    a.p.forEach((_, k) => L.polyline(fr.slice(Math.max(0, i - 15), i + 1).map((q) => q.p[k]), { renderer: canvas, color: css("--police"), weight: 3, opacity: 0.3, interactive: false }).addTo(trailLayer));
    targetLayer.clearLayers();
    const tc = css("--target");
    a.tgt.forEach((t, k) => {
      if (!t) return;
      L.polyline([a.p[k], t], { renderer: canvas, color: tc, weight: 1.5, dashArray: "3 6", interactive: false }).addTo(targetLayer);
      L.rectangle(L.latLng(t).toBounds(16), { renderer: canvas, color: tc, weight: 2, fill: false, interactive: false }).addTo(targetLayer);
    });
    barLayer.clearLayers();
    const bars = a.bars ?? current.setup?.barricades ?? [];
    $("barLegend").hidden = !bars.length;
    bars.forEach((q, k) => barMarker(q, a.bk?.includes(k) ?? false, { interactive: false, keyboard: false }).addTo(barLayer));
    drawSignals(a.sig);
    if (force) drawCrossings();
    $("sigLegend").hidden = !(current.net?.stops.length && a.sig);
    $("crossLegend").hidden = !(current.net?.cross.length);
    bgLayer.clearLayers();
    $("bgLegend").hidden = !a.bg;
    if (a.bg) { const c = css("--bg-car"); a.bg.forEach((q) => L.circleMarker(q, { renderer: canvas, radius: 2.5, color: c, weight: 0, fillOpacity: 0.8, interactive: false }).addTo(bgLayer)); }
    panel(a, i);
  }
  $<HTMLInputElement>("scrub").value = String(cursor);
}

function panel(a: Frame, i: number) {
  if (!current) return;
  $("clock").textContent = `${Math.round(a.t)}초`;
  $("openExits").textContent = String(a.open);
  $("openExits").className = "big " + (a.open === 0 ? "ok" : "");
  const rows = a.p.map((p, k) => {
    const d = L.latLng(p).distanceTo(L.latLng(a.f));
    const tag = a.role[k] === 0 ? '<span class="role pursue">직접 추격</span>' : '<span class="role hold">탈출로 차단</span>';
    const kit = !a.kit?.length ? "" : a.kit[k] ? ' <i class="barr-key" title="바리케이드 보유"></i>' : ' <span class="placed">설치함</span>';
    return `<tr><td class="n"><span class="num">${k + 1}</span></td><td>${tag}${kit}</td><td class="d">${Math.round(d)} m</td></tr>`;
  });
  $("officers").innerHTML = rows.join("");
  spark(i);
}

function spark(i: number) {
  if (!current) return;
  const c = $<HTMLCanvasElement>("spark"), g = c.getContext("2d")!;
  const w = c.width, h = c.height, ys = current.frames.map((f) => f.open), n = ys.length;
  const mx = Math.max(1, ...ys);
  g.clearRect(0, 0, w, h);
  const X = (k: number) => 10 + (w - 20) * k / Math.max(1, n - 1), Y = (v: number) => h - 20 - (h - 34) * v / mx;
  g.fillStyle = css("--muted"); g.font = `20px ${css("--mono")}`;
  g.fillText(String(mx), 12, Y(mx) - 4); g.fillText("0", 12, h - 3);
  g.textAlign = "right"; g.fillText(`${Math.round(current.frames[n - 1].t)}초`, w - 10, h - 3); g.textAlign = "start";
  g.beginPath(); ys.forEach((v, k) => (k ? g.lineTo(X(k), Y(v)) : g.moveTo(X(k), Y(v))));
  g.strokeStyle = css("--region"); g.lineWidth = 3; g.stroke();
  g.strokeStyle = css("--ink"); g.lineWidth = 2; g.beginPath(); g.moveTo(X(i), 6); g.lineTo(X(i), h - 22); g.stroke();
}

// ---------------------------------------------------------------------------------------------
// history
function select(run: Run) {
  replayMap = !!run.recorded;
  if (info !== run.mapInfo) showMap(run.mapInfo);
  if (run.frames.length) fitAction(run.frames[0]);
  current = run; cursor = 0; lastIdx = -1; playing = false; follow = true;
  policeMarkers = []; vehLayer.clearLayers();
  $<HTMLInputElement>("scrub").max = String(Math.max(0, run.frames.length - 1));
  if (run.frames.length) { map.removeLayer(placeLayer); redrawFrame(true); if (run.done) finishStatus(run); }
  renderHistory();
}
function renderHistory() {
  const ol = $("history");
  if (!runs.length) return;
  ol.innerHTML = "";
  for (const r of runs) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-current", String(r === current));
    const res = r.error ? '<span class="bad">오류</span>' : r.outcome ? `<span class="${r.outcome === "success" ? "ok" : "bad"}">${OUTCOME[r.outcome]} ${Math.round(r.seconds ?? 0)}초</span>` : "진행 중";
    b.innerHTML = `<span class="h1">#${r.id} ${r.label}</span><span class="h2">${res}</span>`;
    b.addEventListener("click", () => { select(r); cursor = 0; playing = true; follow = true; tickStart(); });
    li.appendChild(b); ol.appendChild(li);
  }
}

// ---------------------------------------------------------------------------------------------
// server address control
function renderApiHint() {
  const origin = apiBase || "현재 페이지와 같은 주소";
  const secureNote = apiBase && !apiBase.startsWith("https://") ? " · Vercel 배포에서는 HTTPS 주소를 권장해요" : "";
  const tokenNote = apiToken ? " · 토큰은 이 탭 세션에만 저장됨" : " · 토큰이 있으면 X-Demo-Token 헤더로 전송";
  $("apiHint").textContent = `${origin}${secureNote}${tokenNote}`;
}
$<HTMLInputElement>("apiBase").value = apiBase;
$<HTMLInputElement>("apiToken").value = apiToken;
renderApiHint();
$("apiSave").addEventListener("click", () => {
  apiBase = $<HTMLInputElement>("apiBase").value.trim().replace(/\/$/, "");
  apiToken = $<HTMLInputElement>("apiToken").value.trim();
  try { localStorage.setItem("apiBase", apiBase); } catch { /* storage unavailable: keep for this visit */ }
  try {
    if (apiToken) sessionStorage.setItem("apiToken", apiToken);
    else sessionStorage.removeItem("apiToken");
  } catch { /* storage unavailable: keep in memory for this visit */ }
  renderApiHint();
  loadMaps();
});

drawPlacement();
loadMaps();
