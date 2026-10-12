import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./style.css";

// ---------------------------------------------------------------------------------------------
// types
type LatLon = [number, number];
interface MapInfo { name: string; label: string; roads: LatLon[][]; exits: LatLon[]; bounds: [LatLon, LatLon] }
interface Frame { t: number; p: LatLon[]; f: LatLon; role: number[]; tgt: (LatLon | null)[]; reg: string; open: number; ox: LatLon[]; bg?: LatLon[] }
interface Setup { police: LatLon[]; fugitive: LatLon; policy: string; engine: string; profile: string; traffic: number | null }
interface Run { id: number; mapInfo: MapInfo; setup: Setup | null; frames: Frame[]; outcome: string | null; seconds: number | null; error: string | null; label: string; done: boolean; recorded?: boolean }

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const OUTCOME: Record<string, string> = { success: "검거", escape: "탈출", timeout: "시간초과" };
const ENGINE: Record<string, string> = { sumo: "SUMO", point: "빠른 시뮬레이터" };

// ---------------------------------------------------------------------------------------------
// server address: build-time VITE_API_BASE, overridable in the panel (kept in this browser)
function readStored(): string | null { try { return localStorage.getItem("apiBase"); } catch { return null; } }
function readSessionToken(): string { try { return sessionStorage.getItem("apiToken") ?? ""; } catch { return ""; } }
let apiBase = (readStored() ?? (import.meta.env.VITE_API_BASE as string | undefined) ?? "").replace(/\/$/, "");
let apiToken = readSessionToken();
const api = (path: string) => `${apiBase}${path}`;
function apiHeaders(base: HeadersInit = {}) {
  const h = new Headers(base);
  if (apiToken) h.set("X-Demo-Token", apiToken);
  return h;
}
function apiError(context: string, err: unknown) {
  const msg = err instanceof Error ? err.message : String(err);
  const auth = msg.includes("401") ? " 접근 토큰이 맞는지 확인하세요." : "";
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
const vehLayer = L.layerGroup().addTo(map);

const vehIcon = (cls: string, text = "") => L.divIcon({ className: "", html: `<div class="veh ${cls}">${text}</div>`, iconSize: [22, 22], iconAnchor: [11, 11] });

// ---------------------------------------------------------------------------------------------
// state
let info: MapInfo | null = null;
let replayMap = false;
let mode: "police" | "fugitive" = "police";
let police: L.LatLng[] = [];
let fugitive: L.LatLng | null = null;
const opts = { policy: "ga", engine: "sumo", profile: "ideal", traffic: "" as string, ignoreRed: false };
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
function syncOpts() {
  $("sumoOpts").hidden = opts.engine !== "sumo";
  $("realOpts").hidden = !(opts.engine === "sumo" && opts.profile === "real");
}
syncOpts();

function setMode(m: "police" | "fugitive") {
  mode = m;
  $("modePolice").setAttribute("aria-pressed", String(m === "police"));
  $("modeFugitive").setAttribute("aria-pressed", String(m === "fugitive"));
}
$("modePolice").addEventListener("click", () => setMode("police"));
$("modeFugitive").addEventListener("click", () => setMode("fugitive"));

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
  $("counts").textContent = `경찰차 ${police.length}대 · 도주차 ${fugitive ? 1 : 0}대`;
}
map.on("click", (e: L.LeafletMouseEvent) => {
  if (replayMap || current && !current.done) return;
  showPlacement();
  if (mode === "police") { if (police.length < 12) police.push(e.latlng); }
  else fugitive = e.latlng;
  drawPlacement();
});
$("clear").addEventListener("click", () => {
  if (replayMap) return;
  police = []; fugitive = null; showPlacement(); drawPlacement();
});
$("example").addEventListener("click", async () => {
  if (!info) return;
  if (replayMap) { setStatus("error", "녹화 시연에서는 새 배치를 만들 수 없습니다. 서버 주소를 확인하고 지도를 다시 불러오세요."); return; }
  const n = Math.max(2, Math.min(12, Number($<HTMLInputElement>("exampleN").value) || 6));
  try {
    const r = await fetch(api("/api/example"), { method: "POST", headers: apiHeaders({ "content-type": "application/json" }),
      body: JSON.stringify({ map: info.name, n, seed: Math.floor(Math.random() * 1000) }) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json() as { police: LatLon[]; fugitive: LatLon };
    police = d.police.map((p) => L.latLng(p[0], p[1]));
    fugitive = L.latLng(d.fugitive[0], d.fugitive[1]);
    showPlacement(); drawPlacement();
  } catch (err) { setStatus("error", apiError("예시 배치를 불러오지 못했습니다", err)); }
});

function showPlacement() {
  // leaving a finished run's view: back to editing the placement
  playing = false;
  follow = false;
  cancelAnimationFrame(rafId);
  $("play").textContent = "▶";
  regionLayer.clearLayers(); bgLayer.clearLayers(); trailLayer.clearLayers(); targetLayer.clearLayers(); vehLayer.clearLayers(); exitLayer.clearLayers();
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
    if (!list.length) throw new Error("지도 목록이 비어 있습니다");
    sel.innerHTML = list.map((m) => `<option value="${m.name}">${m.label}</option>`).join("");
    sel.onchange = () => loadMap(sel.value).catch((err) => setStatus("error", apiError("지도를 불러오지 못했습니다", err)));
    await loadMap(list[0].name);
    setStatus("idle", "지도를 클릭해 경찰차와 도주차를 배치하세요");
  } catch (err) {
    setStatus("error", `${apiError("서버에 연결하지 못했습니다", err)} 녹화된 SUMO 시연은 재생할 수 있습니다.`);
  }
}
async function loadMap(name: string) {
  const r = await fetch(api(`/api/map/${name}`), { headers: apiHeaders() });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  replayMap = false;
  showMap(await r.json() as MapInfo);
  police = []; fugitive = null; showPlacement(); drawPlacement();
}

// ---------------------------------------------------------------------------------------------
// running
function setStatus(state: string, text: string) { const s = $("status"); s.dataset.state = state; s.textContent = text; }

$("fallback").addEventListener("click", async () => {
  try {
    const res = await fetch("/fallback.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const saved = await res.json() as { map: MapInfo; setup: Setup; frames: Frame[]; outcome: string; seconds: number };
    if (!saved.frames.length) throw new Error("녹화 프레임이 없습니다");
    replayMap = true;
    showMap(saved.map);
    police = saved.setup.police.map((p) => L.latLng(p[0], p[1]));
    fugitive = L.latLng(saved.setup.fugitive[0], saved.setup.fugitive[1]);
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
    setStatus("error", `녹화 시연을 열지 못했습니다 (${(err as Error).message}).`);
  }
});

$("run").addEventListener("click", startRun);
async function startRun() {
  if (!info) return;
  if (replayMap) { setStatus("error", "녹화 시연은 현재 배치로 다시 계산할 수 없습니다. 서버에 연결한 뒤 지도를 다시 불러오세요."); return; }
  if (!police.length || !fugitive) { setStatus("error", "경찰차를 1대 이상, 도주차를 1대 놓아 주세요"); return; }
  const body = {
    map: info.name, police: police.map((p) => [p.lat, p.lng]), fugitive: [fugitive.lat, fugitive.lng],
    policy: opts.policy, engine: opts.engine, profile: opts.profile,
    traffic: opts.engine === "sumo" && opts.profile === "real" && opts.traffic ? Number(opts.traffic) : null,
    ignore_red: opts.engine === "sumo" && opts.profile === "real" && opts.ignoreRed,
  };
  const policyLabel = document.querySelector<HTMLButtonElement>(`#policySeg button[data-v="${opts.policy}"]`)?.textContent ?? opts.policy;
  const cond = opts.engine === "sumo" ? (opts.profile === "real" ? `실제 도로${body.traffic ? (body.traffic < 1 ? "·혼잡" : "·보통") : ""}${body.ignore_red ? "·긴급" : ""}` : "신호 없음") : "";
  const run: Run = { id: ++runSeq, mapInfo: info, setup: null, frames: [], outcome: null, seconds: null, error: null, done: false,
    label: `${policyLabel} · ${ENGINE[opts.engine]}${cond ? ` (${cond})` : ""} · 경찰 ${police.length}대` };
  runs.unshift(run); select(run); renderHistory();
  $<HTMLButtonElement>("run").disabled = true;
  setStatus("running", opts.engine === "sumo" && body.traffic ? "SUMO 준비 중 · 일반 교통을 먼저 채웁니다" : "계산 중…");
  map.removeLayer(placeLayer);
  try {
    const res = await fetch(api("/api/run"), { method: "POST", headers: apiHeaders({ "content-type": "application/json" }), body: JSON.stringify(body) });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
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
    return `<tr><td class="n"><span class="num">${k + 1}</span></td><td>${tag}</td><td class="d">${Math.round(d)} m</td></tr>`;
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
  const secureNote = apiBase && !apiBase.startsWith("https://") ? " · Vercel 배포에서는 HTTPS 주소를 권장합니다" : "";
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
