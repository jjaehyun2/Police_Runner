/* 추격 상황 지도 렌더러.
 *
 * SUMO-GUI는 교통공학용이라 차량을 실제 축척으로 그린다. 도시 한 구역
 * 전체가 들어오는 배율에서 5m짜리 승용차는 1픽셀도 되지 않아 화면이
 * 빈 도로지도처럼 보인다. 발표에서 필요한 것은 정확한 축척이 아니라
 * "누가 어디서 무엇을 하는가"이므로, 차량·신호를 과장해서 그린다.
 *
 * 좌표계: network.json은 미터 단위이고 y가 위로 증가한다. 캔버스는
 * y가 아래로 증가하므로 화면 변환에서 한 번만 뒤집는다.
 */
(function () {
  "use strict";

  var C = {
    land: "#16211b",
    roadCasing: "#20272e",
    road: "#59636f",
    roadLine: "rgba(226,201,120,0.55)",
    police: "#3d8bfd",
    policeDark: "#1b4fa0",
    fugitive: "#e5484d",
    fugitiveDark: "#8f2226",
    background: "#8892a0",
    backgroundDark: "#5b6472",
    glass: "rgba(220,240,255,0.75)",
    arrow: "rgba(61,139,253,0.75)",
    red: "#e5484d", yellow: "#e5a13a", green: "#3fbf6a", off: "#3a424c",
    text: "#e6edf3"
  };

  /* 실제 크기가 아니라 "읽히는 크기". 미터 단위로 지정하고 확대율에
   * 따라 하한을 두어, 멀리 봐도 차가 점으로 사라지지 않게 한다. */
  var CAR = { bg: 20, police: 30, fugitive: 34 };
  var MIN_PX = { bg: 13, police: 20, fugitive: 24 };
  var MAX_PX = { bg: 30, police: 42, fugitive: 48 };
  var CAPTURE_RADIUS_M = 25;
  var PERIMETER_M = 250;

  var canvas = null, ctx = null, dpr = 1;
  var net = null;             // {width,height,roads[],signals[]}
  var scene = null;           // {vehicles[],signals[],arrows[],focus}
  var signalPos = {};         // id -> {x,y}
  var grid = null;            // 도로 공간 색인(뷰포트 컬링용)
  var GRID = 400;             // 색인 셀 크기(m)

  var DEFAULT_SCALE = 1.2;
  var cam = { x: 0, y: 0, scale: DEFAULT_SCALE, follow: true, targetX: 0, targetY: 0 };
  var pointer = { down: false, lx: 0, ly: 0 };
  var t0 = performance.now();

  function init(el) {
    canvas = el;
    ctx = canvas.getContext("2d");
    resize();
    bindInput();
    requestAnimationFrame(frame);
  }

  function resize() {
    if (!canvas) return;
    dpr = window.devicePixelRatio || 1;
    var r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
  }

  function bindInput() {
    canvas.addEventListener("wheel", function (e) {
      e.preventDefault();
      var step = Math.max(-40, Math.min(40, e.deltaY));
      var k = Math.pow(1.0022, -step);
      cam.scale = Math.min(4.5, Math.max(0.06, cam.scale * k));
    }, { passive: false });

    canvas.addEventListener("pointerdown", function (e) {
      pointer.down = true; pointer.lx = e.clientX; pointer.ly = e.clientY;
      cam.follow = false; canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", function (e) {
      if (!pointer.down) return;
      cam.x -= (e.clientX - pointer.lx) / cam.scale;
      cam.y += (e.clientY - pointer.ly) / cam.scale;
      pointer.lx = e.clientX; pointer.ly = e.clientY;
    });
    canvas.addEventListener("pointerup", function () { pointer.down = false; });
    canvas.addEventListener("dblclick", function () { cam.follow = true; });
    window.addEventListener("resize", resize);
  }

  function setNetwork(data) {
    if (!data || !data.roads || !data.roads.length) return;
    net = data;
    signalPos = {};
    (data.signals || []).forEach(function (s) { signalPos[s.id] = s; });
    buildGrid();
    if (!scene || !scene.focus) { cam.x = data.width / 2; cam.y = data.height / 2; }
  }

  /* 도로 2600개를 매 프레임 순회하면 확대했을 때 낭비가 크다.
   * 격자 색인으로 화면에 걸치는 셀의 도로만 그린다. */
  function buildGrid() {
    grid = {};
    net.roads.forEach(function (road, index) {
      var seen = {};
      road.pts.forEach(function (p) {
        var key = Math.floor(p[0] / GRID) + ":" + Math.floor(p[1] / GRID);
        if (seen[key]) return;
        seen[key] = 1;
        (grid[key] || (grid[key] = [])).push(index);
      });
    });
  }

  function setScene(data) {
    if (!data || !data.vehicles) return;
    scene = data;
    if (data.focus) { cam.targetX = data.focus[0]; cam.targetY = data.focus[1]; }
  }

  function toScreen(x, y) {
    return [
      (x - cam.x) * cam.scale + canvas.width / (2 * dpr),
      (cam.y - y) * cam.scale + canvas.height / (2 * dpr)
    ];
  }

  function frame(now) {
    requestAnimationFrame(frame);
    if (!ctx) return;
    if (cam.follow && scene && scene.focus) {
      cam.x += (cam.targetX - cam.x) * 0.12;
      cam.y += (cam.targetY - cam.y) * 0.12;
    }
    draw(now);
  }

  function draw(now) {
    var w = canvas.width / dpr, h = canvas.height / dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = C.land;
    ctx.fillRect(0, 0, w, h);

    if (!net) { hint(w, h, "지도 불러오는 중…"); return; }

    var half = { x: w / (2 * cam.scale), y: h / (2 * cam.scale) };
    var view = {
      x0: cam.x - half.x - 60, x1: cam.x + half.x + 60,
      y0: cam.y - half.y - 60, y1: cam.y + half.y + 60
    };

    drawRoads(view);
    drawSignals(view);
    if (scene) {
      drawPerimeter(now);
      drawArrows();
      drawVehicles(now);
    } else {
      hint(w, h, "시뮬레이션 대기 중…");
    }
    drawHud(w, h);
  }

  function visibleRoads(view) {
    var out = {}, list = [];
    for (var gx = Math.floor(view.x0 / GRID); gx <= Math.floor(view.x1 / GRID); gx++) {
      for (var gy = Math.floor(view.y0 / GRID); gy <= Math.floor(view.y1 / GRID); gy++) {
        var cell = grid[gx + ":" + gy];
        if (!cell) continue;
        for (var i = 0; i < cell.length; i++) {
          if (!out[cell[i]]) { out[cell[i]] = 1; list.push(net.roads[cell[i]]); }
        }
      }
    }
    return list;
  }

  function strokeRoad(road, width, colour) {
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.beginPath();
    for (var i = 0; i < road.pts.length; i++) {
      var p = toScreen(road.pts[i][0], road.pts[i][1]);
      if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
    }
    ctx.stroke();
  }

  function drawRoads(view) {
    var roads = visibleRoads(view);
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (var i = 0; i < roads.length; i++) {
      strokeRoad(roads[i], Math.max(2.5, roads[i].w * cam.scale + 3), C.roadCasing);
    }
    for (var j = 0; j < roads.length; j++) {
      strokeRoad(roads[j], Math.max(1.5, roads[j].w * cam.scale), C.road);
    }
    // 간선도로에만 중앙선 — 모든 도로에 그리면 화면이 어지럽다.
    if (cam.scale > 0.35) {
      ctx.setLineDash([7, 9]);
      for (var k = 0; k < roads.length; k++) {
        if (roads[k].w < 6) continue;
        strokeRoad(roads[k], Math.max(0.8, cam.scale * 0.9), C.roadLine);
      }
      ctx.setLineDash([]);
    }
  }

  function drawSignals(view) {
    if (!scene || !scene.signals || cam.scale < 0.5) return;
    var size = Math.max(16, 15 * Math.min(2.2, cam.scale));
    for (var i = 0; i < scene.signals.length; i++) {
      var s = scene.signals[i], pos = signalPos[s.id];
      if (!pos) continue;
      if (pos.x < view.x0 || pos.x > view.x1 || pos.y < view.y0 || pos.y > view.y1) continue;
      var p = toScreen(pos.x, pos.y);
      drawSignalHead(p[0], p[1], size, s.c);
    }
  }

  function drawSignalHead(x, y, size, colour) {
    var w = size * 0.52, h = size * 1.35, r = size * 0.13;
    ctx.fillStyle = "#0f1319";
    ctx.strokeStyle = "#5a6472"; ctx.lineWidth = 1;
    roundRect(x - w / 2, y - h / 2, w, h, 2);
    ctx.fill(); ctx.stroke();
    var lamps = ["red", "yellow", "green"];
    for (var i = 0; i < 3; i++) {
      var cy = y - h / 2 + h * (0.22 + 0.28 * i);
      var on = lamps[i] === colour;
      ctx.beginPath();
      ctx.arc(x, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = on ? C[lamps[i]] : C.off;
      ctx.fill();
      if (on) {
        ctx.shadowColor = C[lamps[i]]; ctx.shadowBlur = 8;
        ctx.fill(); ctx.shadowBlur = 0;
      }
    }
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function fugitiveVehicle() {
    if (!scene) return null;
    for (var i = 0; i < scene.vehicles.length; i++) {
      if (scene.vehicles[i].k === "fugitive") return scene.vehicles[i];
    }
    return null;
  }

  function drawPerimeter(now) {
    var f = fugitiveVehicle();
    if (!f) return;
    var p = toScreen(f.x, f.y);
    var pulse = 0.5 + 0.5 * Math.sin((now - t0) / 420);

    var glow = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], PERIMETER_M * cam.scale);
    glow.addColorStop(0, "rgba(229,72,77,0.20)");
    glow.addColorStop(1, "rgba(229,72,77,0)");
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(p[0], p[1], PERIMETER_M * cam.scale, 0, Math.PI * 2); ctx.fill();

    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = "rgba(229,72,77," + (0.45 + 0.3 * pulse).toFixed(2) + ")";
    ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(p[0], p[1], CAPTURE_RADIUS_M * cam.scale, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = "rgba(229,72,77,0.22)";
    ctx.beginPath(); ctx.arc(p[0], p[1], PERIMETER_M * cam.scale, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawArrows() {
    if (!scene.arrows) return;
    ctx.lineWidth = 2.2;
    ctx.strokeStyle = C.arrow;
    ctx.fillStyle = C.arrow;
    for (var i = 0; i < scene.arrows.length; i++) {
      var a = scene.arrows[i];
      var s = toScreen(a.x1, a.y1), e = toScreen(a.x2, a.y2);
      var dx = e[0] - s[0], dy = e[1] - s[1];
      var len = Math.hypot(dx, dy);
      if (len < 8 || len > 4000) continue;
      // 살짝 휘게: 직선 화살표는 도로와 겹쳐 읽기 어렵다.
      var mx = (s[0] + e[0]) / 2 - dy * 0.16, my = (s[1] + e[1]) / 2 + dx * 0.16;
      ctx.beginPath();
      ctx.moveTo(s[0], s[1]);
      ctx.quadraticCurveTo(mx, my, e[0], e[1]);
      ctx.stroke();
      var ang = Math.atan2(e[1] - my, e[0] - mx);
      ctx.beginPath();
      ctx.moveTo(e[0], e[1]);
      ctx.lineTo(e[0] - 11 * Math.cos(ang - 0.4), e[1] - 11 * Math.sin(ang - 0.4));
      ctx.lineTo(e[0] - 11 * Math.cos(ang + 0.4), e[1] - 11 * Math.sin(ang + 0.4));
      ctx.closePath(); ctx.fill();
      if (cam.scale > 0.3 && a.o) {
        ctx.fillStyle = "rgba(230,237,243,0.85)";
        ctx.font = "600 11px ui-monospace, Consolas, monospace";
        ctx.fillText(a.o, mx + 4, my - 4);
        ctx.fillStyle = C.arrow;
      }
    }
  }

  function drawVehicles(now) {
    var order = { background: 0, police: 1, fugitive: 2 };
    var list = scene.vehicles.slice().sort(function (a, b) { return order[a.k] - order[b.k]; });
    for (var i = 0; i < list.length; i++) drawCar(list[i], now);
  }

  function drawCar(v, now) {
    var p = toScreen(v.x, v.y);
    if (p[0] < -60 || p[1] < -60 || p[0] > canvas.width / dpr + 60 || p[1] > canvas.height / dpr + 60) return;

    var key = v.k === "background" ? "bg" : v.k;
    var L = Math.min(MAX_PX[key], Math.max(MIN_PX[key], CAR[key] * cam.scale));
    var W = L * 0.46;
    var body = v.k === "police" ? C.police : v.k === "fugitive" ? C.fugitive : C.background;
    var dark = v.k === "police" ? C.policeDark : v.k === "fugitive" ? C.fugitiveDark : C.backgroundDark;

    ctx.save();
    ctx.translate(p[0], p[1]);
    ctx.rotate(-v.a * Math.PI / 180);   // 화면 y가 아래로 증가하므로 부호 반전

    if (v.k === "fugitive") {
      var pulse = 0.5 + 0.5 * Math.sin((now - t0) / 300);
      ctx.shadowColor = "rgba(229,72,77," + (0.5 + 0.4 * pulse).toFixed(2) + ")";
      ctx.shadowBlur = 18;
    }
    ctx.fillStyle = body;
    ctx.strokeStyle = dark;
    ctx.lineWidth = Math.max(1, L * 0.07);
    roundRect(-L / 2, -W / 2, L, W, Math.min(4, L * 0.22));
    ctx.fill(); ctx.stroke();
    ctx.shadowBlur = 0;

    if (L > 16) {
      ctx.fillStyle = C.glass;
      roundRect(L * 0.06, -W * 0.32, L * 0.26, W * 0.64, 1.5); ctx.fill();   // 앞유리
      roundRect(-L * 0.34, -W * 0.30, L * 0.20, W * 0.60, 1.5); ctx.fill();  // 뒷유리
    }
    if (v.k === "police" && L > 18) {
      ctx.fillStyle = "#e8462f";
      ctx.fillRect(-L * 0.06, -W * 0.5, L * 0.12, W * 0.24);
      ctx.fillStyle = "#2f6fe8";
      ctx.fillRect(-L * 0.06, W * 0.26, L * 0.12, W * 0.24);
    }
    ctx.restore();

    if (v.id && L > 18) {
      ctx.fillStyle = "rgba(230,237,243,0.92)";
      ctx.font = "700 11px ui-monospace, Consolas, monospace";
      ctx.textAlign = "center";
      ctx.fillText(v.id, p[0], p[1] - L * 0.75 - 3);
      ctx.textAlign = "start";
    }
  }

  function drawHud(w, h) {
    var lines = [];
    lines.push(cam.follow ? "카메라: 도주차량 추적 중" : "카메라: 자유 이동 (더블클릭 시 재추적)");
    if (scene) {
      var bg = 0, po = 0;
      for (var i = 0; i < scene.vehicles.length; i++) {
        if (scene.vehicles[i].k === "background") bg++;
        else if (scene.vehicles[i].k === "police") po++;
      }
      lines.push("배경차량 " + bg + "대 · 경찰 " + po + "대 · 배율 " + cam.scale.toFixed(2) + "x");
    }
    ctx.font = "12px system-ui, 'Malgun Gothic', sans-serif";
    var pad = 8, lh = 17;
    var bw = 0;
    for (var j = 0; j < lines.length; j++) bw = Math.max(bw, ctx.measureText(lines[j]).width);
    ctx.fillStyle = "rgba(10,13,16,0.72)";
    roundRect(12, h - (lines.length * lh + pad * 2) - 12, bw + pad * 2, lines.length * lh + pad * 2, 6);
    ctx.fill();
    ctx.fillStyle = "rgba(230,237,243,0.85)";
    for (var k = 0; k < lines.length; k++) {
      ctx.fillText(lines[k], 12 + pad, h - (lines.length * lh + pad) - 12 + lh * (k + 1) - 2);
    }

    // 범례
    var legend = [["도주차량", C.fugitive], ["경찰차", C.police], ["일반차량", C.background]];
    ctx.font = "12px system-ui, 'Malgun Gothic', sans-serif";
    var lw = 108;
    ctx.fillStyle = "rgba(10,13,16,0.72)";
    roundRect(w - lw - 12, 12, lw, legend.length * 20 + 12, 6); ctx.fill();
    for (var m = 0; m < legend.length; m++) {
      var y = 12 + 18 + m * 20;
      ctx.fillStyle = legend[m][1];
      roundRect(w - lw - 12 + 10, y - 8, 14, 9, 2); ctx.fill();
      ctx.fillStyle = "rgba(230,237,243,0.85)";
      ctx.fillText(legend[m][0], w - lw - 12 + 32, y);
    }
  }

  function hint(w, h, text) {
    ctx.fillStyle = "rgba(230,237,243,0.5)";
    ctx.font = "14px system-ui, 'Malgun Gothic', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(text, w / 2, h / 2);
    ctx.textAlign = "start";
  }

  function resetView() {
    cam.scale = DEFAULT_SCALE;
    cam.follow = true;
    if (scene && scene.focus) { cam.x = scene.focus[0]; cam.y = scene.focus[1]; }
    else if (net) { cam.x = net.width / 2; cam.y = net.height / 2; }
  }

  window.PursuitMap = {
    init: init, setNetwork: setNetwork, setScene: setScene, resize: resize,
    resetView: resetView,
    hasNetwork: function () { return !!net; },
    camera: function () { return { scale: cam.scale, follow: cam.follow, x: cam.x, y: cam.y }; }
  };
})();