"""Assemble the replay viewer page from exported replay JSON files.

  python viewer/export_sumo.py viewer/sumo_dj.json 6:win 6:both 4:win 8:both 6:lose   # SUMO, Daejeon demo map
  python viewer/export_ring.py viewer/ring.json seoul_zero_shot:6:win busan_zero_shot:6:win  # point-mass sim
  python viewer/build_viewer.py viewer/replay_viewer.html viewer/sumo_dj.json:sumo viewer/ring.json:point

Each input is <json>:<engine> (engine label sumo | point). The page is self-contained HTML.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
out, inputs = sys.argv[1], sys.argv[2:]
data = dict(maps={}, episodes=[])
for item in inputs:
    path, engine = item.rsplit(":", 1)
    d = json.loads(Path(path).read_text())
    for e in d["episodes"]:
        e["engine"] = engine
    data["maps"].update(d["maps"]); data["episodes"] += d["episodes"]
blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
html = (HERE / "viewer_head.html").read_text() + (HERE / "viewer_body.html").read_text().replace("__DATA__", blob)
Path(out).write_text(html)
print(out, len(html))
