#!/usr/bin/env python3
"""Rollout viewer for the RoboTwin hybrid evaluation (simulation only).

    python robotwin/hybrid_eval/rollout_viewer.py build [--results evaluate_results/robotwin] [--out DIR]
        [--tag-prefix hybrid_] [--config hybrid] [--ffmpeg /path/ffmpeg]
    python robotwin/hybrid_eval/rollout_viewer.py serve [--port 8428] [--dir DIR]

`build` writes a static HTML viewer (index.html + view.html + data/ + videos/) from the runs `launch.py` wrote.
Per episode: the rollout video, a channel strip (code vs WAM per frame, from episode{N}_frames.json; reconstructed
from the arbitration log when absent), the generated code of the replan under the playhead (executed / rejected +
reason), and the pure-WAM rollout of the SAME scene when a baseline run exists. A hybrid run's baseline is the
`meta.baseline` tag if set, else the WAM run with the same (model, checkpoint, task, protocol, seed).

Only RoboTwin simulation runs are read: a result directory needs the run_meta.json that launch.py writes, and that
file must describe a sim run (`env: robotwin_sim`, or for runs launched before that field existed, a RoboTwin
`demo_*` protocol). Anything else is skipped and counted.

`serve` is a static file server with HTTP Range support, so the browser can seek inside the mp4s
(`python -m http.server` answers every request with the whole file, which makes the progress bar undraggable).
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # robotwin: the hybrid_eval package
from hybrid_eval import config  # noqa: E402


def _success(d, ep):
    """Outcome from the rollout video's own name (RoboTwin writes episode{N}_<proto>_success-<bool>.mp4)."""
    for x in glob.glob(os.path.join(d, f"episode{ep}_*success-*.mp4")):
        if "_step" not in os.path.basename(x):
            return "success-true" in os.path.basename(x)
    return None


def _rollout_mp4(d, ep):
    c = [x for x in glob.glob(os.path.join(d, f"episode{ep}_*.mp4"))
         if "_step" not in os.path.basename(x) and os.path.getsize(x) > 5_000]   # tiny file = empty pipe
    return c[0] if c else None


def build_episode(d, ep_idx):
    arb = os.path.join(d, f"episode{ep_idx}_arbitration.jsonl")
    if not os.path.exists(arb):
        return None
    R = [json.loads(l) for l in open(arb, errors="replace") if l.strip()]
    st = max((i for i, r in enumerate(R) if "prelude" in r), default=0)
    R = [r for r in R[st:] if "prelude" not in r]
    codes = {}
    cj = os.path.join(d, f"episode{ep_idx}_codes.jsonl")
    if os.path.exists(cj):
        for l in open(cj, errors="replace"):
            try:
                o = json.loads(l); codes[int(o.get("replan", -1))] = o.get("code", "")
            except Exception:
                pass
    fj = os.path.join(d, f"episode{ep_idx}_frames.json")
    fr = json.load(open(fj)) if os.path.exists(fj) else None
    dec = [r for r in R if r.get("channel") == "code" or r.get("reason") in ("no_step_id", "gate_fail")]
    n_dec, n_code_dec = len(dec), sum(1 for r in dec if r.get("channel") == "code")
    # stage coverage: of the program's stages, how many did generated code actually drive?
    n_stages = max((int(r["n_steps"]) for r in R if r.get("n_steps")), default=0)
    # `channel == "code"` is stamped BEFORE _exec_step runs (arbiter.py:427), so a slice that
    # raises immediately still carries it -- counting those overstated the metric (observed:
    # 9 code records, 0 code frames, every one a NameError at wall_s=0.0). Require the step to
    # have actually issued motion, which is what the frame labels and the strip reflect.
    _drove = [r for r in R if r.get("channel") == "code" and r.get("step") is not None
              and int(r.get("n_dense_batches") or 0) > 0]
    _entered = [r for r in R if r.get("channel") == "code" and r.get("step") is not None]
    code_stages = len({int(r["step"]) for r in _drove})
    err_stages = len({int(r["step"]) for r in _entered} - {int(r["step"]) for r in _drove})
    if n_stages:
        code_stages = min(code_stages, n_stages)
    by_rp, ordn = {}, {}
    for r in R:
        rp = int(r.get("replan", -1))
        if r.get("channel") or rp not in by_rp:
            by_rp[rp] = r
        ordn.setdefault(rp, len(ordn) + 1)

    def meta(rp, is_code, a, b):
        r = by_rp.get(rp, {})
        return {"a": a, "b": b, "code": bool(is_code), "replan": rp, "ord": ordn.get(rp, 0), "step": r.get("step"),
                "reason": r.get("reason"), "ncc": (r.get("gate") or {}).get("n_code_commented"),
                "wall": r.get("wall_s"), "src": codes.get(rp, "")}

    if fr and fr.get("replan") and len(fr["replan"]) == len(fr.get("frames", [])):   # frame-exact labels
        chans, rps = fr["frames"], fr["replan"]
        segs, a = [], 0
        for i in range(1, len(chans) + 1):
            if i == len(chans) or (chans[i], rps[i]) != (chans[a], rps[a]):
                segs.append(meta(int(rps[a]), chans[a] == "code", a, i)); a = i
        return {"segs": segs, "total": len(chans), "exact": True, "sim_steps": fr.get("sim_steps", {}),
                "n_dec": n_dec, "n_code_dec": n_code_dec, "n_stages": n_stages, "code_stages": code_stages, "err_stages": err_stages,
                "code_frames": sum(1 for c in chans if c == "code")}
    segs, cursor = [], 0                                                               # reconstructed
    for i, r in enumerate(R):
        is_code = r.get("channel") == "code"
        if is_code:
            n = r.get("n_dense_batches") or 1
        else:
            nxt = R[i + 1].get("take_action_cnt") if i + 1 < len(R) else None
            cur = r.get("take_action_cnt")
            n = (nxt - cur) if (isinstance(nxt, int) and isinstance(cur, int) and nxt > cur) else 24
        m = meta(int(r.get("replan", -1)), is_code, cursor, cursor + n); m["ord"] = i + 1
        segs.append(m); cursor += n
    # The stage fields have to be repeated here: only the frame-exact return above carried them,
    # so every episode without per-frame labels reported 0/0 stages driven by code. That stayed
    # invisible while 1052 of 1053 viewer episodes were frame-exact, then hid the whole metric for
    # arm3final, which is reconstructed for all 2450 episodes (--no-render captures no frames).
    return {"segs": segs, "total": cursor, "exact": False, "n_dec": n_dec, "n_code_dec": n_code_dec,
            "n_stages": n_stages, "code_stages": code_stages, "err_stages": err_stages,
            "code_frames": sum(s["b"] - s["a"] for s in segs if s["code"])}


def remux(ffmpeg, src, dst):
    """Local +faststart copy (browsers otherwise download the whole NFS file before playing); symlink if no ffmpeg."""
    if os.path.exists(dst) and not os.path.islink(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return True
    if os.path.lexists(dst):
        os.remove(dst)
    if ffmpeg and subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", src, "-c:v", "libx264", "-crf", "30",
                                  "-preset", "veryfast", "-movflags", "+faststart", dst]).returncode == 0:
        return True
    os.symlink(os.path.abspath(src), dst)
    return True


def is_sim_run(m):
    """True for a run_meta.json that launch.py wrote for a RoboTwin simulation run."""
    if m.get("env") is not None:
        return m["env"] == "robotwin_sim"
    return (m.get("mode") in ("hybrid", "wam") and str(m.get("proto", "")).startswith("demo_")
            and all(k in m for k in ("tag", "task", "seed")))


def load_runs(results, tag_prefix=None, model=None):
    runs, skipped = [], 0
    for mp in glob.glob(os.path.join(results, "*", "*", "*", "run_meta.json")):
        try:
            m = json.load(open(mp))
        except (OSError, ValueError):
            skipped += 1
            continue
        if not is_sim_run(m):
            skipped += 1
            continue
        m["dir"] = os.path.dirname(mp)
        if tag_prefix and not m["tag"].startswith(tag_prefix) and m.get("mode") == "hybrid":
            continue
        if model and m.get("model") != model:
            continue
        runs.append(m)
    if skipped:
        print(f"skipped {skipped} result dirs that are not RoboTwin simulation runs from launch.py")
    return runs


def pair_baseline(run, runs):
    if run.get("baseline"):
        for r in runs:
            if r["tag"] == run["baseline"] and r["task"] == run["task"] and r["seed"] == run["seed"]:
                return r
    for r in runs:
        if r.get("mode") == "wam" and all(r.get(k) == run.get(k) for k in ("model", "ckpt", "task", "proto", "seed")):
            return r
    return None


INDEX_HTML = """<!doctype html><meta charset="utf-8"><title>hybrid rollouts</title>
<style>
body{font:13px system-ui;margin:16px;background:#fafafa}
table{border-collapse:collapse;font:12px monospace;background:#fff}
th,td{border:1px solid #ddd;padding:4px 10px;text-align:right}
th{background:#f0f0f0}td:first-child,th:first-child{text-align:left}
tr[data-id]{cursor:pointer}tr[data-id]:hover td{background:#eef4ff}
tr.gain td{background:#e8f7ee}tr.loss td{background:#fdeaea}tr.novid{cursor:default;opacity:.55}
select{font:13px system-ui;padding:3px;margin-bottom:10px}.note{color:#666;margin:6px 0 12px}
</style>
<h3>Hybrid rollouts</h3>
<div class=note>1 frame = FRAME_EVERY sim steps; code frame share = fraction of physical time driven by the code channel. Click a row to open the episode.</div>
<label>eval <select id=batch onchange="syncProto();render()"></select></label>
<label style="margin-left:14px">protocol <select id=proto onchange="render()"></select></label>
<div id=tbl></div>
<script>
const EPS=__EPS__;
const batch=document.getElementById('batch'), proto=document.getElementById('proto'), tbl=document.getElementById('tbl');
// one option per EVAL (model, mode) - the run tag embeds the task name, so keying on it
// listed one entry per task instead of one per eval.
// An ablation that reuses a checkpoint (dep-gate A/B, floor-gate re-run) must NOT land in
// that checkpoint's eval bucket - it would inflate the episode count and skew AVERAGE.
// Standard runs are tagged hybrid_<task>_...; anything else is its own eval family, named by
// the tag prefix ahead of the task name (depgate_on, depgate_off, floorgate, ...).
const famOf=e=>{const r=e.run||''; if(r.indexOf('hybrid_')===0) return '';
  const t=e.task||'', i=t?r.indexOf(t):-1; return i>0?r.slice(0,i).replace(/_+$/,''):r;};
const evalKey=e=>{const f=famOf(e);
  return (f?f+' \u00b7 ':'')+(e.model||'?')+(e.mode&&e.mode!=='hybrid'?' ['+e.mode+']':'');};
const EV=[...new Set(EPS.map(evalKey))].sort();
[['ALL','all evals ('+EPS.length+' eps)'],...EV.map(k=>[k,k+' — '+EPS.filter(e=>evalKey(e)===k).length+' eps'])]
  .forEach(([v,t])=>{const o=document.createElement('option');o.value=v;o.textContent=t;batch.appendChild(o);});
// the protocol options depend on the selected eval: an eval that only ran demo_clean must not
// offer "randomized" (picking it would render an empty table). Keep the current choice when the
// new eval still has it, otherwise fall back to ALL.
function syncProto(){
  const r=batch.value, keep=proto.value;
  const pool=EPS.filter(e=>r==='ALL'||evalKey(e)===r);
  const PR=[...new Set(pool.map(e=>e.proto))].sort();
  proto.innerHTML='';
  [['ALL','all ('+pool.length+')'],...PR.map(k=>[k,k+' ('+pool.filter(e=>e.proto===k).length+')'])]
    .forEach(([v,t])=>{const o=document.createElement('option');o.value=v;o.textContent=t;proto.appendChild(o);});
  proto.value=[...proto.options].some(o=>o.value===keep)?keep:'ALL';
  proto.disabled=PR.length<2;
}
const SKEY='rolloutviewer.filters';
function saveFilters(){
  try{ sessionStorage.setItem(SKEY, JSON.stringify({b:batch.value, p:proto.value})); }catch(e){}
}
function restoreFilters(){
  let st=null;
  try{ st=JSON.parse(sessionStorage.getItem(SKEY)||'null'); }catch(e){}
  if(st && [...batch.options].some(o=>o.value===st.b)) batch.value=st.b;
  syncProto();                                   // proto options depend on the eval, so build them first
  if(st && [...proto.options].some(o=>o.value===st.p)) proto.value=st.p;
}
restoreFilters();
const kind=e=>e.ncdec===0?'all-WAM':(e.ncdec===e.ndec?'all-code':'mixed'), R=o=>o===true?'✅':(o===false?'❌':'—');
// Success at the granularity that actually exists. A --no-render run leaves no
// episode*success-*.mp4, so there is no per-episode verdict; the arbitration log does carry a
// check_success, but it is the probe taken at the last CODE replan, not the episode outcome --
// measured on arm3final, 72 of 350 shards disagree with run_meta and 457 of 2450 episodes never
// record one at all. Counting it would have invented outcomes. So when any episode in the set
// lacks a verdict, the whole set falls back to the shard totals in run_meta (7 episodes each),
// which is the only truth those runs have, and the cell is flagged with a dagger.
function succ(L){
  const per=L.filter(e=>e.ok===true||e.ok===false);
  if(per.length===L.length) return {ok:L.filter(e=>e.ok).length, tot:L.length, shard:false};
  const SH={}; L.forEach(e=>{if(e.stot) SH[e.run]=[e.sok,e.stot];});
  const v=Object.values(SH);
  return {ok:v.reduce((a,x)=>a+x[0],0), tot:v.reduce((a,x)=>a+x[1],0), shard:true};
}
const DAG=' <span title="shard-level: this run has no per-episode outcome" style="color:#a86b00">&dagger;</span>';
const succTxt=(s,d)=>s.tot?s.ok+'/'+s.tot+' ('+(100*s.ok/s.tot).toFixed(d)+'%)'+(s.shard?DAG:''):'—';
function render(){
 saveFilters();
 const r=batch.value, pf=proto.value;
 const L=EPS.filter(e=>(r==='ALL'||evalKey(e)===r)&&(pf==='ALL'||e.proto===pf)), groups={};
 if(!L.length){tbl.innerHTML='<div class=note>no episodes for this combination</div>';return;}
 // Rows are keyed by the three columns that identify them -- task, protocol, model --
 // and nothing else. Per-seed tags (..._s1 ... _s7) used to split one eval into seven rows whose
 // percentages each covered a seventh of the episodes; they now pool into one. The key uses
 // evalKey, not a bare model, so an ablation that reuses a checkpoint (depgate_on/off,
 // floorgate) keeps its own row in the ALL view instead of being pooled into that checkpoint's
 // baseline -- and the model cell prints the same family prefix, so each row is keyed by
 // exactly what it displays. Runs that land in one key merge; the run cell then counts them.
 L.forEach(e=>{const g=e.task+'|'+e.proto+'|'+evalKey(e);(groups[g]=groups[g]||[]).push(e);});
 let h='<table style="margin-bottom:10px"><tr><th>task</th><th>protocol</th><th>model</th><th>runs</th><th>eps</th><th>pure-WAM same scenes</th><th>hybrid success</th><th>stages driven by code</th><th>code physical time (eps w/ code)</th><th>code time on rescued (WAM❌→✅)</th></tr>';
 Object.keys(groups).sort().forEach(g=>{const T=groups[g], S=succ(T);
  const bn=T.filter(e=>e.base!==null&&e.base!==undefined).length, bo=T.filter(e=>e.base===true).length;
  const cd=T.reduce((a,e)=>a+e.cstg,0), dd=T.reduce((a,e)=>a+e.nstg,0), ce=T.reduce((a,e)=>a+(e.estg||0),0);
  const C=T.filter(e=>e.codepct>0), cf=C.reduce((a,e)=>a+e.frames*e.codepct/100,0), ff=C.reduce((a,e)=>a+e.frames,0);
  const RS=T.filter(e=>e.base===false&&e.ok), rcf=RS.reduce((a,e)=>a+e.frames*e.codepct/100,0), rff=RS.reduce((a,e)=>a+e.frames,0);
  const RN=[...new Set(T.map(e=>e.run))].sort();
  h+='<tr><td>'+T[0].task+'</td><td>'+T[0].proto+'</td><td>'+evalKey(T[0])+'</td>'+
   '<td title="'+RN.join('&#10;')+'">'+(RN.length===1?RN[0]:RN.length+' runs')+'</td><td>'+T.length+'</td>'+
   '<td>'+(bn?bo+'/'+bn+' ('+(100*bo/bn).toFixed(0)+'%)':'—')+'</td><td>'+succTxt(S,0)+'</td>'+
   '<td>'+cd+'/'+dd+' ('+(100*cd/Math.max(dd,1)).toFixed(1)+'%)</td><td>'+(ff?(100*cf/ff).toFixed(1)+'% ('+C.length+' eps)':'—')+'</td>'+
   '<td>'+(rff?(100*rcf/rff).toFixed(1)+'% ('+RS.length+' eps)':'—')+'</td></tr>';});
 {const S=succ(L);
  const bn=L.filter(e=>e.base!==null&&e.base!==undefined).length, bo=L.filter(e=>e.base===true).length;
  const cd=L.reduce((a,e)=>a+e.cstg,0), dd=L.reduce((a,e)=>a+e.nstg,0), ce=L.reduce((a,e)=>a+(e.estg||0),0);
  const C=L.filter(e=>e.codepct>0), cf=C.reduce((a,e)=>a+e.frames*e.codepct/100,0), ff=C.reduce((a,e)=>a+e.frames,0);
  const RS=L.filter(e=>e.base===false&&e.ok), rcf=RS.reduce((a,e)=>a+e.frames*e.codepct/100,0), rff=RS.reduce((a,e)=>a+e.frames,0);
  const nt=new Set(L.map(e=>e.task)).size;
  // Pooled over ONLY the tasks that scored at least once. The all-task average is
  // dominated by tasks the policy cannot do at all, which hides how it does on the rest.
  const byTask={}; L.forEach(e=>{(byTask[e.task]=byTask[e.task]||[]).push(e);});
  const nzT=Object.keys(byTask).filter(t=>succ(byTask[t]).ok>0);
  const nzE=nzT.reduce((a,t)=>a+succ(byTask[t]).tot,0);
  const nzOk=nzT.reduce((a,t)=>a+succ(byTask[t]).ok,0);
  h+='<tr style="font-weight:600;border-top:2px solid #999;background:#fafafa">'
   +'<td>AVERAGE</td><td>'+(new Set(L.map(e=>e.proto)).size>1?'mixed':L[0].proto)+'</td>'
   +'<td>'+(new Set(L.map(evalKey)).size>1?'mixed':evalKey(L[0]))+'</td>'
   +'<td>'+nt+' tasks / '+new Set(L.map(e=>e.run)).size+' runs</td><td>'+L.length+'</td>'
   +'<td>'+(bn?bo+'/'+bn+' ('+(100*bo/bn).toFixed(0)+'%)':'—')+'</td>'
   +'<td>'+succTxt(S,1)
   +(nzT.length&&nzT.length<nt?'<br><span style="font-weight:400;color:#52514e">nonzero '+nzOk+'/'+nzE+' ('+(100*nzOk/nzE).toFixed(1)+'%, '+nzT.length+' tasks)</span>':'')+'</td>'
   +'<td>'+cd+'/'+dd+' ('+(100*cd/Math.max(dd,1)).toFixed(1)+'%)'+(ce?' <span style="color:#a82c2c">+'+ce+' raised</span>':'')+'</td>'
   +'<td>'+(ff?(100*cf/ff).toFixed(1)+'% ('+C.length+' eps)':'—')+'</td>'
   +'<td>'+(rff?(100*rcf/rff).toFixed(1)+'% ('+RS.length+' eps)':'—')+'</td></tr>';}
 h+='</table>';
 h+='<div class=note style="margin:-6px 0 12px">"stages driven by code" = distinct program stages the generated code actually DROVE (issued motion), over the stage count the program itself declares. +N raised = stages the gate let through but whose code raised before moving anything. A &dagger; on a success cell means that run has no per-episode outcome (it ran --no-render, which writes no success-tagged rollout video), so the number is the sum of its 7-episode shard totals from run_meta rather than a count of episodes -- the episode rows below show a dash for those. AVERAGE is pooled over episodes (sum of successes / sum of episodes), not a mean of the per-task percentages — tasks with unequal episode counts would otherwise be weighted equally. The nonzero line under it pools only the tasks that scored at least once, so it is not diluted by tasks at a flat 0.</div>';
 const diff=L.filter(e=>e.base!==null&&e.base!==undefined&&e.ok!==e.base);
 h+='<ul class=note style="margin:0 0 10px;padding-left:18px"><li><span style="background:#e8f7ee;padding:0 6px">green</span> hybrid rescued (WAM❌→✅): '+diff.filter(e=>e.ok).length+' eps</li>'+
  '<li><span style="background:#fdeaea;padding:0 6px">red</span> hybrid lost (WAM✅→❌): '+diff.filter(e=>!e.ok).length+' eps</li><li>unhighlighted = same outcome on both; greyed = no video</li></ul>';
 h+='<table><tr><th>episode</th><th>task</th><th>run</th><th>stages: code/total</th><th>kind</th><th>frames</th><th>code frame %</th><th>video</th><th>pure WAM</th><th>hybrid</th></tr>';
 L.forEach(e=>{let cls=(e.base===null||e.base===undefined||e.ok===e.base)?'':(e.ok?'gain':'loss'); if(!e.video)cls=(cls+' novid').trim();
  h+='<tr'+(cls?' class="'+cls+'"':'')+(e.video?' data-id="'+e.id+'"':'')+'><td>'+e.id+'</td><td>'+e.task+'</td><td>'+e.run+'</td><td>'+e.cstg+'/'+e.nstg+'</td><td>'+kind(e)+'</td><td>'+e.frames+'</td><td>'+e.codepct+'%</td><td>'+(e.video?'🎬':'—')+'</td><td>'+R(e.base)+'</td><td'+(e.ok===null||e.ok===undefined?' title="no per-episode outcome (--no-render); shard scored '+e.sok+'/'+e.stot+'"':'')+'>'+R(e.ok)+'</td></tr>';});
 tbl.innerHTML=h+'</table>';
 tbl.querySelectorAll('tr[data-id]').forEach(tr=>tr.onclick=()=>location.href='view.html#'+tr.dataset.id);
}
render();
</script>"""

VIEW_HTML = """<!doctype html><meta charset="utf-8"><title>hybrid rollout</title>
<style>
body{font:13px system-ui;margin:12px;background:#fafafa}#wrap{display:flex;gap:14px}#left{flex:0 0 auto}#right{flex:1;min-width:340px}
video{width:640px;background:#000;border-radius:4px}
#strip{position:relative;height:26px;margin-top:6px;border:1px solid #bbb;display:flex;cursor:pointer;background:#fff;border-radius:3px;overflow:hidden}
#strip div{height:100%;box-sizing:border-box;border-right:2px solid #fff;color:#fff;font:10px monospace;text-align:center;line-height:26px;overflow:hidden}
#strip div:last-child{border-right:none}#head{position:absolute;top:-3px;width:2px;height:32px;background:#111}
pre{background:#fff;border:1px solid #ddd;padding:10px;max-height:430px;overflow:auto;white-space:pre-wrap;border-radius:4px;font:12px monospace}
.exec{border-left:5px solid #2b6cb0;background:#f0f6ff}.skip{border-left:5px solid #cbd5e0;color:#666}
.badge{padding:2px 8px;border-radius:3px;color:#fff;font-weight:600}select{font:13px system-ui;padding:3px}a{font-size:12px}
</style>
<h3>Hybrid rollout — <span id=t></span> <a href="index.html" title="returns to the same eval/protocol filter">« index</a> <select id=sel onchange="load(this.value)"></select>
 <span style="font-weight:400;font-size:12px"><span class=badge style="background:#2b6cb0">code</span> <span class=badge style="background:#dd6b20">WAM</span></span></h3>
<div id=sum style="margin:2px 0 8px;font:12px monospace;color:#333"></div>
<div id=wrap><div id=left><video id=v controls></video>
  <div id=rate style="margin-top:4px;font-size:12px">speed: <button data-r="0.5">0.5×</button><button data-r="1">1×</button><button data-r="2">2×</button><button data-r="4">4×</button><button data-r="8">8×</button></div>
  <div id=strip><div id=head></div></div><div id=meta style="margin-top:6px;color:#555"></div></div>
 <div id=right><div id=hdr style="margin-bottom:5px"></div><pre id=code></pre>
  <div id=wamwrap style="margin-top:10px;display:none"><div style="font-size:12px;color:#555;margin-bottom:4px">PURE WAM — same scene, no code channel <span id=wamres></span></div>
   <video id=wv controls muted style="width:420px;max-width:100%;background:#000;border-radius:4px"></video></div></div></div>
<script>
const EPS=__EPS__.filter(e=>e.video); let D=null; const sel=document.getElementById('sel');
EPS.forEach(e=>{const o=document.createElement('option');o.value=e.id;o.textContent=e.id+(e.ok?' ✅':' ❌');sel.appendChild(o);});
const v=document.getElementById('v'), strip=document.getElementById('strip'), head=document.getElementById('head'); let RATE=1;
document.querySelectorAll('#rate button').forEach(b=>{b.onclick=()=>{RATE=parseFloat(b.dataset.r);v.playbackRate=RATE;document.querySelectorAll('#rate button').forEach(x=>x.style.fontWeight=x===b?'700':'400');};});
v.addEventListener('loadedmetadata',()=>{v.playbackRate=RATE;});
const kind=e=>e.ncdec===0?'all-WAM':(e.ncdec===e.ndec?'all-code':'mixed'), R=o=>o===true?'✅':(o===false?'❌':'—');
function load(id){const e=EPS.find(x=>x.id===id), ww=document.getElementById('wamwrap');
 if(e&&e.wam){ww.style.display='';document.getElementById('wv').src='videos/'+id+'_wam.mp4';document.getElementById('wamres').textContent=e.base===true?'(WAM: ✅ success)':(e.base===false?'(WAM: ❌ fail)':'');}else ww.style.display='none';
 if(e)document.getElementById('sum').innerHTML='run <b>'+e.run+'</b> · step: code/dec <b>'+e.cstg+'/'+e.nstg+'</b> · kind <b>'+kind(e)+'</b> · frames <b>'+e.frames+'</b> · code frame % <b>'+e.codepct+'%</b> · alignment <b>'+(e.exact?'frame-exact':'reconstructed')+'</b> · hybrid <b>'+R(e.ok)+'</b> · pure WAM <b>'+R(e.base)+'</b>';
 fetch('data/'+id+'.json').then(r=>{if(!r.ok)throw new Error('HTTP '+r.status+' for data/'+id+'.json');return r.json();}).then(d=>{D=d;v.src='videos/'+id+'.mp4';location.hash=id;document.getElementById('t').textContent=id;
  strip.querySelectorAll('div:not(#head)').forEach(n=>n.remove());
  d.segs.forEach(s=>{const el=document.createElement('div');el.style.width=(100*(s.b-s.a)/d.total)+'%';el.style.background=s.code?'#2b6cb0':'#dd6b20';
   if(s.code&&s.step&&(s.b-s.a)/d.total>0.02)el.textContent='S'+s.step;
   el.title=(s.code?'CODE step'+s.step+' ncc='+s.ncc+' '+s.wall+'s':'WAM ('+(s.reason||'-')+')')+'  · replan #'+s.ord+' (control-step '+s.replan+')';
   el.onclick=()=>{if(v.duration)v.currentTime=v.duration*(s.a+0.5*(s.b-s.a))/d.total;};strip.appendChild(el);});
  const nc=d.segs.filter(s=>s.code).reduce((a,s)=>a+s.b-s.a,0);
  document.getElementById('meta').textContent='frames '+d.total+(d.exact?' (frame-exact)':' (reconstructed)')+' · code '+nc+' ('+(100*nc/d.total).toFixed(1)+'%) · segments '+d.segs.length+(d.sim_steps?' · sim steps '+JSON.stringify(d.sim_steps):'');
  show(0);}).catch(err=>{
   document.getElementById('meta').textContent='';
   document.getElementById('hdr').innerHTML='<span class=badge style="background:#d03b3b">LOAD FAILED</span> '+err;
   document.getElementById('code').textContent=
     'Could not load data/'+id+'.json  --  '+err+'.  '
+'The page shell renders but has no episode data. Most likely the static server or the '
     +'port-forward dropped the request - reload, and if it persists check that the viewer was '
     +'rebuilt after the last eval and that `rollout_viewer.py serve` is still up on the node.';
   console.error('view.html load failed', id, err);
  });}
function show(f){if(!D)return;const s=D.segs.find(s=>f>=s.a&&f<s.b)||D.segs[D.segs.length-1], c=document.getElementById('code'), h=document.getElementById('hdr');
 c.className=s.code?'exec':'skip';
 h.innerHTML=s.code?'<span class=badge style="background:#2b6cb0">EXECUTED</span> replan #'+s.ord+' <small>(control-step '+s.replan+')</small> · plan step '+s.step+' · ncc='+s.ncc+' · '+s.wall+'s'
  :'<span class=badge style="background:#dd6b20">NOT EXECUTED</span> replan #'+s.ord+' <small>(control-step '+s.replan+')</small> · WAM drove · reason: '+(s.reason||'-');
 c.textContent=s.src||'(no generation recorded for this replan)';}
v.addEventListener('timeupdate',()=>{if(!D||!v.duration)return;head.style.left=(100*v.currentTime/v.duration)+'%';show(Math.floor(D.total*v.currentTime/v.duration));});
const want=decodeURIComponent(location.hash.slice(1)), first=EPS.find(e=>e.id===want)||EPS[0]; if(first){sel.value=first.id;load(first.id);}
</script>"""


class RangeHandler(SimpleHTTPRequestHandler):
    # HTTP/1.1 keep-alive: the default 1.0 closes the socket after every
    # response, so a seeking <video> pays a fresh TCP handshake per range
    # request — on a high-latency link that reads as stalls mid-load.
    protocol_version = "HTTP/1.1"

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path) or not os.path.exists(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
        if not m or (not m.group(1) and not m.group(2)):
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
        end = min(int(m.group(2)) if m.group(1) and m.group(2) else size - 1, size - 1)
        if start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_range_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            chunk = source.read(min(64 * 1024, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)

    def end_headers(self):
        if not hasattr(self, "_range_left"):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


def build(a):
    site = _site()
    out = a.out or _default_out(site)
    ffmpeg = a.ffmpeg or site.get("ffmpeg") or "ffmpeg"
    ffmpeg = ffmpeg if os.path.isabs(ffmpeg) else shutil.which(ffmpeg)
    if not ffmpeg:
        print("ffmpeg not found: videos will be symlinked (no +faststart remux)")
    os.makedirs(os.path.join(out, "data"), exist_ok=True); os.makedirs(os.path.join(out, "videos"), exist_ok=True)
    runs = load_runs(a.results, a.tag_prefix, a.model)
    hybrids = [r for r in runs if r.get("mode") == "hybrid"]
    eps = []
    for run in hybrids:
        d, base = run["dir"], pair_baseline(run, runs)
        for f in sorted(glob.glob(os.path.join(d, "episode*_arbitration.jsonl"))):
            i = int(re.search(r"episode(\d+)_", f).group(1))
            ok = _success(d, i)
            # A --no-render run writes no episode*success-*.mp4, so _success cannot see a verdict.
            # Dropping those episodes hid whole evals from the viewer (arm3final: 2450 of them).
            # Keep them once the shard itself has finished -- their arbitration artifacts are the
            # point -- and carry the shard total so the table can report success per shard.
            # Still-rolling shards keep being skipped: there, None really does mean "no result yet".
            if ok is None and not run.get("finished"):
                continue                                        # still rolling
            data = build_episode(d, i)
            if not data:
                continue
            k = f"{run['tag']}_e{i}"
            json.dump(data, open(os.path.join(out, "data", f"{k}.json"), "w"))
            src = _rollout_mp4(d, i)
            if src:
                remux(ffmpeg, src, os.path.join(out, "videos", f"{k}.mp4"))
            base_ok, wam_video = None, False
            if base:
                base_ok = _success(base["dir"], i)
                bsrc = _rollout_mp4(base["dir"], i)
                if bsrc and src:
                    wam_video = remux(ffmpeg, bsrc, os.path.join(out, "videos", f"{k}_wam.mp4"))
            res = run.get("result") or {}
            eps.append({"id": k, "ok": None if ok is None else bool(ok),
                        "sok": res.get("success"), "stot": res.get("total"),
                        "run": run["tag"], "video": bool(src), "wam": wam_video,
                        "mode": run.get("mode", "hybrid"),
                        "task": run["task"], "proto": run["proto"].replace("demo_", ""), "model": run.get("model"),
                        "seed": run["seed"], "ep": i, "base": base_ok,
                        "ndec": data["n_dec"], "ncdec": data["n_code_dec"], "frames": data["total"],
                        "nstg": data.get("n_stages", 0), "cstg": data.get("code_stages", 0),
                        "estg": data.get("err_stages", 0),
                        "codepct": round(100.0 * data["code_frames"] / max(data["total"], 1), 1),
                        "exact": bool(data.get("exact"))})
    fe = next((r["hybrid"]["frame_every"] for r in hybrids if r.get("hybrid")), 15)
    blob = json.dumps(eps)
    idx = INDEX_HTML.replace("__EPS__", blob).replace("FRAME_EVERY", str(fe))
    vw = VIEW_HTML.replace("__EPS__", blob)
    _check_js(idx, "index.html")
    _check_js(vw, "view.html")
    open(os.path.join(out, "index.html"), "w").write(idx)
    open(os.path.join(out, "view.html"), "w").write(vw)
    print(f"wrote {out}/index.html + view.html — {len(eps)} episodes from {len(hybrids)} hybrid runs "
          f"({sum(1 for r in runs if r.get('mode') == 'wam')} WAM runs available as baselines)")


def _site():
    """site.yaml supplies only defaults here (output dir, ffmpeg), so a missing file is not an error."""
    try:
        return config.site()
    except FileNotFoundError:
        return {}


def _default_out(site):
    """site.yaml `viewer_out` (`board_out` is the older name of the same key), else <repo>/runs/rollout_viewer."""
    return site.get("viewer_out") or site.get("board_out") or str(config.REPO / "runs" / "rollout_viewer")


def serve(a):
    root = a.dir or _default_out(_site())
    os.chdir(root)
    print(f"serving {root} on http://0.0.0.0:{a.port}/index.html")
    ThreadingHTTPServer(("0.0.0.0", a.port), RangeHandler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the HTML viewer from launch.py results")
    b.add_argument("--results", default=str(config.REPO / "evaluate_results" / "robotwin"))
    b.add_argument("--out", default=None, help="default: site.yaml viewer_out")
    b.add_argument("--ffmpeg", default=None, help="default: site.yaml ffmpeg (name on PATH or absolute path)")
    b.add_argument("--tag-prefix", default=None, help="only hybrid runs whose tag starts with this")
    b.add_argument("--config", dest="model", default=None)
    s = sub.add_parser("serve", help="serve a built viewer with HTTP Range (mp4 seeking)")
    s.add_argument("--port", type=int, default=8428)
    s.add_argument("--dir", default=None, help="default: site.yaml viewer_out")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else serve(a)


def _check_js(html, where):
    """Refuse to write a page whose <script> has an odd number of unescaped single quotes on any
    line. A bare apostrophe inside a '...' literal terminates it early and the entire script fails
    to parse -- the page then renders as an empty shell with no console-visible clue server-side."""
    i = html.find("<script>")
    while i != -1:
        j = html.index("</script>", i)
        for ln, line in enumerate(html[i + 8:j].splitlines(), 1):
            if line.lstrip().startswith("//"):
                continue
            n = len(re.findall(r"(?<!\\)'", line))
            if n % 2:
                raise SystemExit(
                    f"{where}: unbalanced single quotes on script line {ln} -- an unescaped "
                    f"apostrophe would break the whole page.\n  {line[:160]}")
        i = html.find("<script>", j)


if __name__ == "__main__":
    main()
