#!/usr/bin/env python3
"""Perception servers for the code channel (Molmo via vLLM, SAM3, Contact-GraspNet), all from site.yaml.

    python robotwin/hybrid_eval/vision.py status
    python robotwin/hybrid_eval/vision.py start --gpu 7 [--gpu-seg 7] [--molmo-util 0.28]
    python robotwin/hybrid_eval/vision.py stop
launch.py calls ensure_up() before a hybrid arm; servers already listening are left alone.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_eval import config  # noqa: E402

SERVERS = ("molmo", "sam3", "graspnet")


def is_up(port, host="127.0.0.1"):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, int(port))) == 0


def status(site):
    return {name: is_up(site["vision"][name]["port"]) for name in SERVERS}


def _spawn(cmd, cwd, env, log):
    with open(log, "ab") as fh:
        subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)


def start(site, gpu, gpu_seg=None, molmo_util=None, say=print):
    v = site["vision"]
    gpu_seg = gpu if gpu_seg is None else gpu_seg
    up = status(site)
    if not up["molmo"]:
        m = v["molmo"]
        util = molmo_util if molmo_util is not None else m["gpu_util"]
        say(f"starting Molmo :{m['port']} on GPU {gpu} (util {util})")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME=site["hf_cache_vision"],
                   PATH=f"{site['molmo_venv']}/bin:{os.environ.get('PATH', '')}")
        _spawn([f"{site['molmo_venv']}/bin/vllm", "serve", m["model"], "--port", str(m["port"]), "--host", "127.0.0.1",
                "--trust-remote-code", "--max-model-len", "4096", "--max-num-batched-tokens", "4096",
                # The grounding client sends one image at a time. Avoid profiling
                # video inputs and a large concurrent batch that it never uses.
                "--max-num-seqs", str(m.get("max_num_seqs", 1)),
                "--limit-mm-per-prompt", json.dumps(m.get("limit_mm_per_prompt", {"image": 1, "video": 0})),
                "--enforce-eager", "--gpu-memory-utilization", str(util)], Path.home(), env, m["log"])
    # [dataprep] the segmentation/grasp venv and the serving subdir are site-configurable
    # (the original layout hard-coded `.venv-libero/bin/python3` + `cap/serving/`; the
    # cap-x checkout here is `capx/serving/` served by a separate uv venv).
    seg_py = site.get("seg_venv", ".venv-libero") + "/bin/python3"
    serving = site.get("serving_subdir", "capsim/serving")
    # [dataprep] the serving scripts import `capsim.*`; sim_dir (their CWD) is the sim root,
    # but a child process does not get CWD on sys.path, so put it on PYTHONPATH explicitly.
    _pp = f"{site['sim_dir']}:{os.environ.get('PYTHONPATH', '')}"
    if not up["sam3"]:
        m = v["sam3"]
        say(f"starting SAM3 :{m['port']} on GPU {gpu_seg}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_seg), HF_HOME=site["hf_cache_vision"],
                   PYTHONPATH=_pp)
        # HF_TOKEN / the hub's cached login work without a separate plaintext key file.
        # Retain the optional legacy file for sites that already use it.
        token_file = site.get("hf_token_file")
        if not env.get("HF_TOKEN") and token_file and Path(token_file).is_file():
            env["HF_TOKEN"] = Path(token_file).read_text().strip()
        _spawn([seg_py, "-u", f"{serving}/launch_sam3_server.py", "--device", "cuda",
                "--port", str(m["port"]), "--host", "127.0.0.1"], site["sim_dir"], env, m["log"])
    if not up["graspnet"]:
        m = v["graspnet"]
        say(f"starting GraspNet :{m['port']} on GPU {gpu_seg}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_seg), HF_HOME=site["hf_cache_vision"], PYTHONPATH=_pp)
        _spawn([seg_py, "-u", f"{serving}/launch_contact_graspnet_server.py",
                "--port", str(m["port"]), "--host", "127.0.0.1"], site["sim_dir"], env, m["log"])


def wait_ready(site, say=print):
    v = site["vision"]
    deadline = time.time() + int(v.get("ready_timeout_s", 900))
    while time.time() < deadline:
        up = status(site)
        if all(up.values()):
            say("vision stack READY " + " ".join(f":{v[n]['port']}" for n in SERVERS))
            return True
        time.sleep(10)
    say("ABORT: vision stack not ready: " + ", ".join(f"{n}={'up' if u else 'DOWN'}" for n, u in status(site).items())
        + " (logs: " + " ".join(v[n]["log"] for n in SERVERS) + ")")
    return False


def ensure_up(site, gpu, gpu_seg=None, molmo_util=None, say=print):
    start(site, gpu, gpu_seg, molmo_util, say)
    return wait_ready(site, say)


def stop(site, say=print):
    v = site["vision"]
    # the vLLM engine worker is a separate process; it usually dies with the server but not always
    pats = [f"vllm serve {v['molmo']['model']}", "VLLM::EngineCore", "launch_sam3_server", "launch_contact_graspnet_server"]
    for p in pats:
        r = subprocess.run(["pkill", "-u", os.environ.get("USER", ""), "-f", p])
        say(f"pkill -f '{p}': {'killed' if r.returncode == 0 else 'nothing running'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "start", "stop"])
    ap.add_argument("--gpu", type=int, default=None, help="GPU for Molmo (start)")
    ap.add_argument("--gpu-seg", type=int, default=None, help="GPU for SAM3 + GraspNet (default: --gpu)")
    ap.add_argument("--molmo-util", type=float, default=None)
    a = ap.parse_args()
    s = config.site()
    if a.cmd == "status":
        for n, up in status(s).items():
            print(f"{n:9s} :{s['vision'][n]['port']}  {'up' if up else 'down'}")
    elif a.cmd == "start":
        if a.gpu is None:
            sys.exit("--gpu is required for start")
        sys.exit(0 if ensure_up(s, a.gpu, a.gpu_seg, a.molmo_util) else 1)
    else:
        stop(s)


if __name__ == "__main__":
    main()
