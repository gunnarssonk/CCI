"""The v7 experiments: what is trained on what, what is scored on what, and how it is all queued.

    python -m src.v7.experiments list                          # the ten setups and their names
    python -m src.v7.experiments submit --after 3147731        # queue everything in Slurm (after the build job)
    python -m src.v7.experiments train --setup all6 --model cnn        # one training run (a GPU job does this)
    python -m src.v7.experiments evaluate --setup all6 --model cnn     # score that model (a CPU job does this)
    python -m src.v7.experiments baselines --setup all6                # constant mean, Ridge (and mean per box)
    python -m src.v7.experiments collect                               # one table of all v7 results

Experiments (agreed 2026-11-18), two models each (cnn = SmallCNN, pw = PointWiseModel):
  A  all6          train on all six boxes                      -> scored on the test blocks of all six
  B  no<biome>     leave one biome out, train on four boxes    -> scored on the two boxes of that biome
  C  no<box>       leave one box out, train on five boxes      -> scored on that box
20 training runs, one run each (no seed repeats).

Names of results folders (agreed 2026-11-18): <trained on>_<model>_on_<scored on>_<pile>
  boxes  cs ns nl sws am ep     biomes  boreal temperate tropical     all six  all6
  model  cnn pw ridge mean boxmean
  pile   test (test blocks) or all (all blocks of boxes the model never saw)
"""
import os, re, csv, glob, time, argparse, subprocess

from src.v7.common import BOXES, write_run_record

BOX_CODE = {"central_sweden": "cs", "northern_sweden": "ns", "netherlands": "nl", "southwest_sweden": "sws",
            "amapa_brazil": "am", "east_para_brazil": "ep"}
BIOME_BOXES = {"boreal": ["central_sweden", "northern_sweden"], "temperate": ["netherlands", "southwest_sweden"],
               "tropical": ["amapa_brazil", "east_para_brazil"]}
MODELS = {"cnn": "SmallCNN", "pw": "PointWiseModel"}
BASELINES = {"mean": "ConstantMean", "ridge": "Ridge", "boxmean": "BoxMean"}


def setups():
    all6 = list(BOXES)
    out = {"all6": {"experiment": "A", "train": all6, "score": {"all6": all6}}}
    for biome, bx in BIOME_BOXES.items():
        out[f"no{biome}"] = {"experiment": "B", "train": [b for b in all6 if b not in bx], "score": {biome: bx}}
    for b in all6:
        out[f"no{BOX_CODE[b]}"] = {"experiment": "C", "train": [x for x in all6 if x != b], "score": {BOX_CODE[b]: [b]}}
    return out


def piles_for(setup, boxes):
    """test blocks always; all blocks too when the model never trained on any of these boxes."""
    unseen = not any(b in setup["train"] for b in boxes)
    return ["test", "all"] if unseen else ["test"]


def newest_checkpoint(name, model):
    """checkpoints/v7/<setup>/<model>_<MMDD>_<HHMM>/model.pth, the newest if there are several."""
    pat = re.compile(rf"^{re.escape(model)}_\d{{4}}_\d{{4}}$")
    found = sorted(d for d in glob.glob(os.path.join("checkpoints", "v7", name, f"{model}_*"))
                   if pat.match(os.path.basename(d)) and os.path.exists(os.path.join(d, "model.pth")))
    if not found:
        raise SystemExit(f"no trained {model} found in checkpoints/v7/{name}/")
    return os.path.join(found[-1], "model.pth")


def do_train(name, model):
    from src.v7 import train as T7
    T7.run(exp=name, run_name=model, model_name=MODELS[model], boxes=setups()[name]["train"])


def do_evaluate(name, model, out_root):
    from src.v7 import evaluate as E7
    setup = setups()[name]
    ckpt = newest_checkpoint(name, model)
    for code, boxes in setup["score"].items():
        for pile in piles_for(setup, boxes):
            E7.run(f"{name}_{model}_on_{code}_{pile}", MODELS[model], checkpoint=ckpt, score_boxes=boxes, pile=pile, out_root=out_root)


def do_baselines(name, out_root):
    from src.v7 import evaluate as E7
    setup = setups()[name]
    which = ["mean", "ridge"] + (["boxmean"] if name == "all6" else [])     # mean per box only means something for boxes trained on
    for b in which:
        for code, boxes in setup["score"].items():
            for pile in piles_for(setup, boxes):
                E7.run(f"{name}_{b}_on_{code}_{pile}", BASELINES[b], train_boxes=setup["train"], score_boxes=boxes, pile=pile, out_root=out_root)


def do_collect(out_root):
    """results/v7/results_v7.csv: every number of every evaluation in one table. Rebuilt from the
    results folders each time, so it cannot go out of date. Also the v6-style overview (all_results.csv)."""
    rows, cols = [], None
    for f in sorted(glob.glob(os.path.join(out_root, "*", "by_box_and_range.csv"))):
        with open(f) as fh:
            r = list(csv.DictReader(fh))
        if r:
            cols = cols or list(r[0].keys())
            rows += r
    if not rows:
        print("no evaluations found"); return
    path = os.path.join(out_root, "results_v7.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows from {len({r['name'] for r in rows})} evaluations -> {path}")
    from src.collect_results import collect
    collect(out_root)


def sbatch(script, module_args, dependency=None, extra=()):
    cmd = ["sbatch", "--parsable", *extra]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd += [script, *module_args]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"sbatch failed: {' '.join(cmd)}\n{out.stderr}")
    return out.stdout.strip().split(";")[0], " ".join(cmd)


def do_submit(after, only, out_root):
    """Queue everything. Training starts when the build job has ended well; each scoring job starts
    when its training job has ended well; the table is collected when all scoring jobs have ended."""
    os.makedirs("logs/v7", exist_ok=True)
    first = f"afterok:{after}" if after else None
    jobs, score_ids = [], []
    for name, setup in setups().items():
        if only and setup["experiment"] not in only:
            continue
        jid, cmd = sbatch("setup/jasmin/v7/cpu_job.sbatch", ["src.v7.experiments", "baselines", "--setup", name], first,
                          ["--time=02:00:00", f"--job-name=v7b_{name}"])
        jobs.append({"job": jid, "kind": "baselines", "experiment": setup["experiment"], "setup": name, "model": "mean, ridge", "waits_for": after or "", "command": cmd})
        score_ids.append(jid)
        for model in MODELS:
            tid, cmd = sbatch("setup/jasmin/v7/train_gpu.sbatch", ["src.v7.experiments", "train", "--setup", name, "--model", model], first,
                              [f"--job-name=v7t_{name}_{model}"])
            jobs.append({"job": tid, "kind": "train", "experiment": setup["experiment"], "setup": name, "model": model, "waits_for": after or "", "command": cmd})
            eid, cmd = sbatch("setup/jasmin/v7/cpu_job.sbatch", ["src.v7.experiments", "evaluate", "--setup", name, "--model", model], f"afterok:{tid}",
                              ["--time=02:00:00", f"--job-name=v7e_{name}_{model}"])
            jobs.append({"job": eid, "kind": "evaluate", "experiment": setup["experiment"], "setup": name, "model": model, "waits_for": tid, "command": cmd})
            score_ids.append(eid)
    cid, cmd = sbatch("setup/jasmin/v7/cpu_job.sbatch", ["src.v7.experiments", "collect"], "afterany:" + ":".join(score_ids),
                      ["--time=00:30:00", "--job-name=v7_collect"])
    jobs.append({"job": cid, "kind": "collect", "experiment": "", "setup": "", "model": "", "waits_for": "all scoring jobs", "command": cmd})

    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, f"jobs_submitted_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
        w.writeheader(); w.writerows(jobs)
    for j in jobs:
        print(f"{j['job']:>9s}  {j['kind']:9s} {j['experiment']:1s} {j['setup']:12s} {j['model']:11s} waits for {j['waits_for']}")
    n = {k: sum(j["kind"] == k for j in jobs) for k in ("train", "evaluate", "baselines", "collect")}
    print(f"\nqueued: {n['train']} training, {n['evaluate']} scoring, {n['baselines']} baseline and {n['collect']} collect job(s); list -> {path}")
    write_run_record(out_root, "submit", settings={"after": after, "only": only}, results={"jobs": jobs})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["list", "train", "evaluate", "baselines", "collect", "submit"])
    p.add_argument("--setup", choices=list(setups()))
    p.add_argument("--model", choices=list(MODELS))
    p.add_argument("--after", default=None, help="submit: Slurm job that must end well first (the block build)")
    p.add_argument("--only", default=None, help="submit: only these experiments, e.g. A or A,B")
    p.add_argument("--out_root", default="results/v7")
    a = p.parse_args()
    if a.action == "list":
        for name, s in setups().items():
            scored = ", ".join(f"{code} ({'+'.join(piles_for(s, bx))})" for code, bx in s["score"].items())
            print(f"{s['experiment']}  {name:12s} train on {len(s['train'])} boxes: {', '.join(BOX_CODE[b] for b in s['train']):22s} scored on {scored}")
    elif a.action == "train":
        do_train(a.setup, a.model)
    elif a.action == "evaluate":
        do_evaluate(a.setup, a.model, a.out_root)
    elif a.action == "baselines":
        do_baselines(a.setup, a.out_root)
    elif a.action == "collect":
        do_collect(a.out_root)
    elif a.action == "submit":
        do_submit(a.after, [x.strip() for x in a.only.split(",")] if a.only else None, a.out_root)


if __name__ == "__main__":
    main()
