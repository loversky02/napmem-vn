#!/usr/bin/env bash
# Drive the whole GRPO money plot on a fresh RunPod GPU pod, end to end.
#   MODEL=Qwen/Qwen2.5-3B-Instruct STEPS=50 bash scripts/runpod_bootstrap.sh
# Stages echo markers (SMOKE_OK / ALL_DONE) so a poller can track progress.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
STEPS="${STEPS:-50}"
PY="${PY:-python}"
echo "MODEL=$MODEL STEPS=$STEPS"

echo "=== [1/6] install pinned deps ==="
pip install -q "transformers==4.51.*" "trl==0.17.*" "peft>=0.11,<0.15" "datasets>=2.19" "accelerate>=0.34" \
  || { echo "PINNED_INSTALL_FAILED, trying loose"; pip install -q transformers trl peft datasets accelerate; }
$PY -c "import torch,trl,peft,transformers as t; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'trl',trl.__version__,'tf',t.__version__)" || { echo "IMPORT_FAILED"; exit 1; }

echo "=== [2/6] offline reward smoke ($0 sanity) ==="
$PY scripts/grpo_reward_smoke.py --artifacts results || { echo "REWARD_SMOKE_FAILED"; exit 1; }

echo "=== [3/6] GPU smoke: 3 GRPO steps (catches OOM / API drift cheaply) ==="
$PY scripts/train_grpo.py --model "$MODEL" --out runs/smoke --max-steps 3 --group-size 4 || { echo "GPU_SMOKE_FAILED"; exit 1; }
echo "SMOKE_OK"

echo "=== [4/6] train FCU (F+C+U) and FC (F+C ablation) ==="
$PY scripts/train_grpo.py --model "$MODEL" --out runs/fcu --max-steps "$STEPS" --group-size 4 || { echo "TRAIN_FCU_FAILED"; exit 1; }
$PY scripts/train_grpo.py --model "$MODEL" --out runs/fc  --max-steps "$STEPS" --group-size 4 --no-usage-bonus || { echo "TRAIN_FC_FAILED"; exit 1; }

echo "=== [5/6] eval base + both checkpoints ==="
$PY scripts/eval_grpo.py --base "$MODEL" --label base --out results/grpo_eval_base.json || echo "EVAL_BASE_FAILED"
$PY scripts/eval_grpo.py --base "$MODEL" --adapter runs/fcu --label fcu --out results/grpo_eval_fcu.json || echo "EVAL_FCU_FAILED"
$PY scripts/eval_grpo.py --base "$MODEL" --adapter runs/fc  --label fc  --out results/grpo_eval_fc.json  || echo "EVAL_FC_FAILED"

echo "=== [6/6] money plot ==="
$PY - <<'PY'
import json, os
def m(p):
    try: return json.load(open(p))["metrics"]
    except Exception: return {}
base=m("results/grpo_eval_base.json"); fcu=m("results/grpo_eval_fcu.json"); fc=m("results/grpo_eval_fc.json")
rows=[("base",base),("fcu (F+C+U)",fcu),("fc (F+C)",fc)]
print("checkpoint     acc   mem_recall  unnecessary_calls  malformed")
for name,x in rows:
    if x: print(f"{name:<13}  {x.get('accuracy',0):.2f}  {x.get('memory_recall',0):.2f}       {x.get('unnecessary_memory_call_rate',0):.2f}             {x.get('malformed_rate',0):.2f}")
out={"model":os.environ.get("MODEL"),"steps":os.environ.get("STEPS"),"base":base,"fcu":fcu,"fc":fc}
if fcu and fc:
    out["delta_unnecessary_fcu_minus_fc"]=fcu.get("unnecessary_memory_call_rate",0)-fc.get("unnecessary_memory_call_rate",0)
    print("\ndelta unnecessary (fcu - fc):", round(out["delta_unnecessary_fcu_minus_fc"],3), "(prediction: >= 0, U rewards tool-spam)")
json.dump(out, open("results/grpo_money_plot.json","w"), indent=2)
print("wrote results/grpo_money_plot.json")
PY
echo "ALL_DONE"
