#!/bin/bash
# A/B on one GPU: alternating revisions. usage: ab.sh <data> <rounds> rev...
data=$1; rounds=$2; shift 2
for i in $(seq $rounds); do for r in "$@"; do
  cd /mnt/data/u/feic/tmp_ab/$r && source .venv/bin/activate
  python bench/run_bench.py --stage ab --engine p3_a1 --runner v3 --exclude-tasks 423 --data $data $EXTRA --out /mnt/data/u/feic/tmp_ab/out_${r}_${data}${TAG}_$i > /mnt/data/u/feic/tmp_ab/log_${r}_${data}${TAG}_$i.txt 2>&1
  python -c "import json; m=json.load(open('/mnt/data/u/feic/tmp_ab/out_${r}_${data}${TAG}_$i/metrics.json')); print('$r $data $i', round(m['makespan_s'],3), m['accuracy'])"
done; done
