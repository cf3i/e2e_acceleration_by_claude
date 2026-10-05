#!/bin/bash
# real (non-replay) accuracy over env settings: accenv.sh <data> "ENV=a" "ENV=b" ...
data=$1; shift
for e in "$@"; do
  tag=$(echo $e | tr ' =' '_-')
  out=/mnt/data/u/feic/tmp_ab/acc_${tag}_${data}
  env $e python bench/run_bench.py --stage ab --engine p3_a1 --runner v3 --exclude-tasks 423 --data $data --out $out > $out.log 2>&1
  python -c "import json; m=json.load(open('$out/metrics.json')); print('$e $data', round(m['makespan_s'],3), m['accuracy'])"
done
