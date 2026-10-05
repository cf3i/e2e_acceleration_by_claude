#!/bin/bash
# same-tree A/B over env settings: abenv.sh <data> <replay dir> <rounds> "ENV=a" "ENV=b" ...
data=$1; rp=$2; rounds=$3; shift 3
for i in $(seq $rounds); do for e in "$@"; do
  tag=$(echo $e | tr ' =' '_-')
  out=/mnt/data/u/feic/tmp_ab/env_${tag}_${data}_$i
  env $e python bench/run_bench.py --stage ab --engine p3_a1 --runner v3 --exclude-tasks 423 --data $data --replay $rp --out $out > $out.log 2>&1
  python -c "import json; m=json.load(open('$out/metrics.json')); print('$e $data $i', round(m['makespan_s'],3), m['accuracy'])"
done; done
