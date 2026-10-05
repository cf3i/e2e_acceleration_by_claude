#!/bin/bash
# snapshot a revision for same-GPU A/B: mktree.sh <rev>  -> /mnt/data/u/feic/tmp_ab/<rev>
d=/mnt/data/u/feic/tmp_ab/$1; mkdir -p $d
git archive $1 | tar -x -C $d
[ -d $d/data ] || cp -r data $d/
ln -sfn /mnt/data/u/feic/pjs/playground/optimize_agent_to_the_most/.venv $d/.venv
