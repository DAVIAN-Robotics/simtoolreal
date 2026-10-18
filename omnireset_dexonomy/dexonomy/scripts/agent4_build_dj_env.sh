#!/bin/bash
# Build the conda env "dexonomy" on Daejeon nas4, package for package equal to yj-4 bk_dexonomy, plus the
# Dexonomy clone at d561ff9 with the xhand port, installed editable from that clone.
# Run on a Daejeon host (kaist140). Spec files come from yj-4 work/_graspoc/dexonomy/djenv.
set -euxo pipefail
D=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy
CONDA=/home/nas4_user/sibeenkim/anaconda3
ENV=$CONDA/envs/dexonomy
YJ="ssh -p 10004 -o StrictHostKeyChecking=no -o ConnectTimeout=20"
YJW=sibeenkim@<yj host>:/home/nas5/sibeenkim/work/_graspoc/dexonomy
test ! -e $ENV

# Dexonomy clone from yj-4: the git checkout at d561ff9 plus the untracked xhand port, without run outputs
rsync -a -e "$YJ" --exclude /output/ --exclude __pycache__/ --exclude '*.pyc' --exclude '/agent4_*.py' \
  --exclude /debug.xml --exclude /MUJOCO_LOG.TXT $YJW/Dexonomy/ $D/Dexonomy/
git -C $D/Dexonomy log --oneline -1
git -C $D/Dexonomy status --short

# conda layer, exact package URLs
$CONDA/bin/conda create -y -p $ENV --file $D/djenv/bk_dexonomy_conda_explicit.txt

# pip layer, exact pins, no resolver additions; the editable dexonomy line is replaced by this clone
grep -v '^-e ' $D/djenv/bk_dexonomy_pip_freeze.txt > $D/djenv/pip_requirements_pinned.txt
$ENV/bin/pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu128 -r $D/djenv/pip_requirements_pinned.txt
$ENV/bin/pip install --no-deps -e $D/Dexonomy

# record, then compare with yj-4
$ENV/bin/pip freeze > $D/djenv/dexonomy_pip_freeze.txt
$CONDA/bin/conda list -p $ENV > $D/djenv/dexonomy_conda_list.txt
set +e
$ENV/bin/pip check > $D/djenv/dexonomy_pip_check.txt 2>&1
echo "pip check exit $?"
diff <(grep -v '^-e ' $D/djenv/bk_dexonomy_pip_freeze.txt | sort) <(grep -v '^-e ' $D/djenv/dexonomy_pip_freeze.txt | sort)
echo "pip freeze diff exit $? (0 = identical apart from the editable line)"
set -e
grep '^-e ' $D/djenv/dexonomy_pip_freeze.txt
cd /tmp && $ENV/bin/python -c "import dexonomy, dexonomy.main, dexonomy.sim, dexonomy.op.gen_grasp_admm as g; print('path', list(dexonomy.__path__)); print(dexonomy.sim.__file__); print(g.__file__)"
echo ENV_DONE
