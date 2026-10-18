# Branch unidexgrasp_objects-omnireset_dexonomy-eigendexplore_dexonomy

This branch is SimToolReal on the FR3 with the XHand, trained from grasped starts. The grasps come from Dexonomy, an OmniReset-style pipeline places them, and EigenDEXplore drives exploration.

Read `omnireset_dexonomy/README.md` first. It maps the three stages, Dexonomy grasps then reset states then training. It gives the flags, the data paths on our NAS and the command of the run this branch comes from.

**Rules that bite.**
- Training needs `PYTHONPATH=$TREE:$TREE/rl_games`. The patched rl_games lives in this repo, and an installed one silently drops the EigenDEXplore and checkpoint patches.
- The five features on top of the source run are off by default. Each needs its data path and, for two of them, extra obs list entries. The README table lists them.
- `omnireset_dexonomy/dexonomy/` carries only our files for Dexonomy (CC BY-NC 4.0). Clone upstream d561ff9 and apply the diff there.
- The reset states were tested in Isaac on the `fr3-xhand` hand, not the `fr3-xhand-a4hs` hand training uses.
