def curriculum_stage(epoch, milestones):
    # milestones: [e_wt, e_tc, e_et]; return 1..3 stage index
    if epoch < milestones[0]: return 0  # only WT
    if epoch < milestones[1]: return 1  # WT+TC
    return 2                              # WT+TC+ET
