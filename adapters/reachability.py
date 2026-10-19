"""Immediate exact OBB separation, with the approved mesh-derived bounds."""
import torch


@torch.jit.script
def quaternion_matrix(q: torch.Tensor) -> torch.Tensor:
    # Isaac uses wxyz. Normalization avoids accumulated simulator quaternion drift.
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)
    ), -1).reshape(-1, 3, 3)


@torch.jit.script
def boxes_disjoint(hand_pos: torch.Tensor, hand_rot: torch.Tensor,
                   object_pos: torch.Tensor, object_rot: torch.Tensor,
                   hand_center: torch.Tensor, hand_half: torch.Tensor,
                   object_center: torch.Tensor, object_half: torch.Tensor) -> torch.Tensor:
    hc = hand_pos + torch.matmul(hand_rot, hand_center)
    oc = object_pos + torch.matmul(object_rot, object_center)
    ht = hand_rot.transpose(1, 2)
    r = torch.bmm(ht, object_rot)
    d = torch.bmm(ht, (oc-hc).unsqueeze(-1)).squeeze(-1)
    ar = r.abs()
    outside = (d.abs() > hand_half + torch.matmul(ar, object_half)).any(-1)
    outside |= (torch.bmm(d.unsqueeze(1), r).squeeze(1).abs()
                > object_half + torch.matmul(ar.transpose(1, 2), hand_half)).any(-1)
    # The nine cross axes need no normalization; zero/parallel axes cannot
    # separate. No numerical padding is added to either box.
    for i in range(3):
        i1, i2 = (i+1) % 3, (i+2) % 3
        for j in range(3):
            j1, j2 = (j+1) % 3, (j+2) % 3
            distance = (d[:, i2]*r[:, i1, j] - d[:, i1]*r[:, i2, j]).abs()
            radii = (hand_half[i1]*ar[:, i2, j] + hand_half[i2]*ar[:, i1, j]
                     + object_half[j1]*ar[:, i, j2] + object_half[j2]*ar[:, i, j1])
            outside |= distance > radii
    return outside
