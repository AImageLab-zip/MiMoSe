import torch, math
import torch.nn.functional as F

def sliding_window_inference(vol, net, patch_size=(64,64,64), overlap=0.5, num_classes=4):
    # vol: (1,C,D,H,W)
    _, C, D, H, W = vol.shape
    pd, ph, pw = patch_size
    sd = int(pd * (1-overlap)); sh = int(ph * (1-overlap)); sw = int(pw * (1-overlap))
    sd = max(1, sd); sh = max(1, sh); sw = max(1, sw)

    out = torch.zeros((1, num_classes, D, H, W), device=vol.device)
    norm= torch.zeros_like(out)

    for z in range(0, max(D - pd + 1, 1), sd):
        for y in range(0, max(H - ph + 1, 1), sh):
            for x in range(0, max(W - pw + 1, 1), sw):
                patch = vol[..., z:z+pd, y:y+ph, x:x+pw]
                with torch.no_grad():
                    logits = net(patch)  # (1,Cout, pd,ph,pw) upsampled to input size
                out[..., z:z+pd, y:y+ph, x:x+pw] += logits
                norm[..., z:z+pd, y:y+ph, x:x+pw] += 1.0
    norm[norm==0] = 1.0
    return out / norm
