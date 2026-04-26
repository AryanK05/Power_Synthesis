"""
Pretrained AIG encoder: DeepGate2 (ICCAD'23) via the `python-deepgate` package.

Outputs a single 256-d vector per AIG (mean-pooled gate-level
hs (structural, 128) ++ hf (functional, 128)).
"""
import torch


class DeepGate2Encoder:
    name = "deepgate2"
    out_dim = 256

    def __init__(self, device="cpu"):
        import deepgate
        self.device = torch.device(device)
        self.model = deepgate.Model()
        self.model.load_pretrained()
        self.model.to(self.device).eval()
        self.parser = deepgate.AigParser()

    @torch.no_grad()
    def encode(self, aig_path):
        graph = self.parser.read_aiger(str(aig_path))
        graph = graph.to(self.device)
        hs, hf = self.model(graph)
        g_struct = hs.mean(dim=0)
        g_func = hf.mean(dim=0)
        return torch.cat([g_struct, g_func], dim=0).cpu()


def build_encoder(device="cpu"):
    enc = DeepGate2Encoder(device=device)
    print(f"[encoder] using DeepGate2 (out_dim={enc.out_dim})")
    return enc
