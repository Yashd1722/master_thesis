import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(self, ts_len: int = 500, num_classes: int = 4, c_in: int = 1):
        super().__init__()
        from tsai.models.PatchTST import PatchTST as TsaiPatchTST

        self.ts_len = ts_len
        self.num_classes = num_classes
        self.model = TsaiPatchTST(
            c_in=c_in,
            c_out=1,
            seq_len=ts_len,
            pred_dim=num_classes,
            patch_len=16,
            stride=8,
            n_layers=3,
            n_heads=4,
            d_model=32,
            d_ff=128,
            dropout=0.2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)
