"""ImpalaNet(fp32_core=True) keeps the LSTM core + heads in fp32 under bf16 autocast,
while the conv encoder still runs in bf16; fp32_core=False is unchanged."""
import torch

from playtrain_trainers.impala.net import ImpalaNet


def _run(fp32_core):
    torch.manual_seed(0)
    m = ImpalaNet((3, 64, 64), 8, features_dim=256, use_lstm=True, fp32_core=fp32_core)
    dtypes = {}
    m.encoder.register_forward_hook(lambda mod, i, o: dtypes.__setitem__("enc", o.dtype))
    m.core.register_forward_hook(lambda mod, i, o: dtypes.__setitem__("core", o[0].dtype))
    T, B = 5, 2
    inp = {"frame": torch.randint(0, 255, (T, B, 3, 64, 64), dtype=torch.uint8),
           "done": torch.zeros(T, B, dtype=torch.bool), "reward": torch.zeros(T, B),
           "last_action": torch.zeros(T, B, dtype=torch.int64)}
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out, _ = m(inp, m.initial_state(B))
    return dtypes, out


def test_fp32_core_dtypes():
    d, out = _run(True)
    assert d["enc"] == torch.bfloat16
    assert d["core"] == torch.float32
    assert out["policy_logits"].dtype == torch.float32 and out["baseline"].dtype == torch.float32


def test_default_unchanged():
    d, out = _run(False)
    assert d["enc"] == torch.bfloat16 and out["policy_logits"].dtype == torch.bfloat16
