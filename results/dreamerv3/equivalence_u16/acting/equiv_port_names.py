import re


def official_name(port: str) -> str | None:
    """Port state_dict key -> official ninjax key. None for keys that are aliases."""
    if port.startswith(("opt_modules.", "slowval.source.")):
        return None  # the same tensors registered a second time
    rules = [
        (r"^wm\.enc\.convs\.(\d)\.(kernel|bias)$", lambda m: f"enc/cnn{m[1]}/{m[2]}"),
        (r"^wm\.enc\.norms\.(\d)\.scale$", lambda m: f"enc/cnn{m[1]}norm/scale"),
        (r"^wm\.dyn\.(dynin\d|dyngru|obslogit|priorlogit)\.(kernel|bias)$", lambda m: f"dyn/{m[1]}/{m[2]}"),
        (r"^wm\.dyn\.(dynin\d)norm\.scale$", lambda m: f"dyn/{m[1]}norm/scale"),
        (r"^wm\.dyn\.(dynhid|obs|prior)\.(\d)\.(kernel|bias)$", lambda m: f"dyn/{m[1]}{m[2]}/{m[3]}"),
        (r"^wm\.dyn\.(dynhid|obs|prior)norm\.(\d)\.scale$", lambda m: f"dyn/{m[1]}{m[2]}norm/scale"),
        (r"^wm\.dec\.(sp\d|imgout)\.(kernel|bias)$", lambda m: f"dec/{m[1]}/{m[2]}"),
        (r"^wm\.dec\.(sp1norm|spnorm)\.scale$", lambda m: f"dec/{m[1]}/scale"),
        # the decoder's conv stack is indexed in reverse relative to the port
        (r"^wm\.dec\.convs\.(\d)\.(kernel|bias)$", lambda m: f"dec/conv{2 - int(m[1])}/{m[2]}"),
        (r"^wm\.dec\.norms\.(\d)\.scale$", lambda m: f"dec/conv{2 - int(m[1])}norm/scale"),
        (r"^wm\.(rew|con)\.mlp\.lins\.(\d)\.(kernel|bias)$", lambda m: f"{m[1]}/mlp/linear{m[2]}/{m[3]}"),
        (r"^wm\.(rew|con)\.mlp\.norms\.(\d)\.scale$", lambda m: f"{m[1]}/mlp/norm{m[2]}/scale"),
        (r"^wm\.rew\.out\.(kernel|bias)$", lambda m: f"rew/head/logits/{m[1]}"),
        (r"^wm\.con\.out\.(kernel|bias)$", lambda m: f"con/head/logit/{m[1]}"),
        (r"^(pol|val)\.mlp\.lins\.(\d)\.(kernel|bias)$", lambda m: f"{m[1]}/mlp/linear{m[2]}/{m[3]}"),
        (r"^(pol|val)\.mlp\.norms\.(\d)\.scale$", lambda m: f"{m[1]}/mlp/norm{m[2]}/scale"),
        (r"^slowval\.model\.mlp\.lins\.(\d)\.(kernel|bias)$", lambda m: f"slowval/mlp/linear{m[1]}/{m[2]}"),
        (r"^slowval\.model\.mlp\.norms\.(\d)\.scale$", lambda m: f"slowval/mlp/norm{m[1]}/scale"),
        (r"^pol\.out\.(kernel|bias)$", lambda m: f"pol/head/action/logits/{m[1]}"),
        (r"^val\.out\.(kernel|bias)$", lambda m: f"val/head/logits/{m[1]}"),
        (r"^slowval\.model\.out\.(kernel|bias)$", lambda m: f"slowval/head/logits/{m[1]}"),
        (r"^retnorm\.(lo|hi)$", lambda m: f"retnorm/{m[1]}/value"),
        (r"^slowval\.count$", lambda m: "slowval_count/value"),
    ]
    for pat, fn in rules:
        m = re.match(pat, port)
        if m:
            return fn(m)
    raise KeyError(f"no mapping for port key {port}")
