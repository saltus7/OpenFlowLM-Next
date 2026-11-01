from gguf import GGUFReader

P = r"C:/Users/user/.cache/huggingface/hub/models--GrootTeam--Groot-19B-A3B-261023-GGUF/snapshots/2fa1ec958e3bbf7312cf0af97802ff26bfdc5260/Groot-19B-A3B-261023-Q4_K_M.gguf"
r = GGUFReader(P)
lines = []

# 1) scalar metadata fields we care about
for k in sorted(r.fields):
    lk = k.lower()
    if not any(x in lk for x in ("expert", "ffn", "hidden", "token", "head", "block", "embed", "n_")): 
        continue
    try:
        v = r.fields[k].contents()
    except Exception as e:
        v = f"<unreadable: {e}>"
    s = repr(v)
    if len(s) > 140:
        s = s[:140] + f"... [len={len(v)}]"
    lines.append(f"{k} = {s}")

# 2) ground truth on expert count: shape of blk.0 ffn_gate_exps
for t in r.tensors:
    if t.name.endswith("ffn_gate_exps.weight") and t.name.startswith("blk.0."):
        lines.append(f"TENSOR {t.name} shape={tuple(t.shape)} dtype={t.tensor_type.name}")
        break
for t in r.tensors:
    if t.name.endswith("ffn_up_exps.weight") and t.name.startswith("blk.0."):
        lines.append(f"TENSOR {t.name} shape={tuple(t.shape)} dtype={t.tensor_type.name}")
        break

open("gguf_probe.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"wrote {len(lines)} lines to gguf_probe.txt")
