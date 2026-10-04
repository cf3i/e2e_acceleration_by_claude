# p1_s01_no_cudnn_sdpa — `torch.backends.cuda.enable_cudnn_sdp(False)`

Change: SDPA no longer dispatches to cuDNN (which builds an execution plan per new (q_len, kv_len) shape —
2.4 ms CPU per call in the baseline). It now uses flash / mem-efficient kernels.

Micro (`micro.json`, ctx 1024): decode cold 438 → **59.4** ms/tok, warm 63.0 → 59.9; prefill cold 1302 → 431 ms.
Profile (`profile/`): GPU busy 0.21 → 0.30; `_cudnn_attention_forward` gone from the top CPU ops; kernels/token
unchanged (5.9k) — the remaining cost is per-op host dispatch (mm/mul/copy_/add/empty_strided at ~9 µs CPU each).
