# Resource contention

- **GPU (L40S 48 GB, shared with sglang Gemma; we never touch that process).** Free memory moved between ~2.9 and ~6.5 GB during this work (EDA/Gemma load varies). `fnp.laya.train.plan_resources` reads `torch.cuda.mem_get_info()` at run time: >=12 GB mb8/len768, >=5.5 mb4/len512, >=4 mb2/len512, else mb1/len384; below 2.5 GB it uses CPU. On CUDA OOM it retries down a ladder (smaller micro-batch, shorter max_len, then CPU).
- **Measured (multilingual 322M, embeddings frozen, fp16/bf16 autocast, activation checkpointing):** stage A mb=1, max_len 384: peak allocated 3.35 GB, 0.18 s/micro-batch (656 items, 119 s/epoch). Stage B mb=2, max_len 512, group 4: peak 3.36 GB, 0.12 s/micro-batch. Freezing the 197M-parameter token embedding is what makes it fit (no grads/Adam state for it).
- **Gemma concurrency:** shared endpoint, keep `GEMMA_MAX_CONCURRENCY` <= 12; Laya inference needs no Gemma calls.
- **DuckDB:** single writer per file (`data/fnp.duckdb`); training scripts only read rows from JSONL and write under `data/models/`, so they never open the database for writing. Open read-only (or snapshot) when another stage holds it.
