# Historical research sources

These source snapshots preserve the 2024 implementation separately from the maintained `bandit_cost_quality` package. They are excluded from installation and never imported by the maintained CLI or tests. They contain known defects and legacy `exec` configuration / pickle-loading paths; they are archival source, not supported execution entry points.

## Public snapshot

`public-2024/` contains the exact Python sources and three text scenarios at original public commit `683c41cad1d34bd1388e9fbfea56f72f51dadd45` (11 November 2024). That commit remains in this repository's history. The original README contained only three repeated repository-title lines and remains accessible in Git history.

## Laptop snapshot

`laptop-2024/` contains the exact source files recovered from the author's local research folder during the September 2026 maintenance pass:

- `algorithm.py`, `file_1`, and `file_2` match the public snapshot.
- `main.py` adds plotting and changes random-search repetitions from ten to one.
- `settings.py` makes clearing regret history optional and adds a true-mean getter.
- `file_3` changes costs from `[0,2,0.5,1.5,1]` to `[0,200,50,150,100]` and means from `[495,1,505,450,700]` to `[499.9,1,500.1,450,700]`.
- `bayesian_main.py` was an untracked local addition implementing random search followed by Bayesian optimization.

The file label refers to research provenance, not a verified last-modified date for every local edit. [SHA256SUMS](SHA256SUMS) records the recovered bytes. Run `sha256sum -c historical/SHA256SUMS` from the repository root.

The recovery did not import old pickle contents, bytecode, tuning history, or plots. Tracked pickle/bytecode artifacts were removed from the latest tree but remain recoverable from the existing Git history. The laptop backup was not edited.

## Maintained corrections

The 2026 runtime is a separately designated implementation derived from these sources. It fixes double-counted exploration, histories that omitted initial pulls, early termination when estimates were infeasible, wrong cost tie behavior when all arms were feasible, and stale empirical means. It uses strict JSON scenarios and versioned JSON tuning artifacts. All current figures were generated anew from the maintained runtime.
