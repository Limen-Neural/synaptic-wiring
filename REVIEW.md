# Local review quality gate

These commands are the **human quality bar** beyond GitHub Actions.
Run them before claiming a PR is ready when the change touches `src/`,
`Cargo.toml`, or public APIs.

## MSRV pin rule

`Cargo.toml` `rust-version`, `rust-toolchain.toml` `channel`, the README's
**MSRV** line, the Dockerfile `FROM rust:X.Y.Z` tag, every `toolchain:` /
`rust-version:` pin in `.github/workflows/ci.yml` (the toolchain install,
the packaging job, and the `cargo-deny` action) and
`.github/workflows/coverage.yml` must stay **identical** (currently
**1.98.1**). The `validate` job in `ci.yml` reads those workflow pins and
the Dockerfile rust tag and fails if they drift (issue #35 / LIM-1042).

To bump MSRV:

1. Set the new version in Cargo.toml, rust-toolchain.toml, README.md,
   Dockerfile, ci.yml, and coverage.yml.
2. Run the mandatory commands below on that toolchain
   (`rustup run <ver> cargo test --locked`, etc.).
3. Confirm GitHub Actions is green.

Do not bump only one pin.

## Build profiles

`Cargo.toml` defines four `[profile.*]` sections (issue #40). Use the one
that matches what you're doing instead of reaching for a one-off
`cargo build` flag:

| Profile   | Command                     | Why it's configured that way |
|-----------|------------------------------|-------------------------------|
| `dev`     | `cargo build` / `cargo run`  | `opt-level = 0`, full debug info, incremental compiles — fastest edit/compile loop. |
| `test`    | `cargo test`                 | Inherits `dev` but bumps to `opt-level = 1` so the unit/property-test suite (GH#21) runs faster, while keeping compile times close to `dev`. |
| `release` | `cargo build --release`      | `opt-level = 3`, thin LTO, single codegen unit, stripped — tuned for the crates.io publish (GH#37): best runtime performance and smallest binary. |
| `bench`   | `cargo bench`                | Inherits `release` so criterion benches reflect real release performance, but keeps debug symbols (`strip = false`) so profilers can still symbolize. |

The rationale above is kept in sync with the comments on each
`[profile.*]` block in `Cargo.toml` — update both together if a profile
changes.

## Packaging

Changes to `Cargo.toml` metadata, `exclude`, or anything that adds a
top-level file should be checked against what actually ships:

```bash
cargo package --locked --list   # source, Cargo manifests, licenses, README, CHANGELOG
cargo package --locked          # must succeed
```

CI's `package` job runs both, fails on any packaged file outside that
allowlist, and builds the unpacked `.crate` from a temp directory so a
file removed by `exclude` cannot break the published crate (issue #36).
Add a genuinely new consumer-facing file to the allowlist in
`.github/workflows/ci.yml`; anything else belongs in `exclude`.

## When to run

- Before every push that changes core mesh, topology, or neuromodulation code.
- After resolving merges with `main`.
- Before requesting a review.

## Mandatory commands

### Format + core locked test matrix

```bash
# Success is silent: exit 0 and no stdout means formatting is clean.
cargo fmt --check

cargo test --locked
cargo clippy --all-features -- -D warnings
```

### Checkpoint resume property suite (LIM-1223)

Default CI already runs this via `cargo test --locked`. The bounded profile is
`tests/checkpoint_resume/`: 512 seeded cases plus named boundary/regression
fixtures, restoring through JSON and postcard.

For a longer ignored/nightly run (10_000 additional seeds by default):

```bash
cargo test --locked --test checkpoint_resume resume_equivalence_nightly -- --ignored
CHECKPOINT_RESUME_CASES=50000 cargo test --locked --test checkpoint_resume resume_equivalence_nightly -- --ignored
```

Persist a failing seed in `REGRESSION_SEEDS`. If the on-failure shrinker prints
a smaller scenario, add that as a named fixture next to the other boundary tests.

## Regression guards

After any "security" or dependency PR, confirm core product APIs still exist:

```bash
# Check for key structs and modules
rg -n 'pub struct SynapticMesh' src/mesh.rs
rg -n 'pub struct NeuromodNeuron' src/router.rs
rg -n 'pub struct SynapticGraph' src/topology/graph.rs
rg -n 'pub mod topology' src/lib.rs
rg -n 'pub use router::\{[^}]*NeuromodNeuron' src/lib.rs   # confirms the router re-export (not a removed `neuromod` module)
```

## Diff hygiene

```bash
git fetch origin main
git diff --stat origin/main...HEAD
# Expect only intentional files
```

## Origin hygiene (never push local tooling)

These paths must stay untracked and ignored (aligned with `.gitignore`):

- `.worktrees/`
- `.idea/`
- `target/`

```bash
git ls-files .worktrees .idea target   # must print nothing
git check-ignore -v .worktrees .idea target
```

## Do not merge if

- `src/mesh.rs` or `src/router.rs` are unexpectedly altered or removed.
- `git diff origin/main` shows unexpected public-API removals.

## Pass criteria

- All mandatory commands exit 0
- `cargo fmt --check` is silent (no output) with exit 0
- Clippy reports zero warnings under `-D warnings`
- Regression guards pass
- Diff hygiene: only intentional files for the PR
- Local tooling dirs are not in the commit
