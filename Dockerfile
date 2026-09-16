# Multi-stage Docker for synaptic-mesh (library crate; no binaries / examples).
# Keep `rust:1.98.1` in sync with Cargo.toml rust-version / rust-toolchain.toml
# / CI toolchain pin (see REVIEW.md "MSRV pin rule").
#
# This crate has no examples/ or [[bin]] targets. Sibling images (axon-encoder,
# neuromod) copy example binaries into debian-slim; we do not invent a fake
# binary. The published image is a rustdoc snapshot plus a version stamp so
# releases can ship a GHCR package. Library consumers should depend on crates.io.
#
# Runtime (default): rustdoc under /usr/share/doc/synaptic-mesh + VERSION
#   docker build -t synaptic-mesh:dev .
#   docker run --rm synaptic-mesh:dev
#
# Builder (tests / full toolchain):
#   docker build --target builder -t synaptic-mesh:builder .
#   docker run --rm synaptic-mesh:builder   # re-runs cargo test (CMD)

FROM rust:1.98.1-slim-bookworm AS builder

WORKDIR /app

# System toolchain stays root-owned under /usr/local/{cargo,rustup}.
# Writable Cargo registry/Git cache live under CARGO_HOME; build artifacts
# default to /app/target (WORKDIR), not under CARGO_HOME.
USER root
RUN rustup component add rustfmt clippy \
    && useradd --system --create-home --uid 10001 --shell /usr/sbin/nologin mesh \
    && mkdir -p /home/mesh/.cargo \
    && chown -R mesh:mesh /app /home/mesh

USER mesh
ENV CARGO_HOME=/home/mesh/.cargo
ENV PATH=/usr/local/cargo/bin:${PATH}

# Targeted copies so README/workflow/docs edits do not bust cargo layers.
# README.md is required: crate doctests compile it via include_str!.
COPY --chown=mesh:mesh Cargo.toml Cargo.lock rust-toolchain.toml README.md ./
COPY --chown=mesh:mesh src ./src
COPY --chown=mesh:mesh tests ./tests

# Tests in a cacheable layer (parity with native CI / local builder rechecks).
RUN cargo test --all-features --locked

# rustdoc for the published docs-oriented runtime (no example binaries exist).
RUN cargo doc --no-deps --all-features --locked --document-private-items \
    && mkdir -p /app/out \
    && grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/' > /app/out/VERSION \
    && test -s /app/out/VERSION \
    && test -f /app/target/doc/synaptic_wiring/index.html

# Default re-check when running the builder stage without args.
CMD ["cargo", "test", "--all-features", "--locked"]

# Runtime — smallest useful image CI can verify (docs + version stamp).
# Library consumers should depend on the crates.io package, not this image.
FROM debian:bookworm-slim

RUN useradd --system --create-home --uid 10001 --shell /usr/sbin/nologin mesh \
    && mkdir -p /usr/share/synaptic-mesh /usr/share/doc/synaptic-mesh

COPY --from=builder /app/out/VERSION /usr/share/synaptic-mesh/VERSION
COPY --from=builder /app/target/doc /usr/share/doc/synaptic-mesh

LABEL org.opencontainers.image.title="synaptic-mesh" \
      org.opencontainers.image.description="SNN wiring / topology / delay library (docs snapshot; depend on crates.io for the crate)" \
      org.opencontainers.image.source="https://github.com/Limen-Neural/synaptic-mesh" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0"

USER mesh
WORKDIR /home/mesh

CMD ["cat", "/usr/share/synaptic-mesh/VERSION"]
