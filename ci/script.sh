#!/bin/bash

set -ex

cargo build --target ${TARGET}

# Needs to run after build/check, so that protobuf files are generated.
cargo fmt --all -- --check

cargo test --target ${TARGET}

# On the lower-bound Rust, we only care about passing tests.
if [ "$TRAVIS_RUST_VERSION" = "stable" ] && [ "$TARGET" = "x86_64-unknown-linux-gnu" ]; then
  cargo clippy -- -D warnings
fi


