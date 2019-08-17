#!/bin/bash

set -ex

binaries="sticker-server sticker-tag sticker-train"

cargo build --target "$TARGET" --release

tmpdir="$(mktemp -d)"
name="${PROJECT_NAME}-${TRAVIS_TAG}-${TARGET}"
staging="${tmpdir}/${name}"
mkdir "${staging}"
out_dir="$(pwd)/deployment"
mkdir "${out_dir}"

# Copy binaries
for binary in ${binaries}; do
  cp "target/${TARGET}/release/${binary}" "${staging}/${binary}"
done

# Strip binaries.
for binary in ${binaries}; do
  strip "${staging}/${binary}"
done

cp {README,LICENSE}.md "${staging}/"
cp -r doc "${staging}/"

if [ "${TARGET}" = "x86_64-unknown-linux-gnu" ]; then
  tf_build="linux-x86_64"
  dylib_ext=".so"
elif [ "${TARGET}" = "x86_64-apple-darwin" ]; then
  tf_build="darwin-x86_64"
  dylib_ext=".dylib"
else
  >&2 echo "Cannot build release tarbal for target ${TARGET}"
  exit 1
fi

tf_archive="libtensorflow-cpu-${tf_build}-1.14.0.tar.gz"

# Add Tensorflow library
curl -O https://storage.googleapis.com/tensorflow/libtensorflow/${tf_archive}
mkdir tensorflow
tar -zxf "${tf_archive}" -C tensorflow
cp tensorflow/lib/*${dylib_ext} "${staging}"

# Add binary directory to the search path.
if [ "$TARGET" = "x86_64-unknown-linux-gnu" ]; then
  for binary in ${binaries}; do
    patchelf --set-rpath '$ORIGIN' "${staging}/${binary}"
  done
else
  # Darwin
  for binary in ${binaries}; do
    install_name_tool -add_rpath @executable_path/. "${staging}/${binary}"
  done
fi

# Add Python graph scripts
cp sticker-graph/sticker-write-{conv,rnn,transformer}-graph "${staging}/"
cp -r sticker-graph/sticker_graph "${staging}"

( cd "${tmpdir}" && tar czf "${out_dir}/${name}.tar.gz" "${name}")

rm -rf "${tmpdir}"
