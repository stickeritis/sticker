with import <nixpkgs> {};

let
  libtensorflow_1_13_1 = with pkgs; callPackage ./nix/libtensorflow {
    inherit (linuxPackages) nvidia_x11;
    cudatoolkit = cudatoolkit_10_0;
    cudnn = cudnn_cudatoolkit_10_0;
  };
in stdenv.mkDerivation rec {
  name = "sticker-env";
  env = buildEnv { name = name; paths = buildInputs; };

  nativeBuildInputs = [
    pkgconfig
    latest.rustChannels.stable.rust
    python3Packages.pytest
  ];

  buildInputs = [
    curl
    libtensorflow_1_13_1
    openssl
  ] ++ lib.optional stdenv.isDarwin darwin.apple_sdk.frameworks.Security;

  propagatedBuildInputs = [
    python3Packages.tensorflow
    python3Packages.toml
  ];
}
