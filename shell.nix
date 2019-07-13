with import <nixpkgs> {};

let
  danieldk = pkgs.callPackage (builtins.fetchTarball {
    url = "https://git.sr.ht/~danieldk/nix-packages/archive/709c93a84504d558613bfc2538297ef2c532b890.tar.gz";
    sha256 = "0jspqxz8yzghn4j3awiqz3f76my8slk3s5ckk3gfzvhq1p0wzp5m";
  }) {};
  libtensorflow-cpu = danieldk.libtensorflow_1_14_0;
  libtensorflow-gpu = with pkgs; danieldk.libtensorflow_1_14_0.overrideAttrs (oldAttrs: rec {
    inherit (linuxPackages) nvidia_x11;
    cudatoolkit = cudatoolkit_10_0;
    cudnn = cudnn_cudatoolkit_10_0;
  });
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
    libtensorflow-cpu
    openssl
  ] ++ lib.optional stdenv.isDarwin darwin.apple_sdk.frameworks.Security;

  propagatedBuildInputs = [
    python3Packages.tensorflow
    python3Packages.toml
  ];
}
