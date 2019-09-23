with import <nixpkgs> {};

let
  danieldk = pkgs.callPackage (builtins.fetchTarball {
    url = "https://git.sr.ht/~danieldk/nix-packages/archive/f02c88bdd5e959d2d3a7a71c5c1208431f7107b9.tar.gz";
    sha256 = "05ax116wy35jcnvy82jv38ig0ssivg4lwv2gi2rlqdsbl4bg3gbf";
  }) {};
  libtensorflow-cpu = danieldk.libtensorflow_1_14_0;
  libtensorflow-gpu = with pkgs; danieldk.libtensorflow_1_14_0.override {
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
    libtensorflow-cpu
    openssl
  ] ++ lib.optional stdenv.isDarwin darwin.apple_sdk.frameworks.Security;

  propagatedBuildInputs = [
    python3Packages.tensorflow-bin
    python3Packages.toml
  ];
}
