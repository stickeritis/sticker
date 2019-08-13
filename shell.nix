with import <nixpkgs> {};

let
  danieldk = pkgs.callPackage (builtins.fetchTarball {
    url = "https://git.sr.ht/~danieldk/nix-packages/archive/c4a277bc5afaecbf982c3fa19cd7bac68de1826f.tar.gz";
    sha256 = "1n070qjp31fc94z5s1lglihlk7pnq14xdrvizjah9mfrwgaaa0p0";
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
