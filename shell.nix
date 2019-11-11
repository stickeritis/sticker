with import <nixpkgs> {};

let
  danieldk = pkgs.callPackage (fetchFromGitHub {
    owner = "danieldk";
    repo = "nix-packages";
    rev = "4464a255f5e0adca710f23318ba861bbfa408f8f";
    sha256 = "0pg69q8dyybpa62clfa2krp6pnshali2nsbkkpnahcsjf9zq0nap";
  }) {};
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
    danieldk.libtensorflow.v1_15_0
    openssl
  ] ++ lib.optional stdenv.isDarwin darwin.apple_sdk.frameworks.Security;

  propagatedBuildInputs = [
    python3Packages.tensorflow-bin
    python3Packages.toml
  ];
}
