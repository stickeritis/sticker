# We pin nixpkgs to improve reproducability. We don't pin Rust to a
# specific version, but use the latest stable release.

let
  sources = import ./nix/sources.nix;
  nixpkgs = import sources.nixpkgs {};
  danieldk = nixpkgs.callPackage sources.danieldk {};
  mozilla = nixpkgs.callPackage "${sources.mozilla}/package-set.nix" {};
in with nixpkgs; mkShell {
  nativeBuildInputs = [
    pkgconfig
    mozilla.latest.rustChannels.stable.rust
    python3Packages.pytest
  ];

  buildInputs = [
    curl
    danieldk.libtensorflow.v1_15_0
    openssl
  ] ++ lib.optional stdenv.isDarwin darwin.apple_sdk.frameworks.Security;

  propagatedBuildInputs = [
    (python3.withPackages (ps: with ps; [
      tensorflow-bin
      toml
    ]))
  ];
}
