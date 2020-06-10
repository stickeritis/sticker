{ pkgs ? import (import nix/sources.nix).nixpkgs {} }:

let
  sources = import nix/sources.nix;
  danieldk = pkgs.callPackage sources.danieldk {};
  crateOverrides = with pkgs; defaultCrateOverrides // {
    sticker-utils = attrs: {
      src = lib.sourceFilesBySuffices ./sticker-utils [".rs"];
    };

    tensorflow-sys = attrs: {
      nativeBuildInputs = [ pkgconfig ];

      buildInputs = [ danieldk.libtensorflow.v1_15_0 ] ++
        stdenv.lib.optional stdenv.isDarwin curl;
    };
  };
  buildRustCrate = pkgs.buildRustCrate.override {
    defaultCrateOverrides = crateOverrides;
  };
  cargo_nix = pkgs.callPackage ./nix/Cargo.nix {
    inherit pkgs buildRustCrate;
  };
in cargo_nix.workspaceMembers.sticker-utils.build
