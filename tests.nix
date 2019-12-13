{ pkgs ? import <nixpkgs> {}
, callPackage ? pkgs.callPackage
, lib ? pkgs.lib

, runCommand ? pkgs.runCommand
}:

let
  sources = import ./nix/sources.nix;
  nixpkgs = import sources.nixpkgs {};
  danieldk = nixpkgs.callPackage sources.danieldk {};
  stickerModels = (nixpkgs.callPackage sources.sticker {}).models;
  sticker = nixpkgs.callPackage ./default.nix {
    libtensorflow-bin = danieldk.libtensorflow.v1_15_0;
  };
  src = lib.sourceFilesBySuffices ./sticker-utils [".conll"];
in runCommand "test-sticker" {} ''
  ${sticker}/bin/sticker tag \
    ${stickerModels.de-pos-ud.model}/share/sticker/models/de-pos-ud/sticker.conf \
    ${stickerModels.de-ner-ud-small.model}/share/sticker/models/de-ner-ud-small/sticker.conf \
    --input ${src}/testdata/small.conll | diff -u - ${src}/testdata/small-check.conll

    touch $out
''
