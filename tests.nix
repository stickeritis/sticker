{ pkgs ? import (import nix/sources.nix).nixpkgs {} }:

let
  sources = import ./nix/sources.nix;
  nixpkgs = import sources.nixpkgs {};
  stickerModels = (nixpkgs.callPackage sources.sticker {}).sticker_models;
  sticker = nixpkgs.callPackage ./default.nix {};
  src = pkgs.lib.sourceFilesBySuffices ./sticker-utils [".conll"];
in pkgs.runCommand "test-sticker" {} ''
  ${sticker}/bin/sticker tag \
    ${stickerModels.de-pos-ud.model}/share/sticker/models/de-pos-ud/sticker.conf \
    ${stickerModels.de-ner-ud-small.model}/share/sticker/models/de-ner-ud-small/sticker.conf \
    --input ${src}/testdata/small.conll | diff -u - ${src}/testdata/small-check.conll

    touch $out
''
