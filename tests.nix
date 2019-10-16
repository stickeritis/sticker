{ pkgs ? import <nixpkgs> {}
, callPackage ? pkgs.callPackage
, lib ? pkgs.lib

, runCommand ? pkgs.runCommand
}:

let
  stickerModels = (callPackage ./nix/sticker.nix {}).models;
  sticker = callPackage ./default.nix {};
  src = lib.sourceFilesBySuffices ./sticker-utils [".conll"];
in runCommand "test-sticker" {} ''
  ${sticker}/bin/sticker tag \
    ${stickerModels.de-pos-ud.model}/share/sticker/models/de-pos-ud/sticker.conf \
    ${stickerModels.de-ner-ud-small.model}/share/sticker/models/de-ner-ud-small/sticker.conf \
    --input ${src}/testdata/small.conll | diff -u - ${src}/testdata/small-check.conll

    touch $out
''
