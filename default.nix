{
  callPackage
, lib
, stdenv

, defaultCrateOverrides

, pkgconfig

, curl
, openssl
, libtensorflow-bin
}:

((callPackage ./nix/Cargo.nix {}).workspaceMembers."sticker-utils").build.override {
  crateOverrides = defaultCrateOverrides // {
    sticker-utils = attrs: {
      src = lib.sourceFilesBySuffices ./sticker-utils [".rs"];

      postInstall = ''
        mv $out/bin/sticker-utils $out/bin/sticker
        rm -rf $out/lib
        rm $out/bin/*.d
      '';
    };

    tensorflow-sys = attrs: {
      nativeBuildInputs = [ pkgconfig ];

      buildInputs = [ libtensorflow-bin ] ++
        stdenv.lib.optional stdenv.isDarwin curl;
    };
  };
}
