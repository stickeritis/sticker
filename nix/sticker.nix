{ callPackage, fetchFromGitHub }:

callPackage (fetchFromGitHub {
  owner = "stickeritis";
  repo = "nix-packages";
  rev = "d88a501";
  sha256 = "1qcp7z1hl1qh0dkf1iwh88a2ahwqakfzkqcpzm1lhy4m54p155xc";
}) {}
