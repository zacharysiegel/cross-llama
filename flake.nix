{
  description = "Development environment for cross-llama";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, rust-overlay, ... }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];

      forEachSystem =
        function:
        nixpkgs.lib.genAttrs systems (
          system:
          function (
            import nixpkgs {
              inherit system;
              overlays = [ rust-overlay.overlays.default ];
            }
          )
        );
    in
    {
      devShells = forEachSystem (pkgs: {
        default = pkgs.mkShell {
          packages = [
            # The same toolchain rustup reads, so both tools agree.
            (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml)
            # llama-cpp-sys-2's build script compiles llama.cpp with these.
            pkgs.cmake
            pkgs.pkg-config
          ];
        };
      });
    };
}
