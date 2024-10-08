{
  description = "Captum Nix flake.";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
    nixpkgs.url = "nixpkgs/24.05";
  };
  outputs =
    { self
    , flake-utils
    , nixpkgs
    , nix-filter
    , ...
    }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      pythonPkgs = pkgs.python310Packages;
      lib = pkgs.lib;
      sources = { };
    in
    rec {
      packages = rec {
        default = self.packages.${system}.captum;
        captum = pythonPkgs.buildPythonPackage
          rec {
            pname = "captum";
            version = "0.7.0";
            src = pythonPkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-fKh9DcZ7O3WJpzC5cLlTYXKlRo4OMb+GV/3XOrxWijM=";
            };
            build-system = with pythonPkgs; [ setuptools wheel ];
            doCheck = false;
          };
      };
      apps = rec {
        default = {
          type = "app";
          program =
            "${self.packages.${system}.default}/main.py";
        };
      };
    });
}

