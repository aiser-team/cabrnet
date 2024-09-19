{
  description = "nr-util Nix flake.";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
    nixpkgs.url = "nixpkgs";
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
      pythonPkgs = pkgs.python311Packages;
      lib = pkgs.lib;
      sources = { };
    in
    rec {
      packages = rec {
        default = self.packages.${system}.nr-util;
        nr-util = pythonPkgs.buildPythonPackage
          rec {
            pname = "nr.util";
            version = "0.8.12";
            pyproject = true;
            buildInputs = with pythonPkgs;  [
              poetry-core
              click
              pyyaml
              jinja2
              tomli
              tomli-w
              watchdog
              yapf
              requests
            ];
            build-system = with pythonPkgs; [ setuptools wheel ];
            src = pythonPkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-pFScIDPZnS8DebPz0jP9KoreKGu/CzrQzHzqFgIiFPQ=";
            };
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

