{
  description = "Pydoc markdown Nix flake.";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
    nixpkgs.url = "nixpkgs";
    nr-util.url = "./ci/vendor/nr-util/";
  };
  outputs =
    { self
    , flake-utils
    , nixpkgs
    , nix-filter
    , nr-util
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
        default = self.packages.${system}.pydoc-markdown;
        pydoc-markdown = pythonPkgs.buildPythonApplication
          rec {
            pname = "pydoc_markdown";
            version = "4.8.2";
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
              nr-util.packages.${system}.default
            ];
            build-system = with pythonPkgs; [ wheel ];
            src = pythonPkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-+2ySfjE4beF0ctQvm9PTvikFl30Cb2IWiBxlFFqmfws=";
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
