{
  description = "CaBRNet Nix flake.";
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
      sources = {
        python = nix-filter.lib {
          root = ./.;
          include = [
            "environment.yml"
            "pyproject.toml"
            "requirements.txt"
            (nix-filter.lib.inDirectory "build")
            (nix-filter.lib.inDirectory "configs")
            (nix-filter.lib.inDirectory "tests")
            (nix-filter.lib.inDirectory "docs")
            (nix-filter.lib.inDirectory "src")
            (nix-filter.lib.inDirectory "tools")
            (nix-filter.lib.inDirectory "website")
          ];
        };
      };
    in
    rec {
      packages = rec {
        default = self.packages.${system}.cabrnet;
        # CaBRNet package: include binary in src/apps and documentation
        cabrnet = pythonPkgs.buildPythonPackage
          {
            pname = "cabrnet";
            version = "0.2";
            pyproject = true;
            src = sources.python;
            build-system = with pythonPkgs; [ setuptools wheel ];
            dependencies = with pythonPkgs;[ setuptools ];
            nativeCheckInputs = with pythonPkgs;[ mypy ];
          };
      };

      apps = rec {
        default = {
          type = "app";
          program =
            "${self.packages.${system}.default}/src/main.py";
        };
      };
      devShells = rec {
        default = pkgs.mkShell {
          name = "CaBRNet development shell environment.";
          inputsFrom = [ self.packages.${system}.cabrnet ];
          packages = with pythonPkgs; [ torch torchvision numpy pillow tqdm ];
          shellHook = ''
            echo "Welcome in the development shell for CaBRNet."
          '';
        };
      };
    });
}

