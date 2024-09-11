# A Nix flake to serve as the CI environment for CaBRNet.
# Not expected to be used for actual deployment of the library.
# TODO: use only one nixpkgs for all the provided dependencies
{
  description = "CaBRNet Nix flake.";
  inputs = {
    captum.url = "./ci/vendor/captum/"; # relative path for flake is a best effort, see https://github.com/NixOS/nix/issues/9339
    nixpkgs.follows = "captum/nixpkgs";
    nix-filter.follows = "captum/nix-filter";
    flake-utils.follows = "captum/flake-utils";
    # pydoc-markdown.url = "./ci/vendor/pydoc-markdown/";
    # disabled for now as more work is needed for the generation of
    # documentation in pure nix
  };
  outputs =
    { self
    , flake-utils
    , nixpkgs
    , nix-filter
    , captum
      #, pydoc-markdown
    , ...
    }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      pythonPkgs = pkgs.python310Packages;
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
            (nix-filter.lib.inDirectory "tests")
            (nix-filter.lib.inDirectory "tools")
            (nix-filter.lib.inDirectory "website")
          ];
        };
      };
    in
    rec {
      packages = rec {
        pname = "cabrnet";
        version = "1.0";
        default = self.packages.${system}.cabrnetWithoutGUI;
        # CaBRNet package without GUI, as it is intended for CI usage only
        cabrnetWithoutGUI = pythonPkgs.buildPythonPackage
          {
            inherit pname version;
            pyproject = true;
            src = sources.python;
            build-system = with pythonPkgs; [ setuptools wheel ];
            propagatedBuildInputs = with pythonPkgs; [
              torch
              torchvision
              numpy
              pillow
              tqdm
              gdown
              pyyaml
              matplotlib
              scipy
              loguru
              graphviz
              opencv4
              mkdocs
              pydocstyle
              captum.packages.${system}.default
              pandas
              pythonRelaxDepsHook
              py-cpuinfo
            ];
            pythonRelaxDeps = [
              "tensorboard"
              "torchvision"
              "setuptools"
            ];
            # Not needed for non-gui CI or access the internet
            pythonRemoveDeps = [
              "gradio"
              "ray"
              "optuna"
              "zenodo-get"
              "opencv-python-headless"
            ];
            nativeCheckInputs = [ pkgs.pyright pythonPkgs.black ];
            importCheck = with pythonPkgs; [ scipy torch numpy ];
          };
        cabrnet-doc = pkgs.stdenv.mkDerivation
          {
            # TODO: this derivation only evaluates to the user manual.
            # More work is needed for website generation
            inherit version;
            pname = "cabrnet-doc";
            src = sources.python;
            nativeBuildInputs = with pkgs; [
              pandoc
              python3Minimal
              texliveFull
            ];
            buildPhase = ''
              cd docs/manuals/
              make clean
              make
            '';
            installPhase = ''
              mkdir -p $out/docs/manuals
              cp user_manual.pdf $out/docs/manuals
            '';
          };
      };
      checks = {
        formattingCode =
          self.packages.${system}.default.overrideAttrs
            (oldAttrs: {
              doCheck = true;
              name = "check-${oldAttrs.name}-code";
              checkPhase = ''
                black --check src/ tools/
              '';
            });
        formattingDocstring =
          self.packages.${system}.default.overrideAttrs
            (oldAttrs: {
              doCheck = true;
              name = "check-${oldAttrs.name}-docstring";
              checkPhase = ''
                python tools/check_docstrings.py -d src/
              '';
            });
        # TODO: integrate the whole test suite
        testLoadingModules =
          self.packages.${system}.default.overrideAttrs
            (oldAttrs: {
              doCheck = true;
              name = "check-${oldAttrs.name}-unittests";
              checkPhase = ''
                python -m tests/loading.py
              '';
            });
        typing =
          self.packages.${system}.default.overrideAttrs
            (oldAttrs: {
              doCheck = true;
              name = "check-${oldAttrs.name}-typing";
              checkPhase = ''
                pyright src/cabrnet
              '';
            });

      };
      devShells =
        let venvDir = "./.cabrnet-venv-nix"; in
        rec {
          default = self.devShells.${system}.install;
          inputsFrom = self.packages.${system}.default;
          install = pkgs.mkShell {
            name = "CaBRNet development shell environment.";
            shellHook = ''
              echo "Welcome in the development shell for CaBRNet."
              SOURCE_DATE_EPOCH=$(date +%s)
              if [ -d "${venvDir}" ];
              then
              echo "Skipping venv creation, '${venvDir}' already exists"
              else
              echo "Creating new venv environment in path: '${venvDir}'"
              ${pythonPkgs.python.interpreter} -m venv "${venvDir}"
              fi

              # Under some circumstances it might be necessary to add your virtual
              # environment to PYTHONPATH, which you can do here too;
              PYTHONPATH=$PWD/${venvDir}/${pythonPkgs.python.sitePackages}/:$PYTHONPATH
              source "${venvDir}/bin/activate"
            '';
          };
        };
    });
}




