# A Nix flake to serve as the CI environment for CaBRNet.
# Not expected to be used for actual deployment of the library.
# TODO: use only one nixpkgs for all the provided dependencies
{
  description = "CaBRNet Nix flake.";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
    nixpkgs.url = "github:nixos/nixpkgs/25.05";
    # pydoc-markdown.url = "./ci/vendor/pydoc-markdown/";
    # disabled for now as more work is needed for the generation of
    # documentation in pure nix
  };
  outputs =
    { self
    , flake-utils
    , nixpkgs
    , nix-filter
    , ...
    }:
    flake-utils.lib.eachDefaultSystem
      (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonPkgs = pkgs.python3Packages;
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
            ];
          };
        };
      in
       {
        packages = rec {
          pname = "cabrnet";
          version = "1.2";
          # CaBRNet package without GUI, as it is intended for CI usage only
          default = self.packages.${system}.cabrnetWithoutGUI;
          # Overriding some dependencies from torch that prevents building
          tensorboardWithoutTests =
            (pythonPkgs.tensorboard.overridePythonAttrs
              (oldAttrs: {
                propagatedBuildInputs = with pythonPkgs; [
                  absl-py
                  grpcio
                  # google-auth-oauthlib removed because some tests are
                  # failing on CI
                  markdown
                  numpy
                  protobuf
                  setuptools
                  tensorboard-data-server
                  tensorboard-plugin-profile
                  tensorboard-plugin-wit
                  werkzeug
                  # not declared in install_requires, but used at runtime
                  # https://github.com/NixOS/nixpkgs/issues/73840
                  wheel
                ];
                # we don't want to test tensorboards imports on Frama-C CI
                pythonImportsCheck = [
                ];
              }));
          torchWithoutTests = (pythonPkgs.torch.override
            {
              tensorboard = tensorboardWithoutTests;
            }
          );
          onnxGraphsurgeon = (pythonPkgs.buildPythonPackage
          {
              pname = "onnx_graphsurgeon";
              version = "0.5.3";
              buildInputs = with pythonPkgs; [onnx numpy];
              src = pkgs.fetchFromGitHub {
                owner = "NVIDIA";
                repo = "TensorRT";
                rev = "v10.9.0";
                hash = "sha256-n19fgavWzByaPdY6Obv+NHI15HnU+NnkfNiSAQeeLJo=";
                sparseCheckout = [
                  "tools/onnx-graphsurgeon"
                ];
              };

              preBuild = "cd tools/onnx-graphsurgeon";
          });
          pacmap = (pythonPkgs.buildPythonPackage
            rec {
              pname = "pacmap";
              version = "0.7.3";
              src = pythonPkgs.fetchPypi {
                inherit pname version;
                hash = "sha256-RQ7R/VcsB+Cyv2asTaLpW1BlSVyEowRaQ1Qmm6yUxHM=";
              };
              doCheck = false;
              pytestCheckPhase = false;
              pyproject = true;
              build-system = with pythonPkgs; [
                setuptools
                wheel
              ];
              buildInputs = with pythonPkgs; [ scikit-learn numba annoy numpy ];
            });
          loguruWithoutTests = (pythonPkgs.loguru.overridePythonAttrs
                    (oldAttrs: {
                      doCheck = false;
                      pytestCheckPhase = false;
                      disabledTests =
                        [
                          "test_file_buffering"
                          "test_time_rotation"
                          "test_rust_notify"
                          "test_watch"
                          "test_awatch_interrupt_raise"
                        ];}));
              allDeps = with pythonPkgs ; [
                  numpy
                  pillow
                  tqdm
                  gdown
                  pyyaml
                  matplotlib
                  scipy
                  graphviz
                  opencv4
                  scikit-learn
                  pacmap
                  loguru
                  torchvision
                  mkdocs
                  pydocstyle
                  captum
                  pandas
                  pythonRelaxDepsHook
                  py-cpuinfo
                  onnx
                  onnxruntime
                  numba
                  annoy
                  onnxGraphsurgeon
                  ultralytics-thop
              ];
          cabrnetWithoutGUI = pythonPkgs.buildPythonPackage
            {
              inherit pname version;
              pyproject = true;
              src = sources.python;
              build-system = with pythonPkgs; [ setuptools wheel ];
              buildInputs = allDeps;
              propagatedBuildInputs = allDeps;
              nativeBuildInputs = allDeps;
              pythonRelaxDeps = [
                "tensorboard"
                "setuptools"
                "scikit-learn"
                "onnx-graphsurgeon"
              ];
              # Not needed for non-gui CI or access the internet
              pythonRemoveDeps = [
                "gradio"
                "ray"
                "optuna"
                "zenodo-get"
                "opencv-python-headless"
                "onnxruntime"
              ];
              # skip import checks, tests will handle that
              dontCheckRuntimeDeps = true;
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
                  python -m unittest tests/imports.py
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
           {
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




