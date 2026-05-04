import importlib
import unittest


class TestImports(unittest.TestCase):
    """Tests that all cabrnet modules are correctly installed
    and reachable.
    """

    def test_cabrnet_generic(self):
        """
        Tests the import of cabrnet generic module.
        """
        importlib.import_module("cabrnet.archs.generic")

    def test_cabrnet_protopnet(self):
        """
        Tests the import of cabrnet protopnet module.
        """
        importlib.import_module("cabrnet.archs.protopnet")

    def test_cabrnet_prototree(self):
        """
        Tests the import of cabrnet prototree module.
        """
        importlib.import_module("cabrnet.archs.prototree")

    def test_cabrnet_utils(self):
        """
        Tests the import of cabrnet utils module.
        """
        importlib.import_module("cabrnet.core.utils")

    def test_cabrnet_evaluation(self):
        """
        Tests the import of cabrnet evaluation module.
        """
        importlib.import_module("cabrnet.core.evaluation")

    def test_cabrnet_visualization(self):
        """
        Tests the import of cabrnet visualization module.
        """
        importlib.import_module("cabrnet.core.visualization")
