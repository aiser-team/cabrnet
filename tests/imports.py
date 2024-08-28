import unittest


class TestImports(unittest.TestCase):
    """Tests that all cabrnet modules are correctly installed
    and reachable.
    """

    def test_cabrnet_generic(self):
        """
        Tests the import of cabrnet generic module.
        """
        import cabrnet.generic

    def test_cabrnet_protopnet(self):
        """
        Tests the import of cabrnet protopnet module.
        """
        import cabrnet.protopnet

    def test_cabrnet_prototree(self):
        """
        Tests the import of cabrnet prototree module.
        """
        import cabrnet.prototree

    def test_cabrnet_utils(self):
        """
        Tests the import of cabrnet utils module.
        """
        import cabrnet.utils

    def test_cabrnet_evaluation(self):
        """
        Tests the import of cabrnet evaluation module.
        """
        import cabrnet.evaluation

    def test_cabrnet_visualization(self):
        """
        Tests the import of cabrnet visualization module.
        """
        import cabrnet.visualization
