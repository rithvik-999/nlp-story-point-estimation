#!/usr/bin/env python3
"""
Unit test suite verifying submission-readiness, structure, schema validity,
and executed outputs of `story_point_estimation.ipynb`.
"""

import os
import unittest
import json

try:
    import nbformat
    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, 'story_point_estimation.ipynb')

class TestNotebookIntegrity(unittest.TestCase):
    def setUp(self):
        self.assertTrue(os.path.exists(NOTEBOOK_PATH), f"Notebook missing at {NOTEBOOK_PATH}")
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            self.raw_json = json.load(f)
        if HAS_NBFORMAT:
            with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
                self.nb = nbformat.read(f, as_version=4)
        else:
            self.nb = None

    def test_schema_validity(self):
        """Test that notebook validates under official nbformat 4.5 schema."""
        self.assertEqual(self.raw_json.get('nbformat'), 4)
        self.assertEqual(self.raw_json.get('nbformat_minor'), 5)
        self.assertIn('cells', self.raw_json)
        self.assertIn('metadata', self.raw_json)
        if HAS_NBFORMAT:
            nbformat.validate(self.nb)

    def test_cell_counts(self):
        """Test that notebook has both markdown and code cells."""
        cells = self.raw_json['cells']
        code_cells = [c for c in cells if c['cell_type'] == 'code']
        md_cells = [c for c in cells if c['cell_type'] == 'markdown']
        self.assertGreaterEqual(len(code_cells), 12, "Should have at least 12 code cells")
        self.assertGreaterEqual(len(md_cells), 10, "Should have at least 10 markdown cells")

    def test_all_code_cells_have_outputs(self):
        """Test that EVERY single code cell has non-empty executed outputs and no error outputs."""
        cells = self.raw_json['cells']
        code_cells = [c for c in cells if c['cell_type'] == 'code']
        for idx, cell in enumerate(code_cells, 1):
            self.assertIsNotNone(cell.get('execution_count'), f"Code cell {idx} has None execution_count")
            outputs = cell.get('outputs', [])
            self.assertGreater(len(outputs), 0, f"Code cell {idx} has 0 outputs! Source preview: {str(cell.get('source', ''))[:60]}")
            has_error = any(o.get('output_type') == 'error' for o in outputs)
            self.assertFalse(has_error, f"Code cell {idx} contains an error output traceback!")

    def test_images_rendered_as_base64(self):
        """Test that EDA and comparative evaluation plots are rendered as base64 images."""
        found_images = 0
        for cell in self.raw_json['cells']:
            if cell['cell_type'] == 'code':
                for out in cell.get('outputs', []):
                    if 'data' in out and 'image/png' in out['data']:
                        b64_str = out['data']['image/png']
                        self.assertGreater(len(b64_str), 1000, "Base64 image string too short")
                        found_images += 1
        self.assertGreaterEqual(found_images, 2, f"Expected at least 2 base64 plot outputs, found {found_images}")

    def test_comparative_table_present(self):
        """Test that the Section 3.4 Comparative Evaluation Table is present in outputs."""
        found_table = False
        for cell in self.raw_json['cells']:
            if cell['cell_type'] == 'code':
                for out in cell.get('outputs', []):
                    text_content = ""
                    if 'text' in out:
                        text_content += "".join(out['text'])
                    if 'data' in out and 'text/html' in out['data']:
                        text_content += "".join(out['data']['text/html'])
                    if 'AUTHORITATIVE COMPARATIVE EVALUATION TABLE' in text_content or 'BERT word embeddings' in text_content:
                        found_table = True
                        break
        self.assertTrue(found_table, "Comparative evaluation table header missing from code cell outputs")

    def test_canonical_mappings_present(self):
        """Test that the 4 mandatory mappings are documented in markdown/code."""
        full_text = " ".join(["".join(c.get('source', '')) for c in self.raw_json['cells']])
        self.assertIn("Evergreen", full_text)
        self.assertIn("The MongoDB Engineering", full_text)
        self.assertIn("Aptana Studio", full_text)
        self.assertIn("The Mongo C++ driver", full_text)
        self.assertIn("The Mongo Java driver", full_text)

    def test_temporal_splitting_zero_leakage(self):
        """Test that zero date leakage and 5-fold temporal walk-forward CV are documented."""
        full_text = " ".join(["".join(c.get('source', '')) for c in self.raw_json['cells']])
        self.assertIn("Zero Date Leakage", full_text)
        self.assertIn("5-Fold", full_text)

if __name__ == '__main__':
    unittest.main()

