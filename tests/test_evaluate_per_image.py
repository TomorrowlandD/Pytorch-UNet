import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import torch

from evaluate_per_image import per_image_scores, read_validation_ids, summarise


class ValidationIdTests(unittest.TestCase):
    def test_reads_text_ids_and_strips_extensions(self):
        path = Path('val_ids.txt')
        with patch.object(Path, 'is_file', return_value=True), \
                patch.object(Path, 'open', mock_open(read_data='first.jpg\nsecond\n')):
            self.assertEqual(read_validation_ids(path), ['first', 'second'])

    def test_reads_image_id_column_from_csv(self):
        path = Path('metrics.csv')
        csv_text = 'image_id,dice\nsample.jpg,0.9\n'
        with patch.object(Path, 'is_file', return_value=True), \
                patch.object(Path, 'open', mock_open(read_data=csv_text)):
            self.assertEqual(read_validation_ids(path), ['sample'])


class PerImageMetricTests(unittest.TestCase):
    def test_perfect_two_class_prediction(self):
        logits = torch.tensor(
            [[
                [[5.0, -5.0], [5.0, -5.0]],
                [[-5.0, 5.0], [-5.0, 5.0]],
            ]]
        )
        target = torch.tensor([[[0, 1], [0, 1]]])
        dice, iou = per_image_scores(logits, target, n_classes=2, mask_threshold=0.5)
        self.assertAlmostEqual(dice, 1.0)
        self.assertAlmostEqual(iou, 1.0)

    def test_summary_uses_per_image_macro_average(self):
        summary = summarise([
            {'image_id': 'a', 'dice': 0.8, 'iou': 0.7},
            {'image_id': 'b', 'dice': 1.0, 'iou': 0.9},
        ])
        self.assertEqual(summary['images'], 2)
        self.assertAlmostEqual(summary['mean_dice'], 0.9)
        self.assertAlmostEqual(summary['mean_iou'], 0.8)


if __name__ == '__main__':
    unittest.main()
