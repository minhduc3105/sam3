import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from r2s_sam3.core import class_map, colorize, confusion_for, decode_gt, evaluate, label_for, merge_masks, metrics_from_confusion
from r2s_sam3.taxonomy import model_to_dataset_ids, model_vocabulary


class PipelineTests(unittest.TestCase):
    def test_vocabulary(self):
        self.assertEqual(len(model_vocabulary()), 22)
        self.assertEqual(model_to_dataset_ids(), list(range(1, 15)) + [0] * 8)
        self.assertEqual(model_vocabulary()[0], "slightly wet road surface")
        self.assertEqual(model_vocabulary()[-1], "roadside barrier")

    def test_fusion_and_ties(self):
        scores = np.full((2, 2), -np.inf, np.float32)
        prompts = np.full((2, 2), -1, np.int16)
        merge_masks(scores, prompts, 0, [np.ones((2, 2), bool)], [.2])
        self.assertEqual(class_map(prompts).sum(), 0)
        merge_masks(scores, prompts, 0, [np.ones((2, 2), bool)], [.3])
        merge_masks(scores, prompts, 1, [np.ones((2, 2), bool)], [.3])
        self.assertTrue(np.all(class_map(prompts) == 1))
        merge_masks(scores, prompts, 14, [np.array([[True, False], [False, False]])], [.4])
        self.assertEqual(class_map(prompts)[0, 0], 0)
        self.assertEqual(class_map(prompts)[1, 1], 1)

    def test_palette_and_missing_gt(self):
        self.assertEqual(int(decode_gt(np.array([[[0, 0, 0], [17, 163, 74]]], np.uint8))[0, 1]), 2)
        np.testing.assert_array_equal(colorize(np.array([[0, 2]], np.uint8)), np.array([[[0, 0, 0], [17, 163, 74]]], np.uint8))
        with self.assertRaises(ValueError):
            decode_gt(np.array([[[1, 2, 3]]], np.uint8))
        with self.assertRaises(FileNotFoundError):
            label_for(Path("/missing"), "val", Path("a.jpg"))

    def test_metrics_and_reevaluation(self):
        gt = np.array([[0, 1], [1, 2]], np.uint8)
        pred = np.array([[0, 1], [2, 2]], np.uint8)
        cm = confusion_for(gt, pred)
        m = metrics_from_confusion(cm)
        self.assertAlmostEqual(m["miou_15"], (1 + .5 + .5) / 3)
        self.assertEqual(m["classes_in_miou_15"], 3)
        self.assertAlmostEqual(m["foreground_binary_iou"], 1)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img = root / "a.jpg"
            label = root / "a.jpg___fuse.png"
            Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(img)
            colors = np.array([[[0, 0, 0], [2, 79, 59]], [[2, 79, 59], [17, 163, 74]]], np.uint8)
            Image.fromarray(colors).save(label)
            preds = root / "predictions"
            preds.mkdir()
            Image.fromarray(pred).save(preds / "a.jpg.png")
            first = evaluate([img], {img.name: label}, preds, root / "evaluation")
            second = evaluate([img], {img.name: label}, preds, root / "evaluation")
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
