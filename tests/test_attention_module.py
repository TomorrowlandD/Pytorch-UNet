import unittest

import torch
from torch import nn

from unet import LiteSpatialReductionMHSA, UNet
from utils.model_loading import load_model_state


class LiteSpatialReductionMHSATest(unittest.TestCase):
    def test_module_preserves_shape_and_backpropagates(self):
        module = LiteSpatialReductionMHSA(
            channels=32,
            attention_dim=16,
            num_heads=4,
            sr_ratio=2,
        )
        inputs = torch.randn(2, 32, 8, 10, requires_grad=True)

        outputs = module(inputs)
        self.assertEqual(outputs.shape, inputs.shape)

        outputs.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(module.reduce.weight.grad)
        self.assertIsNotNone(module.expand.weight.grad)
        self.assertIsNotNone(module.layer_scale.grad)

    def test_invalid_head_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'divisible'):
            LiteSpatialReductionMHSA(
                channels=32,
                attention_dim=15,
                num_heads=4,
            )

    def test_attention_none_keeps_original_state_dict_layout(self):
        model = UNet(3, 2, bilinear=True, attention='none')
        self.assertIsInstance(model.bottleneck_attention, nn.Identity)
        self.assertFalse(any(
            key.startswith('bottleneck_attention.')
            for key in model.state_dict()
        ))

    def test_complete_attention_unet_preserves_segmentation_shape(self):
        model = UNet(
            3,
            2,
            bilinear=True,
            attention='lite_sr_mhsa',
            attention_dim=32,
            attention_heads=4,
            attention_sr_ratio=2,
        ).to(memory_format=torch.channels_last)
        inputs = torch.randn(1, 3, 64, 96).to(memory_format=torch.channels_last)

        with torch.no_grad():
            outputs = model(inputs)

        self.assertEqual(outputs.shape, (1, 2, 64, 96))

    def test_production_attention_parameter_budget(self):
        baseline = UNet(3, 2, bilinear=True, attention='none')
        attention = UNet(
            3,
            2,
            bilinear=True,
            attention='lite_sr_mhsa',
            attention_dim=128,
            attention_heads=4,
            attention_sr_ratio=2,
        )
        baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
        attention_parameters = sum(parameter.numel() for parameter in attention.parameters())
        added_parameters = attention_parameters - baseline_parameters

        self.assertGreaterEqual(added_parameters, 190_000)
        self.assertLessEqual(added_parameters, 210_000)
        self.assertLess(added_parameters / baseline_parameters, 0.02)


class CheckpointCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.baseline = UNet(3, 2, bilinear=True, attention='none')
        self.attention = UNet(
            3,
            2,
            bilinear=True,
            attention='lite_sr_mhsa',
        )
        self.baseline_state = dict(self.baseline.state_dict())
        self.baseline_state['mask_values'] = [0, 255]

    def test_backbone_mode_loads_old_unet_into_attention_model(self):
        mask_values = load_model_state(
            self.attention,
            self.baseline_state,
            load_mode='backbone',
        )
        self.assertEqual(mask_values, [0, 255])
        self.assertTrue(torch.equal(
            self.attention.inc.double_conv[0].weight,
            self.baseline.inc.double_conv[0].weight,
        ))

    def test_strict_mode_rejects_old_unet_for_attention_model(self):
        with self.assertRaises(RuntimeError):
            load_model_state(
                self.attention,
                self.baseline_state,
                load_mode='strict',
            )

    def test_backbone_mode_rejects_non_attention_mismatch(self):
        broken_state = dict(self.baseline_state)
        broken_state.pop('inc.double_conv.0.weight')

        with self.assertRaisesRegex(RuntimeError, 'outside the attention branch'):
            load_model_state(
                self.attention,
                broken_state,
                load_mode='backbone',
            )


if __name__ == '__main__':
    unittest.main()
