import unittest

from aethryx_adaptive import (
    HardwareSnapshot,
    build_adaptive_plan,
    closest_profile_target,
)


class AdaptiveEngineTests(unittest.TestCase):
    def test_profile_targets_cover_supported_tiers(self):
        self.assertEqual(
            [closest_profile_target(vram) for vram in (4, 6, 8, 12, 16, 24)],
            [4, 6, 8, 12, 16, 24],
        )

    def test_12_gib_profile_uses_fp16_and_cpu_offload(self):
        hardware = HardwareSnapshot(32, 12.2, 20, "RTX 5070", 500)
        plan = build_adaptive_plan(hardware)
        self.assertEqual(plan.profile.target_vram_gib, 12)
        self.assertEqual(plan.profile.dtype, "fp16")
        self.assertTrue(plan.profile.cpu_offload)
        self.assertTrue(plan.profile.quality_priority)

    def test_low_ram_enables_basic_edit_mode(self):
        hardware = HardwareSnapshot(6, 8, 8, "Test GPU", 100)
        plan = build_adaptive_plan(hardware)
        self.assertTrue(plan.profile.basic_edit_mode)
        self.assertFalse(plan.profile.generation_enabled)

    def test_triposg_quantization_is_not_claimed_as_supported(self):
        hardware = HardwareSnapshot(32, 12, 20, "RTX 5070", 500)
        plan = build_adaptive_plan(hardware, engine_id="triposg")
        self.assertEqual(plan.profile.quantization, "none")

    def test_no_gpu_falls_back_to_safe_profile(self):
        hardware = HardwareSnapshot(16, None, 8, None, 100)
        plan = build_adaptive_plan(hardware)
        self.assertEqual(plan.profile.target_vram_gib, 4)
        self.assertEqual(plan.profile.dtype, "fp32")


if __name__ == "__main__":
    unittest.main()
