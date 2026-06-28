import unittest

from tools.check_yolo_depth_deploy import (
    build_deploy_check_summary,
    make_check_item,
    _check_import,
)


class YoloDepthDeployCheckTests(unittest.TestCase):
    def test_make_check_item_uses_stable_fields(self):
        item = make_check_item("model", "ok", "模型存在")

        self.assertEqual(
            item,
            {
                "name": "model",
                "status": "ok",
                "message": "模型存在",
            },
        )

    def test_build_summary_reports_error_when_required_check_fails(self):
        summary = build_deploy_check_summary(
            [
                make_check_item("model", "ok", "模型存在"),
                make_check_item("ultralytics", "error", "未安装"),
                make_check_item("engine", "warn", "尚未导出"),
            ]
        )

        self.assertFalse(summary["ready"])
        self.assertEqual(summary["status"], "error")
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["warn_count"], 1)

    def test_build_summary_allows_warning_only_state(self):
        summary = build_deploy_check_summary(
            [
                make_check_item("model", "ok", "模型存在"),
                make_check_item("engine", "warn", "尚未导出"),
            ]
        )

        self.assertTrue(summary["ready"])
        self.assertEqual(summary["status"], "warn")
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["warn_count"], 1)

    def test_required_import_failure_is_error(self):
        item = _check_import("module_that_does_not_exist_for_test", "missing", required=True)

        self.assertEqual(item["status"], "error")

    def test_optional_import_failure_is_warning(self):
        item = _check_import("module_that_does_not_exist_for_test", "missing", required=False)

        self.assertEqual(item["status"], "warn")


if __name__ == "__main__":
    unittest.main()
