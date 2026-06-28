import unittest

from tools.serve_yolo_depth_serial import (
    DETECT_COMMAND,
    YoloDepthSerialState,
    handle_serial_receive_buffer,
    is_detect_command,
)


class YoloDepthSerialServiceTests(unittest.TestCase):
    def test_recognizes_detect_command(self):
        self.assertTrue(is_detect_command("@DETECT#"))
        self.assertTrue(is_detect_command("  @DETECT#\n"))
        self.assertFalse(is_detect_command("@POSE,0,0,0,0,0,0,1#"))

    def test_keeps_partial_command_in_receive_buffer(self):
        calls = []

        receive_buffer, responses = handle_serial_receive_buffer(
            receive_buffer="@DET",
            incoming_text="ECT",
            capture_scene=lambda: calls.append("called") or [],
        )

        self.assertEqual(receive_buffer, DETECT_COMMAND[:-1])
        self.assertEqual(responses, [])
        self.assertEqual(calls, [])

    def test_handles_complete_detect_command(self):
        def capture_scene():
            return ["@OBJ,0,remote,0.90,1.0,2.0,3.0,1,2,3,4,9,yolo_depth#", "@END,1#"]

        receive_buffer, responses = handle_serial_receive_buffer(
            receive_buffer="",
            incoming_text="@DETECT#",
            capture_scene=capture_scene,
            state=YoloDepthSerialState(base_from_tool=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        )

        self.assertEqual(receive_buffer, "")
        self.assertEqual(
            responses,
            ["@OBJ,0,remote,0.90,1.0,2.0,3.0,1,2,3,4,9,yolo_depth#", "@END,1#"],
        )

    def test_handles_multiple_frames_and_invalid_command(self):
        calls = []

        receive_buffer, responses = handle_serial_receive_buffer(
            receive_buffer="",
            incoming_text="@BAD#@DETECT#",
            capture_scene=lambda: calls.append("detect") or ["@NOOBJ#", "@END,0#"],
            state=YoloDepthSerialState(base_from_tool=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        )

        self.assertEqual(receive_buffer, "")
        self.assertEqual(calls, ["detect"])
        self.assertEqual(responses[0], "@ERR,BAD_COMMAND,use DETECT#")
        self.assertEqual(responses[1:], ["@NOOBJ#", "@END,0#"])

    def test_pose_frame_updates_state_before_detect(self):
        state = YoloDepthSerialState()
        calls = []

        _receive_buffer, responses = handle_serial_receive_buffer(
            receive_buffer="",
            incoming_text="@POSE,0.1,0.2,0.3,0,0,0,1#@DETECT#",
            capture_scene=lambda: calls.append(state.base_from_tool[0][3]) or ["@NOOBJ#", "@END,0#"],
            state=state,
        )

        self.assertEqual(responses, ["@POSE_OK#", "@NOOBJ#", "@END,0#"])
        self.assertEqual(calls, [0.1])

    def test_detect_requires_pose_when_state_has_no_pose(self):
        state = YoloDepthSerialState()

        _receive_buffer, responses = handle_serial_receive_buffer(
            receive_buffer="",
            incoming_text="@DETECT#",
            capture_scene=lambda: ["@NOOBJ#", "@END,0#"],
            state=state,
        )

        self.assertEqual(responses, ["@ERR,NO_POSE,send_POSE_before_DETECT#"])


if __name__ == "__main__":
    unittest.main()
