"""AprilTag end-effector pose serial protocol."""

from typing import Tuple


GET_TOOL_COMMAND = "@GET_TOOL#"


def is_get_tool_command(message: str) -> bool:
    """Return whether a serial line is the tool-coordinate query command."""
    return message.strip() == GET_TOOL_COMMAND


def format_tag_pose(position_m: Tuple[float, float, float], age_ms: int) -> str:
    """Format tag1-in-tag0 translation as millimeters."""
    x_m, y_m, z_m = position_m
    return "@TOOL,{:.1f},{:.1f},{:.1f},{}#".format(
        x_m * 1000.0,
        y_m * 1000.0,
        z_m * 1000.0,
        int(age_ms),
    )


def format_no_tag() -> str:
    """Return the no-valid-AprilTag-pose frame."""
    return "@NO_TAG#"


def format_error(message: str) -> str:
    """Return an ASCII-safe error frame."""
    safe_message = str(message).replace(",", " ").replace("#", " ").replace("@", " ")
    return "@ERR,{}#".format(safe_message.strip())
