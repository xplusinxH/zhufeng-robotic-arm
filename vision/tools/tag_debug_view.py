"""OpenCV debug overlay helpers for AprilTag pose service."""


def build_debug_overlay_items(detections, base_tag_id, tool_tag_id, base_ref_source, last_status):
    """Build simple overlay data from AprilTag detections."""
    tags = []
    for detection in detections:
        tag_id = int(detection.tag_id)
        if tag_id == int(base_tag_id):
            role = "base_ref"
        elif tag_id == int(tool_tag_id):
            role = "tool0"
        else:
            role = "tag"
        tags.append(
            {
                "id": tag_id,
                "label": "ID {} {}".format(tag_id, role),
                "corners": _corners_as_int_pairs(detection.corners),
            }
        )
    return {
        "tags": tags,
        "status_lines": [
            "base_ref_source: {}".format(base_ref_source),
            "last_status: {}".format(last_status),
        ],
    }


def draw_debug_overlay(image_bgr, detections, base_tag_id, tool_tag_id, base_ref_source, last_status):
    """Draw tag outlines and status text on an OpenCV BGR image."""
    import cv2

    overlay = build_debug_overlay_items(
        detections=detections,
        base_tag_id=base_tag_id,
        tool_tag_id=tool_tag_id,
        base_ref_source=base_ref_source,
        last_status=last_status,
    )
    for tag in overlay["tags"]:
        corners = tag["corners"]
        color = _tag_color(tag["id"], base_tag_id, tool_tag_id)
        for index in range(4):
            cv2.line(image_bgr, corners[index], corners[(index + 1) % 4], color, 2)
        cv2.putText(
            image_bgr,
            tag["label"],
            corners[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    y = 24
    for line in overlay["status_lines"]:
        cv2.putText(
            image_bgr,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 26
    return image_bgr


def should_quit_from_key(key_code):
    """Return whether an OpenCV waitKey result asks to quit."""
    key = int(key_code) & 0xFF
    return key in (ord("q"), 27)


def _corners_as_int_pairs(corners):
    return [(int(round(point[0])), int(round(point[1]))) for point in corners]


def _tag_color(tag_id, base_tag_id, tool_tag_id):
    if int(tag_id) == int(base_tag_id):
        return (0, 255, 0)
    if int(tag_id) == int(tool_tag_id):
        return (255, 0, 0)
    return (255, 255, 255)
