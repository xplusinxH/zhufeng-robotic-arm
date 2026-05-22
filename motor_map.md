# Sukinee Motor Map

| Joint | Motor Type | Target CAN_ID | Current CAN_ID | Direction | Zero Position | Joint Limit | Note |
|------|------------|---------------|----------------|-----------|---------------|-------------|------|
| Joint1 | RS00 | 1 | 1 | ? | ? | ? | base yaw |
| Joint2 | RS00 | 2 | 2 | ? | ? | ? | shoulder |
| Joint3 | RS00 | 3 | 3 | ? | ? | ? | elbow |
| Joint4 | RS05 | 4 | 4 | ? | ? | ? | wrist |
| Joint5 | RS05 | 5 | 5 | ? | ? | ? | wrist |
| Joint6 | RS05 | 6 | 6 | ? | ? | ? | wrist |
| Joint7 | RS05 | 7 | 7 | ? | ? | ? | gripper drive |

## Current Hardware Status

- CAN_ID mapping has been configured and recorded.
- Motor direction has NOT been verified yet.
- Mechanical zero position has NOT been verified yet.
- Joint software limits have NOT been verified against the real robot yet.
- Real motors must NOT be controlled by MoveIt at this stage.

## Safety Rules Before Real Motor Motion

1. Read-only CAN communication must be tested first.
2. Test only one motor at a time before testing all motors together.
3. Do not set zero position unless the mechanical pose is confirmed.
4. Do not send large position commands before direction and limits are confirmed.
5. Do not connect MoveIt execution to real hardware before low-level motor tests are complete.

## Next Verification Items

| Item | Status | Note |
|------|--------|------|
| Ubuntu detects USB-CAN device | TODO | check with lsusb and ip link |
| can0 exists | TODO | if yes, use SocketCAN |
| CAN bitrate 1 Mbps configured | TODO | required by RS00 / RS05 |
| Read motor feedback only | TODO | no motion command |
| Joint direction verified | TODO | one motor at a time |
| Zero position verified | TODO | do not set casually |
| Joint limits verified | TODO | compare URDF / MoveIt / real robot |
