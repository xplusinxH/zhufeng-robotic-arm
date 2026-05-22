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

## CAN Communication Test Log

### Joint1 / RS00 / CAN_ID 1

- Test type: read-only device ID request
- Interface: can0
- Bitrate: 1 Mbps
- Request frame: 0000FD01#0000000000000000
- Response frame: 000001FE#649932313037350D
- Result: OK, motor replied
- Motion command sent: NO
- Enable command sent: NO
- Zero position command sent: NO

### Joint2 / RS00 / CAN_ID 2

- Test type: read-only device ID request
- Interface: can0
- Bitrate: 1 Mbps
- Request frame: 0000FD02#0000000000000000
- Response frame: 000002FE#B62032313037350D
- Result: OK, motor replied
- Motion command sent: NO
- Enable command sent: NO
- Zero position command sent: NO

### Joint3 / RS00 / CAN_ID 3

- Test type: read-only device ID request
- Interface: can0
- Bitrate: 1 Mbps
- Request frame: 0000FD03#0000000000000000
- Response frame: 000003FE#911F32313037350D
- Result: OK, motor replied
- Motion command sent: NO
- Enable command sent: NO
- Zero position command sent: NO

### Joint4 / RS05 / CAN_ID 4

- Test type: read-only device ID request
- Interface: can0
- Bitrate: 1 Mbps
- Request frame: 0000FD04#0000000000000000
- Response frame: 000004FE#395A93C79C90B008
- Result: OK, motor replied
- Motion command sent: NO
- Enable command sent: NO
- Zero position command sent: NO

### Joint5 / RS05 / CAN_ID 5

- Test type: read-only device ID request
- Interface: can0
- Bitrate: 1 Mbps
- Request frame: 0000FD05#0000000000000000
- Response frame: 000005FE#C86E93C79C90B015
- Result: OK, motor replied
- Motion command sent: NO
- Enable command sent: NO
- Zero position command sent: NO
