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
## Gate2 Read-only Communication Status

Latest result:

- Joint1-Joint7 are all online.
- `read_all_motors_monitor.py` runs with `SKIP_MOTOR_IDS = set()` / skipped motor list empty.
- Each cycle reads 7 motors × 4 parameters = 28 read-only parameter reads.
- Observed 5 cycles: OK=140, TIMEOUT=0, READ_FAIL=0, SEND_ERROR=0, PARSE_ERROR=0.
- Joint5 has recovered and now returns pos / iqf / vel / vbus.
- No enable command sent.
- No torque/current/position/velocity command sent.
- No zero setting command sent.
- No MoveIt real execution.

Result: Gate2 read-only communication validation is initially PASS.
If the 5-minute monitor also has no errors, record Gate2 as full PASS.

## Gate3A Read-only Feedback Sign Mapping

This section records feedback sign only.
It does not define command sign, URDF sign, zero offset, or real control direction.

| Joint | Motor Type | CAN_ID | Manual Motion | Feedback pos result | Gate3A Result |
|------|------------|--------|---------------|---------------------|---------------|
| Joint1 | RS00 | 1 | CCW | decreases | PASS |
| Joint1 | RS00 | 1 | CW | increases | PASS |
| Joint2 | RS00 | 2 | lift | decreases | PASS |
| Joint2 | RS00 | 2 | lower | increases | PASS |
| Joint3 | RS00 | 3 | lift | decreases | PASS |
| Joint3 | RS00 | 3 | lower | increases | PASS |
| Joint4 | RS05 | 4 | lift | decreases | PASS |
| Joint4 | RS05 | 4 | lower | increases | PASS |
| Joint5 | RS05 | 5 | left rotation | increases | PASS |
| Joint5 | RS05 | 5 | right rotation | decreases | PASS |
| Joint6 | RS05 | 6 | CCW | increases | PASS |
| Joint6 | RS05 | 6 | CW | decreases | PASS |
| Joint7 | RS05 | 7 | close | decreases | PASS |
| Joint7 | RS05 | 7 | open | increases | PASS |

Gate3A result: 7/7 complete.

Safety note:

- Feedback sign mapping was read-only.
- No enable command was sent.
- No torque/current/position/velocity command was sent.
- No zero setting command was sent.
- MoveIt was not connected to the real robot.
- These results must not be used as command sign mapping.

## Safety Rules Before Real Motor Motion

1. Read-only CAN communication must be tested first.
2. Test only one motor at a time before testing all motors together.
3. Do not set zero position unless the mechanical pose is confirmed.
4. Do not send large position commands before direction and limits are confirmed.
5. Do not connect MoveIt execution to real hardware before low-level motor tests are complete.

## Next Verification Items

| Item | Status | Note |
|------|--------|------|
| Ubuntu detects USB-CAN device | OK | PEAK System PCAN-USB detected by lsusb |
| can0 exists | OK | can0 detected as SocketCAN interface |
| CAN bitrate 1 Mbps configured | OK | can0 configured to 1000000 bitrate |
| Read motor feedback only | OK | Joint1-Joint7 replied to read-only device ID request |
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

### Joint6 / RS05 / CAN_ID 6

- Test type: read-only device ID request
- Interface: can0
- Bitrate: 1 Mbps
- Request frame: 0000FD06#0000000000000000
- Response frame: 000006FE#6C2293C79C90B012
- Result: OK, motor replied
- Motion command sent: NO
- Enable command sent: NO
- Zero position command sent: NO

### Joint7 / RS05 / CAN_ID 7

- Test type: read-only device ID request
- Interface: can0
- Bitrate: 1 Mbps
- Request frame: 0000FD07#0000000000000000
- Response frame: 000007FE#A66293C79C90B008
- Result: OK, motor replied
- Motion command sent: NO
- Enable command sent: NO
- Zero position command sent: NO
