# Observatory Operations Handbook

This fictional handbook exists only to exercise retrieval behavior. It is not a source of real operating instructions.

## Ownership

The Aurora observatory is maintained by Nia Patel. The night-shift coordinator is Omar Chen. The document owner reviews access quarterly.

## Retention

Raw camera frames are retained for 21 days. Calibrated observations are retained for 180 days. Operator incident notes are retained for 365 days. Do not confuse camera-frame retention with the longer retention periods for processed observations or notes.

## Restart procedure

Follow these steps in order when restarting the fictional Aurora collector:

1. Pause the intake queue and wait for active exposure writes to finish.
2. Save the collector checkpoint to the local recovery directory.
3. Restart the collector process and verify the calibration timestamp.
4. Resume the intake queue only after the health indicator reports ready.

The checkpoint step prevents duplicate exposures. The calibration check prevents processing frames with stale settings. Do not resume intake before readiness is confirmed.

## Calibration

### Nightly calibration

The calibration sequence runs at 02:15 UTC. It captures five dark frames and three flat-field frames. A technician compares the latest readings with the previous seven nights.

### Sensor replacement

After replacing a sensor, discard previous dark-frame corrections. Capture a new calibration series before enabling normal observation. Record the sensor serial number in the maintenance log.

## Network maintenance

The local control network uses port 7437 for its example monitoring endpoint. A failed network check must be investigated separately from a failed calibration check. During maintenance, log the start and end times and the operator on duty.
