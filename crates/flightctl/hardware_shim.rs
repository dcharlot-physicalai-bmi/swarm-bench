// hardware_shim.rs — the OTHER side of the seam (reference sketch; not built here,
// needs the `mavlink` crate + a vehicle/SITL). The point: the control() call in the
// middle is byte-identical to the sim host. Only the I/O around it changes.
//
// Loop on the companion computer (PX4 Offboard):
//
//   loop {
//       // ---- INPUT shim: MAVLink telemetry -> the control ABI state[17] ----
//       // LOCAL_POSITION_NED   -> p (x,y,z), v (vx,vy,vz)     [NED, z-down]
//       // ATTITUDE_QUATERNION  -> q (w,x,y,z) and body rates  (rollspeed,...)
//       // NOTE frame conversion: PX4 is NED/FRD; this controller is world-z-up/FLU.
//       //   convert once here (negate z and the appropriate axes) so control() sees
//       //   exactly the frame the sim used. Rotor speeds r0..r3 are internal to the
//       //   plant and unused by control(); zero-fill them.
//       let state: [f32;17] = ned_flu_from_mavlink(&telemetry);
//       let setpoint: [f32;4] = [x_des, y_des, z_des, yaw_des];
//
//       // ---- identical call ----
//       let mut out = [0f32;4];
//       control(state.as_ptr(), setpoint.as_ptr(), out.as_mut_ptr());
//
//       // ---- OUTPUT shim: control ABI out[4] -> a MAVLink setpoint ----
//       // This module closes the FULL loop (down to per-motor Ω). Two options:
//       //  (a) direct actuators: normalize Ω_i -> [0,1] and send
//       //      SET_ACTUATOR_CONTROL_TARGET / actuator_motors  (PX4 in direct-actuator
//       //      offboard, or Betaflight-class stacks). Same module, motors out.
//       //  (b) keep PX4's inner loops: build a variant that exports the OUTER loop
//       //      only (position -> attitude+thrust) and send SET_ATTITUDE_TARGET; PX4
//       //      runs the rate loop. Same source tree, different exported entry point.
//       send_setpoint(&mut mavlink, &out);
//
//       wait_next_tick(); // match the control rate the module was tuned at
//   }
//
// Determinism note: compose control() with the M4 fixed-point substrate (both are
// integer-only, self-contained) and the whole perceive->control->act loop becomes
// bit-exact and hash-anchorable end to end, not just the physics.
