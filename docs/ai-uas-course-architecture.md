# AI + UAS — Course Architecture & Platform Stack

**Context:** Semester course, "Physical AI" track. Designed for a programmable, NDAA-clean teaching fleet spanning simulation, indoor swarm, onboard-AI airframes, and turnkey Blue UAS field platforms.

**Operating assumptions** (adjust and the BOM re-flows):
- ~20 students, lab pods of 4 → 5 pods.
- ~15-week semester.
- Both **indoor** (unregulated airspace — most hardware lab work) and **outdoor** field components.
- **NDAA / Blue UAS treated as binding.** No DJI, no Tello/Ryze, no Insta360/Antigravity — nothing Chinese-sourced in the supply chain. This decision propagates through every line of the BOM.
- Budget figures below are **order-of-magnitude budgetary estimates**, not quotes. Enterprise platforms are quote-based.

---

## 1. Why not the obvious consumer drones

The default education drone for a decade was the Tello (DJI flight controller, Python SDK, swarm mode). Disqualified here on supply chain. The Antigravity A1 you asked about is doubly disqualified: no SDK (the team confirmed none is planned), a payload-detection system that auto-lands on any modification, goggle-only control, an FAA visual-observer requirement per aircraft, and Chinese manufacture. It's a capture instrument, not a controllable node. The only legitimate role for an A1 in this course is as a **dataset source** — 360° footage feeding a perception module — not as fleet hardware.

The cost of going NDAA-clean is real: you give up the cheapest hardware tier and pay more per airframe. The offset is that everything students touch transfers directly to defense and federal contexts, which for BMI is the point.

---

## 2. Pedagogical spine

The course backbone is the autonomy stack — **sense → estimate → plan → act** — built up one layer per phase, **simulation-first** so the AI content is decoupled from crash risk and hardware cost. Hardware enters only after a capability works in sim. The arc terminates in a multi-agent capstone, because "fleet" is the differentiator and coordination is where the hard AI lives.

Mapping AI concepts onto UAS problems:

| AI concept | UAS instantiation |
|---|---|
| Perception / CV | onboard object detection, tracking, depth, optical flow |
| State estimation | EKF sensor fusion, GPS-denied VIO/SLAM |
| Planning / search | path planning, coverage, obstacle avoidance |
| Control / RL | attitude/position control, learned policies in sim |
| Multi-agent | formation control, consensus, decentralized task allocation |

---

## 3. The four platform tiers

| Tier | Platform | Teaches | Where it slots |
|---|---|---|---|
| **1 — Sim** | PX4/ArduPilot SITL + Gazebo, MAVSDK, ROS 2 | the entire autonomy stack at zero crash cost; a 20-vehicle "fleet" runs on one workstation | weeks 2–5, and the cheap swarm sandbox throughout |
| **2 — Indoor swarm** | Crazyflie 2.1+ (Bitcraze, Swedish) + Crazyswarm2 | sim-to-real, coordinated multi-drone flight, GPS-denied positioning | weeks 6–7 single, 8–13 swarm |
| **3 — Onboard AI** | Pixhawk/PX4 airframe + NVIDIA Jetson Orin, **or** ModalAI VOXL 2 | real perception/inference running onboard over MAVLink; outdoor field flight | onboard-AI track + outdoor component |
| **4 — Blue UAS** | Skydio X10 (SDK) or Parrot ANAFI USA (Olympe Python SDK) | defense-grade autonomy out of the box; procurement-clean field platform | field demos + capstone outdoor option |

Tier 1 carries most of the AI grade. Tier 2 is the workhorse and the cheapest *touchable* hardware. Tiers 3–4 are smaller fleets — shared, instructor-supervised — that expose students to production-grade and defense-grade systems.

---

## 4. Module map (15 weeks)

| Wk | Module | Tier | Deliverable |
|---|---|---|---|
| 1 | UAS anatomy, flight dynamics, the autonomy stack, regulatory landscape (Part 107, Remote ID, NDAA/Blue UAS, VO rules) | — | reading + dev environment stood up |
| 2 | Sim & control I: PX4 SITL + Gazebo, MAVLink, MAVSDK | 1 | fly a single sim vehicle programmatically |
| 3 | Sim & control II: ROS 2 nodes, telemetry, mission scripting | 1 | autonomous waypoint mission in sim |
| 4 | Perception I: onboard CV, object detection/tracking on sim camera | 1 | detector running on simulated feed |
| 5 | Perception II: depth, optical flow, intro Jetson deployment | 1→3 | model ported to Jetson |
| 6 | State estimation: EKF, GPS vs GPS-denied, VIO/SLAM | 1 | localization in GPS-denied sim |
| 7 | Single-drone hardware: Crazyflie one-vehicle, sim-to-real | 2 | first real autonomous flight |
| 8 | Indoor positioning: Lighthouse / UWB, the netted cage | 2 | closed-loop position hold on hardware |
| 9 | Planning & avoidance: path planning, behavior trees | 1→2 | reactive obstacle avoidance |
| 10 | Learned control: RL for attitude/position in sim | 1 | trained policy beats PID baseline |
| 11 | Multi-agent I: Crazyswarm2, formation control | 2 | 4-drone formation |
| 12 | Multi-agent II: consensus, decentralized coordination | 2 | leaderless formation, fault tolerance |
| 13 | Multi-agent III: cooperative task allocation, coverage | 2 | swarm cooperative search in the cage |
| 14 | Field block: onboard-AI airframe + Blue UAS outdoor ops | 3→4 | outdoor autonomous mission, VO-compliant |
| 15 | Capstone presentations | all | integrated demo |

---

## 5. Capstone options

Pick one per pod:
- **Cooperative search/map (indoor):** Crazyflie swarm covers the cage, fuses detections into a shared map. Pure multi-agent + perception.
- **Outdoor coordinated survey:** Pixhawk+Jetson airframes (or Blue UAS) fly a coordinated area survey, onboard detection, ground-station aggregation.
- **Sim-only mega-fleet:** 20–50 vehicles in SITL running a decentralized mission — for students who want to push coordination algorithms past what the hardware budget allows.

---

## 6. Bill of materials

Quantities assume the 20-student / 5-pod structure. Prices are **rough budgetary estimates (USD), verify current** — vendor quotes will move these.

### Tier 1 — Simulation & compute
| Item | Qty | Est. unit | Notes |
|---|---|---|---|
| GPU lab workstation (or use existing lab) | 5 | $1,500–3,000 | one per pod; mid-range GPU sufficient for SITL + training |
| PX4/ArduPilot, Gazebo, ROS 2, MAVSDK | — | $0 | open source |
| **Subtotal** | | **~$7,500–15,000** | skip entirely if pods use existing machines |

### Tier 2 — Indoor swarm (the core spend)
| Item | Qty | Est. unit | Notes |
|---|---|---|---|
| Crazyflie 2.1+ | 12 | ~$225 | 10 flying + 2 spare; enough for a real swarm shared across pods |
| Crazyradio 2.0 dongle | 6 | ~$35 | radios for parallel pod work |
| Lighthouse positioning deck | 12 | ~$20 | per-vehicle |
| SteamVR base station 2.0 | 4 | ~$250–300 | covers the cage volume |
| Spare props / motors / batteries / chargers | lot | ~$600 | swarms crash; budget consumables |
| Flight cage / netting | 1 | ~$1,500–4,000 | indoor safety enclosure |
| **Subtotal** | | **~$7,000–10,000** | |

> Alternative positioning: UWB (Loco) instead of Lighthouse if your cage geometry fights the base stations. Mocap (OptiTrack/Vicon) is the research-grade option but adds five figures — only if you already have it.

### Tier 3 — Onboard-AI airframes
| Item | Qty | Est. unit | Notes |
|---|---|---|---|
| Pixhawk 6C/6X flight controller | 3 | ~$250 | |
| Airframe + motors/ESC/props kit | 3 | ~$400–800 | quad, sub-defined payload |
| NVIDIA Jetson Orin Nano dev kit | 3 | ~$250–500 | onboard inference |
| Camera / depth sensor | 3 | ~$200–400 | |
| **— or — ModalAI VOXL 2 / Starling dev drone** | 2 | quote (board ~$1k+; integrated drone several $k) | US-made, NDAA-compliant, PX4-native, SDK; cleaner integration than rolling your own |
| **Subtotal** | | **~$4,000–9,000** | 2–3 shared airframes, instructor-supervised |

### Tier 4 — Blue UAS field platform
| Item | Qty | Est. unit | Notes |
|---|---|---|---|
| Skydio X10 platform + controller | 1 | quote — budget low–mid five figures | best onboard autonomy + SDK |
| **— or — Parrot ANAFI USA + Olympe SDK** | 1 | ~$7,000–10,000 | cheaper Blue UAS entry, Python SDK |
| **Subtotal** | | **~$8,000–40,000+** | 1 shared field platform; defines the high end of the budget |

### Program total (excl. existing compute)
**~$26,000 (lean: ANAFI USA, minimal airframes) → ~$70,000+ (Skydio + ModalAI).** The Blue UAS platform choice dominates the spread.

---

## 7. Regulatory & operational checklist

- **Indoor = no FAA airspace.** Most hardware work (all of Tier 2) flies indoors with zero Part 107 burden. Lean into this — it's why the Crazyflie cage is the backbone.
- **Outdoor (Tiers 3–4):** Part 107 for the operator, or fly under FAA's educational/recreational provisions; **Remote ID** compliance required on outdoor airframes.
- **Visual observer:** required for any FPV/beyond-line-of-sight operation. Build VO procedure into the field block.
- **Blue UAS / NDAA:** confirm whether BMI's federal funding or DoD affiliation makes the cleared-list binding for *purchasing* (vs. just good practice). If binding, every airframe must be on the DoD cleared list at time of purchase — verify SKUs against the current list, since it churns.
- **Safety:** netted cage for indoor swarm, props guards, battery (LiPo) storage/charging protocol and fire safety, pre-flight checklists as graded artifacts.
- **Insurance:** institutional UAS liability coverage for outdoor ops.

---

## 8. To finalize — tell me four things

1. **Real headcount and pod size** → re-flows every quantity.
2. **Hard budget ceiling** → decides Skydio vs ANAFI USA, ModalAI vs DIY airframe, and Tier-1 workstation spend.
3. **Lab space** → cage dimensions drive Lighthouse vs UWB and swarm size.
4. **Is NDAA/Blue UAS actually binding for procurement, or just preferred?** → if only preferred, a cheaper non-cleared field platform reopens, though I'd still hold the line for a military institute.

Give me those and I'll pull live vendor quotes/links and turn Section 6 into a procurement-ready BOM.
