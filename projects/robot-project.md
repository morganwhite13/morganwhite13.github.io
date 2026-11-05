---
title: Autonomous Robot Jar Collection System
permalink: /projects/robot-project/
description: "An autonomous Webots robot that finds, collects and stores honey jars in a 20×20 m warehouse, plus a tested v2 rewrite of its controller with a measured before/after comparison."
---

## Quick Summary

An autonomous mobile robot built in Webots that uses computer vision, a compass and hand-tuned waypoint routes to navigate a 20×20 meter warehouse, locate honey jars, collect them, and place them in designated storage areas—all without human intervention.

I later came back to the project and rewrote the controller as **v2**, keeping the original **v1** unchanged next to it so the two can be compared directly.

**Tech Stack:** Webots R2025a, Java, Computer Vision, Robotics, JUnit  
**Status:** ✅ v1 complete; v2 written and tested on a robot model, Webots comparison pending  
**GitHub:** [View Source Code](https://github.com/morganwhite13/Autonomous-Robot-Jar-Collection-System)

---

## Video Demo

<div class="video-embed" data-url="https://youtu.be/kO9e8dzjCfg"><iframe src="https://www.youtube.com/embed/kO9e8dzjCfg" title="Autonomous Robot Jar Collection Demo" loading="lazy" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>

The video shows the original controller.

---

## Before and After: the v2 Rewrite

v2 keeps every waypoint, gripper timing and vision threshold from v1, and changes how the robot moves between waypoints and recovers when something goes wrong. Both versions were measured on the same model of the robot:

![Bar and line charts comparing v1 and v2: the jar scan turns 271 degrees in v1 (188 with wheel slip) and 371 in v2; a 530 cm leg started 2.9 degrees off ends 26.8 cm from the waypoint in v1 and 1.1 cm in v2; after a failed search v1 drives 41.4 m and v2 0.8 m](/assets/charts/robot-before-after.svg){: .chart .chart-wide}

| Situation | v1 (original) | v2 (rewrite) |
|---|---|---|
| After a failed search for jars 2-5 | Drives a 17.8 m straight line through 4 immovable boxes | Returns 0.8 m to the search point |
| Robot blocked while driving | Pushes forever | Detects the stall in about 4 s and moves on |
| "360°" jar scan | Turns about 271° (fixed time) | Turns a full circle, measured by the compass |
| End of a 530 cm leg started 2.9° off | 26.8 cm from the waypoint | 1.1 cm |
| No jar can be found | Retries forever | Stops after 3 attempts with a mission report |
| Tests and CI | None | 66 JUnit tests covering 96% of v2's logic; static analysis; GitHub Actions |

These numbers come from a kinematic model of the Pioneer 3-DX (no box physics), not from Webots. A script that runs both versions headless in Webots and compares them is ready, and that run is the next step. The full write-up, with each fix and the original line numbers, is in the [repository's pull request](https://github.com/morganwhite13/Autonomous-Robot-Jar-Collection-System/pull/2).

---

## The Challenge

Design and implement a fully autonomous robot system that can:
- Navigate a warehouse full of walls, immovable wooden boxes and pushable cardboard boxes
- Detect and locate small objects with an onboard camera
- Manipulate objects with a gripper system
- Return collected items to a designated drop-off zone
- Repeat this process for multiple objects

Jar detection and grasping rely on the robot's own camera and touch sensor. To move between the search area and the drop-off zone, the controller follows hand-tuned waypoint routes, using the compass and the robot's exact position from the simulator (the controller runs as a Webots Supervisor).

---

## System Architecture

### Hardware Configuration

The robot is built on a Pioneer 3-DX platform equipped with:

**Sensors:**
- 64×64 RGB camera, 45° field of view, for colour-based jar detection
- Compass for heading
- Touch sensor (front bumper) to know when a jar is between the gripper fingers
- 6 ultrasonic distance sensors (enabled and logged for debugging; not used in control)

**Actuators:**
- Differential drive motors (left & right wheels)
- Gripper lift motor
- Gripper finger motors (left & right)

**Key Specs:**
- Max wheel speed: 5 rad/s (about 0.49 m/s)
- Arena: 20 m × 20 m
- Jars: 5, each with its own drop-off pad
- Obstacles: 32 pushable cardboard boxes, 10 immovable wooden boxes and the walls

---

## How It Works

### 1. Colour-Based Jar Detection

The camera image is split into left, centre and right thirds (21 / 22 / 21 columns), and the controller counts jar-green pixels in each:

- A pixel counts as green when R < 120, G > 35 and B < 120. v2 also requires green to be the strongest channel, so grey and brown pixels no longer count.
- **Search:** the robot turns on the spot and scores each frame by its green pixels, with a bonus when they're in the centre third, then turns to the best heading.
- **Approach:** it creeps forward, steering towards whichever third leads the others by more than 10 pixels, until the front bumper is pressed.

### 2. Waypoint Navigation

The mission runs in phases, with hand-tuned waypoints:

- **Clearing:** a 15-waypoint route pushes cardboard boxes out of the way and ends at the search point.
- **Search and grab:** scan, approach, close the gripper, lift.
- **Carry:** a 10-waypoint route to the drop-off lane.
- **Place:** line up with that jar's pad, release, back out, and go back for the next jar.

Each leg turns on the spot towards the next waypoint, then drives to it. v1 turns at full speed until within ±3° and drives without steering. v2 slows down as it nears the target heading, works correctly across the 0°/360° boundary, steers towards the waypoint while driving, and detects when it's stuck.

---

## Results

| Metric | Value |
|--------|-------|
| Jars collected (original runs) | 5/5 |
| Total mission time (original runs) | About 13-15 minutes |
| Control loop | One 32 ms simulation step (about 31 Hz) |
| Turn tolerance | ±3° |
| v2 on the robot model | 5/5 jars, no stalls, including with motor lag and wheel slip |

---

## Challenges & Solutions

### Challenge: A Failed Search Could Strand the Robot

**Problem:** In v1, if the robot failed to find jars 2-5 it took the "return from drop-off" route even though it was still at the search point. That route's first leg is a straight line across the arena through four immovable wooden boxes, and with no stall detection the robot could push against one forever.

**Solution:** v2 tracks where the robot is and takes the short route back to the search point. After 3 failed searches in a row it stops and reports what it achieved.

---

### Challenge: Turning and Driving Precisely

**Problem:** At full speed one 32 ms step turns the robot about 5°, so it could jump over the ±3° target window and swing back and forth. Driving without steering, a heading error inside that tolerance became about 27 cm of sideways error over the longest leg.

**Solution:** v2 slows the turn as it nears the target and steers towards the waypoint while driving. On the model, every turn finishes and the long leg ends within about 1 cm.

---

### Challenge: Gripper Synchronization

**Problem:** Premature gripper opening could drop collected jars

**Solution:** Timed sequences give the motors time to reach their positions: close the fingers for 100 steps, lift for 120, hold for 20 before driving off.

**Result:** No jar drops in the original runs

---

### Challenge: Testing a Robot Without the Simulator

**Problem:** Webots is slow to iterate with, and the original controller called the Webots API from every method, so none of it could be tested on its own.

**Solution:** v2 puts all Webots access behind one interface. The mission logic runs unchanged against a kinematic model of the robot with a synthetic camera, and line-for-line ports of v1's methods run on the same model, so both versions can be measured the same way.

---

## Code Architecture

**v1:** one file, `ProjectController3.java` (765 lines), with every method `static`: `intelligentJarSearch`, `rotateAndFindGreen`, `carefullyApproachGreen`, `executeGrab`, `makeHardcodedTurn`, `moveHardcodedAhead` and the route methods.

**v2:** 13 classes and one interface, each with one job:

| Class | Responsibility |
|---|---|
| `Mission` | The phase sequence, where the robot is, retry limit |
| `Navigator` | Turning, driving, stall detection |
| `JarHandler` | Scan, approach, grab, place |
| `Vision` | Pixel classification, sector counts, steering decision |
| `Routes`, `Config` | Every waypoint and tunable number, defined once |
| `RobotPlatform` / `WebotsPlatform` | The robot interface and its Webots implementation (the only class that touches Webots) |
| `MissionStats`, `Log` | Simulated-time logging and an end-of-run mission report |

---

## What I Learned

This project taught me:

**Robotics Fundamentals**
- Sensor integration and fusion
- Motor control and feedback loops
- Path planning in constrained environments

**Computer Vision**
- Color space analysis (RGB thresholding)
- Spatial segmentation techniques
- Real-time image processing

**Real-Time Systems**
- Timing-critical operations
- Sensor synchronization
- Control loop optimization

**Problem-Solving**
- Debugging with limited feedback
- Handling sensor noise and ambiguity
- Iterative algorithm refinement

**Software Engineering**
- Large codebase organization, and later splitting it into focused classes
- API design and abstraction
- Testing control code without the simulator, and measuring old and new versions the same way

---

## Future Improvements

If I were to extend this project, I would:

1. **Run the Webots comparison** of v1 and v2 and tune v2 from the results
2. **Dynamic Pathfinding** - Replace hardcoded waypoints with A* or Dijkstra algorithm. A first A* analysis finds routes about 14% shorter than the hand-tuned ones once the boxes are cleared, and no route at all without clearing
3. **Obstacle detection** - Use the distance sensors to slow down before hitting something
4. **SLAM Implementation** - Add simultaneous localization and mapping for unknown environments
5. **Machine Learning** - Train CNN for robust object detection in variable lighting
6. **Multi-Robot Coordination** - Enable swarm behavior for collaborative collection
7. **Real Hardware Deployment** - Port to actual Pioneer 3-DX or mobile manipulator

---

## Files & Resources

**Project Files:**
- `worlds/ProjectWorld2025.wbt` - the Webots world (v1); `ProjectWorld2025_v2.wbt` is the same world running v2
- `controllers/ProjectController3/` - the original controller (v1)
- `controllers/JarCollectorV2/` - the rewritten controller (v2)
- `protos/` - the Pioneer 3-DX robot and gripper models
- `tests/`, `tools/` - unit tests, the robot model, benchmarks and the Webots comparison script

**Tools & Technologies:**
- Webots R2025a - Professional robotics simulator
- Java - Control system implementation
- JUnit 5, JaCoCo, SpotBugs, GitHub Actions - testing and CI
- Git - Version control

**How to Run:**
1. Install Webots R2025a from [https://cyberbotics.com](https://cyberbotics.com)
2. Build the controllers once with `tools/build-controllers.sh` (Webots doesn't compile Java controllers by itself)
3. Open `worlds/ProjectWorld2025.wbt` (v1) or `worlds/ProjectWorld2025_v2.wbt` (v2) in Webots
4. Press Play to start the simulation

---

## Takeaway

This project demonstrates end-to-end robotics development: from hardware design and sensor selection, through algorithm development and testing, to successful autonomous operation. Coming back to it later, I kept the original intact, measured its weaknesses, and fixed them in a tested rewrite. It showcases my ability to work with complex systems, solve real-time constraints, and improve a working system without losing track of what it did before.
