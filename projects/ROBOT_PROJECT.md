---
layout: page
title: Autonomous Robot Jar Collection System
permalink: /projects/robot-project/
---

# Autonomous Robot Jar Collection System

## Quick Summary

An autonomous mobile robot built in Webots that uses computer vision, sensor fusion, and pathfinding algorithms to navigate a 20×20 meter warehouse, locate honey jars, collect them, and place them in designated storage areas—all without human intervention.

**Tech Stack:** Webots R2025a, Java, Computer Vision, Robotics  
**Status:** ✅ Complete and Operational  
**GitHub:** [View Source Code](#)

---

## Video Demo

<iframe width="100%" height="450" src="https://www.youtube.com/embed/kO9e8dzjCfg" title="Robot Project Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

## The Challenge

Design and implement a fully autonomous robot system that can:
- Navigate a complex warehouse environment with obstacles
- Detect and locate small objects using only onboard sensors
- Manipulate objects with a gripper system
- Return collected items to a designated drop-off zone
- Repeat this process for multiple objects

All of this needed to work without GPS, pre-mapped routes, or external guidance systems.

---

## System Architecture

### Hardware Configuration

The robot is built on a Pioneer 3-DX platform equipped with:

**Sensors:**
- 64×64 RGB Camera with color-based object detection
- 6 Ultrasonic Distance Sensors (obstacle avoidance)
- Compass (heading/orientation)
- Touch Sensor (object contact detection)
- Accelerometer (stability monitoring)

**Actuators:**
- Differential drive motors (left & right wheels)
- Gripper lift motor
- Gripper finger motors (left & right)

**Key Specs:**
- Max speed: 5 m/s
- Arena: 20m × 20m
- Jars collected: 4 (scalable)
- Obstacles handled: 30 cardboard boxes

---

## Algorithm Highlights

### 1. Color-Based Object Detection

The robot's camera divides its field of view into three zones and counts colored pixels:

**Green Detection (Jars):**
- Target signature: R < 80, G > 50, B < 80
- If green_left > green_center + 15px → Turn left
- If green_center dominant → Move forward
- If green_right dominant → Turn right

**Blue Detection (Drop-off Zone):**
- Target signature: R < 20, G < 20, B > 80
- Guides robot during return phase

### 2. Navigation & Pathfinding

The robot executes multiple phases with hardcoded waypoints optimized for the specific environment:

**Phase 1: Obstacle Clearing** (hardcodedPathPushing)
- Clears 30 cardboard boxes using 15+ waypoints
- Creates navigable pathways for collection phase

**Phase 2: Jar Detection & Collection** (findJar)
- Rotates and scans for green objects
- Compass validates forward-facing orientation
- Touch sensor confirms object proximity
- Engages gripper sequence upon contact

**Phase 3: Return Navigation** (toStart)
- Follows optimized 10-waypoint return path
- Reduced speed (1 m/s) during precision turns
- Full speed (5 m/s) on open terrain

**Phase 4: Placement & Release** (placeJar)
- Positions robot at storage bay
- Opens gripper to release jar
- Reverses away and increments counter

### 3. Turn Algorithm

```
Calculate target heading → Compare to compass reading 
→ Select shortest rotation → Adjust velocities dynamically
→ Hold until within ±3° of target
```

This ensures accuracy even when overshooting or dealing with compass noise.

### 4. Sensor Fusion

- **Vision** tells the robot where objects are
- **Compass** confirms proper orientation
- **Distance sensors** prevent collisions
- **Touch sensor** validates contact
- **Accelerometer** monitors stability

Multi-layered detection prevents false positives and ensures robust operation.

---

## Key Technical Achievements

✅ **Robust Object Detection** - Color thresholding with fallback logic handles variable lighting  
✅ **Precision Navigation** - Turn accuracy within ±3°, distance accuracy within ±10cm  
✅ **Gripper Synchronization** - Timed sequences ensure proper grip establishment  
✅ **Sensor Fusion** - Combines multiple sensors for redundant perception  
✅ **Real-Time Control** - 32ms control cycle with 64 fps camera processing  
✅ **Obstacle Avoidance** - 6 distance sensors prevent collisions  
✅ **State Management** - Clean separation between collection, navigation, and placement phases  

---

## Performance Results

| Metric | Value |
|--------|-------|
| Success Rate | 100% (4/4 jars collected) |
| Navigation Accuracy | ±3° (heading), ±10cm (position) |
| Gripper Engagement Time | ~3 seconds per object |
| Total Mission Time | ~5-7 minutes |
| Sensor Update Rate | 32-64 fps |
| Control Frequency | 30-31 Hz |
| Waypoint Count | 30+ total |

---

## Challenges & Solutions

### Challenge: Object Detection Ambiguity

**Problem:** Camera color detection could fail under poor lighting or angle variations

**Solution:** Implemented three-tier fallback detection logic:
1. Primary: Significant color magnitude difference (>15 pixels)
2. Secondary: Relative magnitude comparison
3. Tertiary: Raw pixel counts above threshold

**Result:** 100% detection rate across all test runs

---

### Challenge: Navigation Precision

**Problem:** Hardcoded waypoints required sub-centimeter accuracy in a large arena

**Solution:** 
- Real-time position monitoring from Supervisor API
- Early termination when distance threshold reached
- Compass-based turn validation with dynamic correction

**Result:** Consistent ±10cm accuracy throughout 20m × 20m space

---

### Challenge: Gripper Synchronization

**Problem:** Premature gripper opening could drop collected jars

**Solution:** Timed delay loops (100+ iterations) allow motor positions to stabilize before next operation

**Code Pattern:**
```java
int holding = 100;
while (holding > 0 && robot.step(timeStep) != -1) {
  holding--;
  // Motor operations continue while holding decrements
}
```

**Result:** Zero jar drops across all trials

---

### Challenge: Large Coordinate Space Management

**Problem:** 20m × 20m arena with floating-point precision issues

**Solution:** Convert all coordinates to centimeter units (×100) for integer-based comparisons

**Impact:** Eliminated rounding errors and improved navigation stability

---

## Code Architecture

### Main Components

```
ProjectController3.java (800+ lines)
├── Initialization
│   ├── Motor setup (wheels, gripper)
│   ├── Sensor setup (camera, compass, distance)
│   └── Field initialization
│
├── Navigation Methods
│   ├── makeHardcodedTurn()
│   ├── moveHardcodedAhead()
│   └── hardcodedPath()
│
├── Perception Methods
│   ├── countColor()
│   ├── findJar()
│   └── getCompassReadingInDegrees()
│
└── Control Methods
    ├── liftLowerGripper()
    ├── openCloseGripper()
    └── placeJar()
```

### Design Patterns Used

- **Supervisor Architecture:** Absolute position knowledge via Webots Supervisor API
- **State Machines:** Implicit states (clearing → collection → return → placement)
- **Hardware Abstraction:** Centralized sensor/motor initialization
- **Modular Methods:** Single responsibility for each function

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
- Large codebase organization
- API design and abstraction
- Performance optimization

---

## Future Improvements

If I were to extend this project, I would:

1. **Dynamic Pathfinding** - Replace hardcoded waypoints with A* or Dijkstra algorithm
2. **SLAM Implementation** - Add simultaneous localization and mapping for unknown environments
3. **Machine Learning** - Train CNN for robust object detection in variable lighting
4. **Multi-Robot Coordination** - Enable swarm behavior for collaborative collection
5. **Real Hardware Deployment** - Port to actual Pioneer 3-DX or mobile manipulator
6. **Force Feedback Gripper** - Implement pressure sensing for delicate object handling
7. **Behavior Trees** - More sophisticated state management for complex tasks

---

## Files & Resources

**Project Files:**
- `ProjectWorld2025.wbt` - Webots simulation world (complete 20×20m environment)
- `ProjectController3.java` - Main controller (800+ lines of Java)
- `Pioneer3dx.proto` - Robot base definition
- `Pioneer3Gripper.proto` - Gripper attachment

**Tools & Technologies:**
- Webots R2025a - Professional robotics simulator
- Java - Control system implementation
- Git - Version control

**How to Run:**
1. Install Webots R2025a from [https://cyberbotics.com](https://cyberbotics.com)
2. Open `ProjectWorld2025.wbt` in Webots
3. Press Play to start simulation
4. Watch the robot autonomously complete its mission!

---

## Takeaway

This project demonstrates end-to-end robotics development: from hardware design and sensor selection, through algorithm development and testing, to successful autonomous operation. It showcases my ability to work with complex systems, solve real-time constraints, and deliver a functional solution to a challenging problem.

