---
layout: post
title: "Building an Autonomous Warehouse Robot: From Concept to Complete System"
date: 2024-11-20 09:00:00 -0500
categories: robotics computer-vision systems-design
---

I recently completed an autonomous robot system that can navigate a warehouse, find objects, and return them to designated locations—all without human guidance. Here's how I did it.

## The Problem

Imagine you have a 20×20 meter warehouse filled with obstacles, scattered objects, and no GPS. Can you build a robot that can autonomously find specific items, pick them up, and bring them to a collection point? That was my challenge.

## The Solution: Multi-Sensor Fusion

Rather than relying on a single sensor, I combined:
- **Camera** for object detection (color-based)
- **Compass** for orientation
- **Distance sensors** for obstacle avoidance
- **Touch sensor** for contact confirmation

This redundancy made the system robust to individual sensor failures.

## Computer Vision Breakthrough

The key insight was dividing the camera's field of view into three zones:

```
┌──────────────┬──────────────┬──────────────┐
│   LEFT       │    CENTER    │    RIGHT     │
│  (0-21px)    │   (21-42px)  │  (42-64px)   │
└──────────────┴──────────────┴──────────────┘
```

For each zone, I counted colored pixels:
- **Green pixels** = honey jar location
- **Blue pixels** = drop-off zone

A simple comparison determined direction: if green_left > green_center + 15px, turn left. This approach was simple, fast, and incredibly effective.

## Navigation: The Hard Part

Getting a robot to move from point A to point B accurately sounds easy until you try it. Here's what I implemented:

### Turn Algorithm

```java
int targetAngle = calculateTargetHeading(currentPos, targetPos);
int currentAngle = compass.getHeading();
int turnAmount = (targetAngle - currentAngle + 360) % 360;

// Adjust turn amount to shortest rotation
if (turnAmount > 180) turnAmount -= 360;

// Rotate until within ±3 degrees
while (abs(compass.getHeading() - targetAngle) > 3) {
  if (turnAmount > 0) rotateLeft();
  else rotateRight();
}
```

The tricky part? Compass noise and motor overshoot. I solved this with dynamic velocity adjustment—as the robot gets closer to the target angle, I reduce motor speed.

### Movement Algorithm

Similar to turning, but for forward movement:

```java
double distance = sqrt((x2-x1)² + (y2-y1)²);
moveForward(MAX_SPEED);

while (distanceTraveled < distance) {
  // Monitor current position via Supervisor API
  // Stop when distance reached
}
```

## The Gripper Choreography

Picking up objects required precise timing:

```
1. Open gripper fully
2. Lower gripper to ground
3. Move forward to engage object
4. Close gripper (jar captured!)
5. Retract gripper upward
6. Move away
```

Each step neede
