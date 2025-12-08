---
layout: page
title: Can Agent Elimination Improve Cooperation in MARL?
permalink: /projects/marl-elimination/
---

## Quick Summary

An honours research project investigating whether agent elimination mechanisms can improve cooperation in Multi-Agent Reinforcement Learning (MARL) environments where individual greediness is rewarded but group cooperation is optimal. Built a custom Unity ML-Agents environment with configurable elimination rules, testing how agents balance self-interest against the threat of being eliminated by peers.

**Tech Stack:** Unity ML-Agents, C#, Python, Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC)  
**Status:** ✅ Research Complete - Published Honours Project  
**Supervisor:** Dr. Zinovi Rabinovich, Carleton University  
**GitHub:** [View Source Code](#)

---

## The Challenge

Multi-Agent Reinforcement Learning faces a fundamental problem: **How do you encourage cooperation when being greedy helps individuals but hurts the group?**

This is the classic **Sequential Social Dilemma (SSD)**—agents must choose between:
- **Cooperate:** Share resources, ensuring long-term sustainability (good for group)
- **Defect:** Grab as much as possible, depleting resources (good for individual)

Real-world examples:
- Overfishing depletes fish populations
- Traffic congestion from selfish driving
- Corporate competition destroying shared markets
- Environmental resource depletion

**The Research Question:** Can allowing agents to eliminate each other create a "fear of punishment" that encourages cooperation?

---

## Inspiration: Nature, Society, and *Limitless*

### Natural & Social Precedent

In nature and human societies, elimination serves as a deterrent:
- **Animal kingdoms:** Predators eliminate threats to resources
- **Human justice:** Criminal punishment deters harmful behavior  
- **Social systems:** Ostracism removes bad actors from communities

### The *Limitless* Connection

Inspired by the 2011 film *Limitless*, where Bradley Cooper's character gains superhuman intelligence and masters day trading by analyzing not just financial data, but **rumors, sentiment, and social dynamics**.

**Key Insight:** Markets (and multi-agent systems) aren't just about numbers—they're about **understanding what others will do** and **how to influence behavior**.

[Watch the Trading Scene](https://www.youtube.com/watch?v=XMQC01n7hHo)

---

## System Architecture

### Environment Design Evolution

**Original Plan: MeltingPot (Google DeepMind)**
- Started with MeltingPot's **Harvest environment**
- Features: Apple respawning based on local density (cooperation rewards)
- Problem: Sparse documentation, incompatible with popular RL libraries
- Result: **3 months of development abandoned** due to training failures

**Final Solution: Unity ML-Agents**
- Built on Unity's **FoodCollector** example environment
- Advantages: Tight integration, clear documentation, visual debugging
- Added custom elimination mechanics to study cooperation

### Custom Environment Features

**Arena Layout:**
- 40×40 unit bounded area
- Multiple agents (1-10 configurable)
- Good food (green, +1 reward)
- Bad food (red, -1 reward)
- Laser weapons with elimination mechanics

**Resource Dynamics:**
- Food respawns based on local density (inspired by Harvest)
- Respawn probability: P = proximityFactor × nearbyFoodCount
- Encourages sustainable harvesting vs. greedy depletion
- Creates natural tension between individual gain and group benefit

**Elimination Mechanics:**
```
Configuration Options:
├── No Elimination (baseline control)
├── Individual Elimination (1 agent can eliminate another)
├── Pair Elimination (2 agents needed)
├── Majority Elimination (3+ agents needed)
└── Temporary Freeze (10 second timeout) vs. Permanent Elimination
```

---

## Algorithm Highlights

### 1. Agent Observation System

**Traditional FoodCollector:** Simple grid sensor for food detection

**The Enhanced System:** 20+ normalized observations

```csharp
public override void CollectObservations(VectorSensor sensor)
{
    // Agent's own state
    Vector3 localVelocity = transform.InverseTransformDirection(
        m_AgentRb.velocity
    ) / 10f;
    sensor.AddObservation(localVelocity.x);
    sensor.AddObservation(localVelocity.z);
    sensor.AddObservation(m_Frozen ? 1.0f : 0.0f);
    
    // Weapon cooldown (normalized)
    float cooldownNormalized = Mathf.Clamp01(
        (Time.time - lastFiredTime) / laserCooldown
    );
    sensor.AddObservation(cooldownNormalized);
    
    // Spatial awareness
    float distanceToNearestFood = closestFood ? 
        Vector3.Distance(closestFood.transform.position, transform.position) 
        / m_MyArea.areaRange : 1.0f;
    sensor.AddObservation(distanceToNearestFood);
    
    // Other agents (for each agent)
    foreach (var agent in agents)
    {
        Vector3 relativePosition = (agent.transform.position - transform.position) 
            / m_MyArea.areaRange;
        sensor.AddObservation(relativePosition.x);
        sensor.AddObservation(relativePosition.z);
        sensor.AddObservation(agent.m_Frozen ? 1.0f : 0.0f);
        sensor.AddObservation(agent.m_PermanentlyEliminated ? 1.0f : 0.0f);
        sensor.AddObservation(hitTracker[agent].Count / 5f);
    }
}
```

**Why This Matters:**
- Agents see who's frozen, who's eliminated, who's aggressive
- Can make strategic decisions: "Do I help eliminate the greedy agent?"
- Tracks hit counts → learns who's a threat
- Normalized values → faster PPO convergence

### 2. Elimination Tracking System

**Problem:** How do agents know who to eliminate?

**Solution:** Distributed hit tracking

```csharp
private static Dictionary<FoodCollectorAgent, HashSet<FoodCollectorAgent>> 
    hitTracker = new Dictionary<FoodCollectorAgent, HashSet<FoodCollectorAgent>>();

void TrackHit(FoodCollectorAgent targetAgent, FoodCollectorAgent shooter)
{
    if (!hitTracker.ContainsKey(targetAgent))
    {
        hitTracker[targetAgent] = new HashSet<FoodCollectorAgent>();
    }
    
    if (!hitTracker[targetAgent].Contains(shooter))
    {
        hitTracker[targetAgent].Add(shooter);
        CheckAgentStatus(targetAgent);
    }
}

void CheckAgentStatus(FoodCollectorAgent agent)
{
    int uniqueHitCount = hitTracker[agent].Count;
    
    if (uniqueHitCount >= agentsToEliminateThreshold && permanentlyEliminates)
    {
        agent.PermanentlyEliminate();
    }
    else if (uniqueHitCount >= agentsToFreezeThreshold)
    {
        agent.Freeze();
    }
}
```

**Key Insight:** Only unique shooters count → prevents one agent from spam-eliminating

### 3. Reward Structure Design

**Critical Balance:** Discourage random shooting, but allow strategic elimination

```csharp
Reward Structure:
├── Collect Good Food:     +1.0
├── Collect Bad Food:      -1.0
├── Fire Laser (miss):     -0.5  (discourages spam)
├── Zap Another Agent:     -5.0  (significant penalty)
└── Get Zapped:            -5.0  (victim also penalized)
```

**Why Both Shooter and Victim Penalized?**
- **Shooter:** "Was this elimination worth 5 food?"
- **Victim:** "My behavior got me eliminated—avoid this in future"
- Creates **strategic calculation** rather than random aggression

**Example Decision:**
```
Agent thinks: "If I eliminate this greedy agent, I gain access to 8 more food over time.
8 food × 1.0 reward = +8.0
Elimination penalty = -5.0
Net gain = +3.0 → WORTH IT"
```

### 4. Food Respawn Logic

**Inspired by DeepMind's Harvest Environment:**

```csharp
private float CalculateRespawnChance(Vector3 position, FoodCollectorArea area)
{
    int nearbyFoodCount = 0;
    
    foreach (var food in FindObjectsOfType<FoodLogic>())
    {
        if (Vector3.Distance(position, food.transform.position) <= respawnRadius)
        {
            nearbyFoodCount++;
        }
    }
    
    // Higher density increases respawn probability
    return Mathf.Clamp01(proximityFactor * nearbyFoodCount);
}

private IEnumerator PeriodicRespawn()
{
    while (true)
    {
        yield return new WaitForSeconds(respawnInterval);
        
        Vector3 randomPoint = GetRandomPosition();
        if (Random.value < CalculateRespawnChance(randomPoint, this))
        {
            SpawnFoodAtPosition(randomPoint);
        }
    }
}
```

**Mechanism:**
- Food spawns near other food (clusters)
- Depleted areas don't respawn (punishment for overharvesting)
- Creates natural "commons tragedy" scenario

### 5. Three RL Algorithms Tested

**Proximal Policy Optimization (PPO):**
- Industry standard for MARL
- Stable policy updates via clipping
- Fast convergence with curiosity-driven exploration

**Hyperparameters:**
```yaml
batch_size: 1024
buffer_size: 10240
learning_rate: 0.0002
time_horizon: 256  # Long-term strategy learning
curiosity_strength: 0.02  # Encourages novel tactics
hidden_units: 256
num_layers: 2
```

**Policy Optimization for Cooperative Agents (POCA):**
- PPO variant designed for cooperation
- Shared value function between agents
- Group reward optimization

**Soft Actor-Critic (SAC):**
- Off-policy algorithm with entropy maximization
- Encourages diverse exploration strategies
- Uses replay buffer → learns from historical eliminations

---

## Key Technical Achievements

✅ **Custom Unity Environment** - Modified FoodCollector with 2,000+ lines of C# code  
✅ **Configurable Elimination Rules** - 1-9 agents to eliminate, freeze vs. permanent  
✅ **Advanced Observation Space** - 20+ normalized features including agent states  
✅ **Hit Tracking System** - Distributed HashSet tracking unique shooters  
✅ **Resource Respawn Logic** - Density-based food regeneration (Harvest-inspired)  
✅ **Three RL Algorithms** - PPO, POCA, SAC with custom hyperparameters  
✅ **ScriptableObject Architecture** - Modular agent settings for rapid experimentation  
✅ **Real-Time Visualization** - Material changes for frozen/eliminated states  

---

## Experimental Results

### Experiment 1: Number of Agents to Eliminate

**Setup:** Varied from 0 (no elimination) to 4 agents needed

**Surprising Result:** Individual elimination (1 agent) performed **better** than group elimination!

**Cumulative Reward (500K steps):**
- No Elimination: ~12,000 (baseline)
- 1 Agent to Eliminate: ~11,500 (2nd best)
- 2 Agents to Eliminate: ~10,000
- 3 Agents to Eliminate: ~9,500
- 4 Agents to Eliminate: ~9,000

**Why This Happened:**
- Agents learned to eliminate each other **early**
- Fewer agents → less overharvesting → more food respawns
- Small elimination penalty (-5) outweighed by increased food access
- **Strategy:** "Kill first, harvest alone" rather than "cooperate to eliminate greed"

**Entropy Decay:**
- All configurations showed decreasing entropy (exploration → exploitation)
- Agents found stable strategies within 200K steps
- Value loss stabilized → confident in elimination tactics

### Experiment 2: RL Algorithm Comparison

**Setup:** PPO vs. POCA vs. SAC (3 agents to eliminate)

**Results:**
- **SAC:** ~11,800 cumulative reward (BEST)
- **PPO:** ~10,500 cumulative reward
- **POCA:** ~10,200 cumulative reward

**Why SAC Won:**
- **Entropy maximization** → explored diverse elimination strategies
- **Replay buffer** → learned from past cooperation/defection patterns
- **Off-policy** → didn't forget old strategies when environment changed

**Drawback:** SAC trained 3× slower (critical limitation for research timeline)

**Extrinsic Reward Stability:**
- SAC: Stable from 100K steps
- PPO/POCA: Oscillated until 300K steps
- Faster convergence = more efficient learning

### Experiment 3: Freeze vs. Elimination

**Setup:** 1 agent to freeze (10 seconds) vs. 1 agent to permanently eliminate

**Results:**
- **Elimination:** ~11,500 cumulative reward
- **Freezing:** ~10,800 cumulative reward

**Why Elimination Performed Better:**
- Freezing created **revenge cycles** (frozen agents retaliate)
- Constant firing → more penalties
- Elimination removed problem agents permanently

**Curiosity Loss:**
- Elimination: Decreased steadily (learned stable strategy)
- Freezing: Remained high (continued exploring due to complexity)

**Interpretation:** Like real justice systems—temporary punishment (jail) doesn't always reform behavior

### Experiment 4: Sparse vs. Rich Resources

**Setup:** 10 food vs. 30 food (2 agents to eliminate)

**Results:**
- **10 Food:** ~8,500 cumulative reward
- **30 Food:** ~10,000 cumulative reward (baseline)

**Analysis:**
- Slightly better cooperation in sparse conditions
- Forced sustainable harvesting due to scarcity
- Not statistically significant enough for strong conclusions

---

## Challenges & Solutions

### Challenge: 3 Months Lost on MeltingPot

**Problem:** Google DeepMind's MeltingPot had poor documentation, incompatible with RLlib/PettingZoo/Gym

**Attempted Solutions:**
- Read source code to understand internals
- Tried 10+ different library integration approaches
- Posted on forums (limited community)

**Final Decision:** Abandon MeltingPot, switch to Unity

**Impact:** 
- Lost 75% of research timeline
- Rebuilt environment from scratch
- **Lesson learned:** Prioritize tool maturity and community support

---

### Challenge: Training Time Constraints

**Problem:** Complex observation space (20+ features) → slow training

**Training Times:**
- PPO: ~8 hours for 500K steps
- SAC: ~24 hours for 500K steps

**Solution:** 
- Reduced episode length initially
- Ran overnight training sessions
- Prioritized PPO for most experiments (efficiency)

**Future Improvement:** Cloud GPU training (AWS, Google Colab)

---

### Challenge: Agents Eliminated for "Wrong" Reasons

**Problem:** Agents eliminated each other to reduce competition, not punish greed

**Why This Happened:**
- Individual reward (+1 per food) > elimination penalty (-5)
- "Kill early, harvest alone" more profitable than cooperation
- No way to distinguish greedy agents from normal agents

**Attempted Solution:** Adjusted penalty structure (-0.5, -5.0, -10.0)

**Result:** Only changed breakeven point, not fundamental strategy

**Future Solution:** 
- Track food collection rates as observable feature
- Reward eliminations only when target is above average consumption
- Explicit "greed metric" in observations

---

### Challenge: Observation Space Explosion

**Problem:** Tracking all agents → N² observations

**Example:** 8 agents × 6 features each = 48 extra observations

**Solution:**
- Normalized all values to [-1, 1] or [0, 1]
- Used relative positions (not absolute)
- Stacked observations (4-frame history)

**Trade-off:** 
- Larger neural network (256 hidden units × 2 layers)
- More training data required
- But agents learned complex strategies

---

## Code Architecture

### Main Components

```
Unity Project Structure (C# Scripts)
├── Agent Logic
│   ├── FoodCollectorAgent.cs (500+ lines)
│   │   ├── Observation collection
│   │   ├── Action processing
│   │   ├── Laser firing logic
│   │   ├── Hit tracking
│   │   ├── Freeze/elimination mechanics
│   │   └── Reward processing
│   └── AgentSettings.cs (ScriptableObject)
│       └── Modular reward/penalty configuration
│
├── Environment Management
│   ├── FoodCollectorArea.cs (300+ lines)
│   │   ├── Agent spawning
│   │   ├── Food spawning
│   │   ├── Respawn coroutines
│   │   └── Area reset logic
│   ├── FoodCollectorSettings.cs
│   │   ├── Global score tracking
│   │   ├── Environment reset
│   │   └── Stats recording
│   └── FoodLogic.cs
│       └── Collision handling
│
├── Training Configuration
│   └── FoodCollectorPPO.yaml
│       ├── Hyperparameters
│       ├── Network architecture
│       └── Curiosity settings
│
└── Deprecated/Experimental
    ├── FoodRespawner.cs (commented out)
    ├── FoodResetAgent.cs
    └── CustomGridSensor.cs (commented out)
```

### Design Patterns Used

**1. ScriptableObject Pattern:**
```csharp
[CreateAssetMenu(fileName = "AgentSettings", menuName = "ScriptableObjects/AgentSettings")]
public class AgentSettings : ScriptableObject
{
    [Header("Rewards and Penalties")]
    public float rewardForZapping = 0.5f;
    public float rewardForBeingZapped = -0.5f;
    public float rewardForFiring = -0.1f;
    
    [Header("Thresholds")]
    public int agentsToFreezeThreshold = 1;
    public int agentsToEliminateThreshold = 3;
}
```
**Benefit:** Change parameters without recompiling, easy experimentation

**2. Singleton Pattern (Avoided):**
- Originally considered for FoodRespawner
- Removed due to Unity ML-Agents multi-environment conflicts
- Used per-area coroutines instead

**3. Observer Pattern:**
- Agents observe each other's states
- Hit tracker updates trigger state changes

**4. State Machine:**
```
Agent States:
Normal → Frozen (10 sec) → Normal
Normal → Eliminated → [removed from training]
```

---

## What I Learned

This honours project taught me:

**Multi-Agent Reinforcement Learning**
- Sequential Social Dilemmas and tragedy of the commons
- PPO, POCA, SAC algorithms and their trade-offs
- Importance of reward shaping (elimination penalties)
- Emergent behavior from simple rules

**Unity ML-Agents Development**
- C# scripting for agent behavior
- Observation and action space design
- Training configuration (YAML hyperparameters)
- TensorBoard visualization and debugging

**Research Methodology**
- Experimental design for MARL
- Baseline comparisons and ablation studies
- Statistical significance and graphing
- Academic writing and literature review

**Game Development**
- Unity physics and collision detection
- Coroutines for periodic spawning
- Material swapping for visual feedback
- ScriptableObjects for modular design

**Problem-Solving Under Constraints**
- Pivoting when tools don't work (MeltingPot → Unity)
- Working with limited compute resources
- Prioritizing experiments when time is limited
- Making research decisions with incomplete data

**Soft Skills**
- Working with academic supervisor
- Technical writing for research reports
- Presenting complex AI concepts clearly
- Acknowledging limitations honestly

---

## Future Improvements

If I were to extend this research, I would:

1. **Explicit Greed Detection** - Add "food per minute" metric to observations
2. **Communication Protocol** - Allow agents to signal intentions before eliminating
3. **Leader Election** - Democratic voting for "enforcer" agent with elimination rights
4. **Curriculum Learning** - Start simple (no elimination) → gradually add mechanics
5. **Longer Episodes** - 10,000 steps instead of 1,000 (capture long-term effects)
6. **Cloud Training** - AWS/GCP GPUs for faster SAC training
7. **Custom RL Algorithm** - Architecture specifically designed for SSD cooperation
8. **Multi-Environment Training** - Vary respawn rates during training for robustness
9. **Incremental Punishment** - Small penalty per timestep eliminated (like jail time)
10. **Other SSD Environments** - Test on Cleanup, Stag Hunt, Public Goods games
11. **Real-World Application** - Apply to traffic flow, resource allocation problems
12. **Human Studies** - Test with human players vs. AI agents

---

## Research Impact & Insights

### Key Finding: Elimination Creates New Problems

**Expected:** Agents eliminate greedy members → cooperation improves

**Reality:** Agents eliminate each other early → fewer competitors → more food

**Why This Matters:**
- Shows complexity of punishment in multi-agent systems
- Suggests need for sophisticated "justice" mechanisms
- Highlights gap between human social norms and RL optimization

### Contribution to MARL Literature

**Novel Aspect:** First study to systematically test elimination thresholds (1-4 agents)

**Confirmed Prior Work:**
- PPO effective for MARL (Guckelsberger 2018)
- Social influence matters (Jaques 2018)
- SSDs require long-term strategy learning (Leibo 2017)

**New Insight:** Entropy maximization (SAC) helps discover cooperative strategies faster

### Potential Applications

**1. Autonomous Vehicle Networks:**
- Eliminate aggressive drivers from autonomous fleets
- Enforce traffic rules through peer mechanisms

**2. Distributed Computing:**
- Remove nodes hogging resources in cloud systems
- Ensure fair resource allocation

**3. Robotics Swarms:**
- Self-policing behavior in robot teams
- Expel malfunctioning units automatically

**4. Economic Systems:**
- Model market regulation mechanisms
- Understand how punishment affects trader behavior

**5. Game AI:**
- Multiplayer games with community-driven moderation
- NPCs that enforce social norms

---

## Files & Resources

**Project Files:**
- `FoodCollectorAgent.cs` - Main agent logic (500+ lines)
- `FoodCollectorArea.cs` - Environment management (300+ lines)
- `AgentSettings.cs` - Configuration ScriptableObject
- `FoodCollectorSettings.cs` - Global environment settings
- `FoodLogic.cs` - Food item behavior
- `FoodCollectorPPO.yaml` - Training hyperparameters
- `4905 Project Report.pdf` - Full honours thesis (40+ pages)

**Unity Assets Required:**
- Unity 2021.3 LTS or later
- ML-Agents Package 2.3.0
- Python 3.8+ with mlagents library

**Training Setup:**
```bash
# Install ML-Agents
pip install mlagents==0.30.0

# Train PPO
mlagents-learn FoodCollectorPPO.yaml --run-id=elimination_test

# TensorBoard visualization
tensorboard --logdir results/
```

**Key Dependencies:**
```
Unity ML-Agents 2.3.0
TensorFlow 2.x
PyTorch 1.13+ (for SAC)
NumPy, Matplotlib (analysis)
```

**Hyperparameter Highlights:**
```yaml
PPO:
  batch_size: 1024
  buffer_size: 10240
  learning_rate: 0.0002
  time_horizon: 256
  curiosity_strength: 0.02
  max_steps: 500000
  
Network:
  hidden_units: 256
  num_layers: 2
  normalize: true
```

---

## Academic Context

### Supervisor

**Dr. Zinovi Rabinovich**
- School of Computer Science, Carleton University
- Research: Multi-agent systems, game theory, decision-making
- Provided guidance on experimental design and MARL theory

### Course

**COMP 4905 - Honours Project**
- 4th year capstone research project
- Fall 2024 semester
- Carleton University, Ottawa, Ontario

### Related Coursework

This project built on:
- **COMP 3106** - Artificial Intelligence (foundations)
- **COMP 4107** - Neural Networks (deep RL algorithms)
- **COMP 4905** - Honours Project (research methodology)

### References & Prior Work

**Key Papers Cited:**

1. **Leibo et al. (2017)** - "Multi-agent Reinforcement Learning in Sequential Social Dilemmas"
   - Introduced SSD framework
   - Harvest environment design principles

2. **Jaques et al. (2018)** - "Social Influence as Intrinsic Motivation for Multi-Agent Deep RL"
   - Influence modeling and cooperation rewards
   - Inspired penalty structure

3. **Guckelsberger et al. (2018)** - "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
   - Validated PPO for MARL baseline
   - Convergence properties

4. **Schulman et al. (2017)** - "Proximal Policy Optimization Algorithms"
   - PPO algorithm design
   - Clipped surrogate objective

5. **Haarnoja et al. (2018)** - "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
   - SAC algorithm and entropy regularization
   - Exploration strategies

**Full citations in project report.**

---

## Presentation & Documentation

### Deliverables

**1. Honours Thesis (40+ pages):**
- Abstract and motivation
- Literature review
- Methodology (MeltingPot failure + Unity pivot)
- Experimental results with 10+ graphs
- Limitations and future work
- Academic references

**2. Unity Project:**
- Fully functional MARL environment
- Configurable parameters via Inspector
- Training configurations for 3 algorithms
- Commented C# codebase

**3. Trained Models:**
- PPO checkpoint (500K steps)
- SAC checkpoint (300K steps)
- TensorBoard logs and graphs

**4. Defense Presentation:**
- 20-minute presentation to faculty
- Live demo of trained agents
- Q&A on methodology and results

---

## Takeaway

This honours research project demonstrates the complexity of engineering cooperation in multi-agent systems. While agent elimination did not achieve the hypothesized deterrent effect against greed, it revealed deeper insights into how punishment mechanisms can create unintended strategic dynamics.

The project showcases end-to-end research skills: literature review, environment design, algorithm implementation, experimental methodology, statistical analysis, and academic writing. By pivoting from MeltingPot to Unity under significant time constraints, it also demonstrates adaptability and problem-solving in the face of technical challenges.

Most importantly, the research highlights a critical gap in MARL: **agents optimize rewards, not ethics**. Teaching AI to cooperate like humans requires more than punishment—it requires understanding *why* cooperation emerges in nature and society, then encoding those incentives explicitly into reward structures.

This work contributes to the growing body of MARL research on Sequential Social Dilemmas and provides a foundation for future investigations into sophisticated cooperation mechanisms beyond simple elimination. The insights gained here are directly applicable to real-world multi-agent systems in autonomous vehicles, distributed computing, robotics, and economic modeling.
