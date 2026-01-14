---
layout: page
title: Improving MARL Robustness for Agent Blindness in VMAS
permalink: /projects/vmas-blindness/
---

# Improving MARL Robustness for Agent Blindness in VMAS

## Quick Summary

A Multi-Agent Reinforcement Learning research project investigating how swarm agents handle partial observability through random "blindness" events. Built custom blindness scenarios in VMAS (Vectorized Multi-Agent Simulator) where agents lose sensory input unpredictably, testing whether cooperative strategies can emerge despite information loss. Applied MAPPO (Multi-Agent Proximal Policy Optimization) with extensive hyperparameter tuning to achieve robust performance even when agents randomly lose observations.

**Tech Stack:** Python, PyTorch, TorchRL, VMAS, MAPPO, Multi-Agent Deep RL  
**Status:** ✅ Complete - Course Capstone Project  
**GitHub:** [View Source Code](#)

---

## The Challenge

**The Real-World Problem:** Autonomous systems must cooperate even when sensors fail unexpectedly.

**Examples:**
- **Autonomous vehicles:** Camera systems malfunction mid-drive—must avoid collisions using limited data
- **Drone swarms:** GPS signal loss—drones must maintain formation without position data
- **Robot teams:** Sensor failures—robots must complete tasks despite missing observations
- **Satellite networks:** Communication blackouts—satellites must coordinate blindly

**The Research Question:** Can multi-agent reinforcement learning systems learn to cooperate robustly when agents randomly lose sensory input (go "blind")?

---

## System Architecture

### VMAS Framework Selection

**What is VMAS?**
- **V**ectorized **M**ulti-**A**gent **S**imulator
- GPU-accelerated physics simulator for MARL research
- Developed by Prorok Lab (University of Cambridge)
- Supports continuous control tasks with swarm-like behavior

**The Balance Scenario:**

The base environment features agents cooperating to roll a red ball to a green goal by collectively pushing a platform the ball rests on.

**Key Mechanics:**
- **Physics:** Realistic gravity, momentum, balance dynamics
- **Cooperation Required:** One agent can't succeed alone—must coordinate pushing
- **Swarm Behavior:** Parameter sharing across all agents (homogeneous policy)
- **Full Observability (baseline):** All agents see complete state

[View Balance Scenario GIF](https://github.com/matteobettini/vmas-media/raw/main/media/scenarios/balance.gif?raw=true)

### Custom Blindness Extension

**Core Innovation:** Random sensory deprivation during episodes

**Six Blindness Scenarios Implemented:**

**1. BlindOneRandomAgentEveryStep**
```python
class BlindOneRandomAgentEveryStep(Transform):
    def _step(self, tensordict, next_tensordict):
        # One random agent blinded every single step
        next_tensordict[("agents", "observation")][
            ..., random.randrange(self._n_agents), :
        ] = 0
        return next_tensordict
```

**2. BlindAllAgentsEveryStep**
- All agents simultaneously blinded every step
- Extreme difficulty baseline (near-impossible task)

**3. BlindOneRandomAgentIfProbability**
```python
class BlindOneRandomAgentIfProbability(Transform):
    def __init__(self, n_agents, blind_prob=0.1):
        self._blind_prob = blind_prob
    
    def _step(self, tensordict, next_tensordict):
        if random.random() < self._blind_prob:
            # 10% chance one agent goes blind for 1 step
            next_tensordict[("agents", "observation")][
                ..., random.randrange(self._n_agents), :
            ] = 0
        return next_tensordict
```

**4. BlindRandomAgentsIfProbability**
- Multiple agents can be blinded simultaneously
- Each agent has independent probability per step

**5. BlindOneRandomAgentIfProbabilityForJSteps**
```python
class BlindOneRandomAgentIfProbabilityForJSteps(Transform):
    def __init__(self, n_agents, blind_prob=0.1, max_blind_steps=10):
        self.blind_remaining = {i: 0 for i in range(n_agents)}
    
    def _step(self, tensordict, next_tensordict):
        for agent_idx in range(self._n_agents):
            if self.blind_remaining[agent_idx] > 0:
                # Agent still blind, decrement counter
                self.blind_remaining[agent_idx] -= 1
                next_tensordict[("agents", "observation")][
                    ..., agent_idx, :
                ] = 0
            elif random.random() < self._blind_prob:
                # New blindness event: 1-10 steps
                blind_duration = random.randint(1, self.max_blind_steps)
                self.blind_remaining[agent_idx] = blind_duration - 1
                next_tensordict[("agents", "observation")][
                    ..., agent_idx, :
                ] = 0
        return next_tensordict
```

**6. BlindRandomAgentsIfProbabilityForJSteps**
- Multiple agents can experience multi-step blindness simultaneously
- Most realistic scenario (models real sensor failures)

**Blindness Mechanism:**
- **Observation vector set to zeros** (agent receives no sensory input)
- **Actions still processed** (agent can move/act, just blindly)
- **Other agents unaffected** (only blinded agent loses vision)

---

## Algorithm Highlights

### 1. MAPPO (Multi-Agent Proximal Policy Optimization)

**Why MAPPO?**
- State-of-the-art for cooperative MARL benchmarks
- Outperforms many off-policy methods (MADDPG, QMIX)
- Stable training with centralized critic, decentralized execution
- Parameter sharing enables swarm-like behavior

**Architecture:**

```python
Policy Network (Decentralized Actor):
├── MultiAgentMLP
│   ├── Input: Agent observation (n_obs_per_agent)
│   ├── Hidden: 2 layers × 256 units (Tanh activation)
│   ├── Output: 2 × n_actions (mean & std for continuous actions)
│   └── Share parameters: True (all agents use same policy)
├── NormalParamExtractor (split output into loc & scale)
└── TanhNormal distribution (bounded continuous actions)

Critic Network (Centralized Value Function):
├── MultiAgentMLP
│   ├── Input: All agent observations (centralized)
│   ├── Hidden: 2 layers × 256 units (Tanh activation)
│   ├── Output: 1 value per agent
│   └── Share parameters: True
└── State value estimation for PPO advantage
```

**Key MAPPO Features:**
- **Centralized Critic:** Sees all agent observations during training
- **Decentralized Actor:** Each agent acts from own observation only
- **Parameter Sharing:** Single policy network shared across all agents
- **PPO Clipping:** Prevents drastic policy updates (stability)

### 2. Hyperparameter Configuration

**Sampling Parameters:**
```python
frames_per_batch = 6_000  # Collected experiences per iteration
n_iters = 25              # Training iterations
total_frames = 150_000    # Total training frames
max_steps = 200           # Episode length
num_vmas_envs = 30        # Parallel environments
```

**Training Parameters:**
```python
num_epochs = 20           # Optimization passes per iteration
minibatch_size = 200      # Batch size for gradient updates
lr = 3e-4                 # Learning rate (Adam optimizer)
max_grad_norm = 1.0       # Gradient clipping
```

**PPO-Specific:**
```python
clip_epsilon = 0.2        # PPO clipping range
gamma = 0.9               # Discount factor
lmbda = 0.9               # GAE lambda
entropy_eps = 1e-4        # Entropy bonus coefficient
```

### 3. Normalization Strategy

**Problem:** Agents experience wildly different scenarios
- Blinded agent: Receives no reward temporarily
- Non-blinded agents: Continue collecting rewards
- Variance in returns destabilizes training

**Solution:** Advantage normalization across agents

```python
loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    normalize_advantage=True  # Critical for blindness scenarios
)
```

**Impact:**
- Prevents blinded agents from dominating gradient updates
- Stabilizes learning despite observation variance
- Essential for scenarios with random blindness events

### 4. GAE (Generalized Advantage Estimation)

**Purpose:** Estimate how much better an action was than expected

```python
loss_module.make_value_estimator(
    ValueEstimators.GAE, 
    gamma=0.9,   # Future reward discount
    lmbda=0.9    # Bias-variance tradeoff
)
```

**Why GAE for Blindness:**
- Smooths advantage estimates across uncertain states
- Reduces variance when agents randomly lose observations
- Helps credit assignment: "Was my action good despite blindness?"

### 5. Training Loop Architecture

```python
def train_environment_variables(env, description, norm=True, 
                                clipVal=0.2, batchSize=200, 
                                numEpochs=20):
    # 1. Setup networks (policy + critic)
    policy = ProbabilisticActor(...)
    critic = TensorDictModule(...)
    
    # 2. Create data collector
    collector = SyncDataCollector(env, policy, frames_per_batch)
    
    # 3. Setup replay buffer
    replay_buffer = ReplayBuffer(storage=LazyTensorStorage(...))
    
    # 4. Create PPO loss module
    loss_module = ClipPPOLoss(...)
    
    # 5. Training loop
    for tensordict_data in collector:
        # Compute GAE advantages
        with torch.no_grad():
            GAE(tensordict_data, ...)
        
        # Store experiences
        replay_buffer.extend(tensordict_data.reshape(-1))
        
        # Multiple optimization epochs
        for _ in range(numEpochs):
            for _ in range(frames_per_batch // batchSize):
                # Sample minibatch
                subdata = replay_buffer.sample()
                
                # Compute losses
                loss_vals = loss_module(subdata)
                loss = (loss_vals["loss_objective"] + 
                        loss_vals["loss_critic"] + 
                        loss_vals["loss_entropy"])
                
                # Backprop and update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
        
        # Update collector policy
        collector.update_policy_weights_()
        
        # Log episode rewards
        episode_reward_mean = tensordict_data[
            ("next", "agents", "episode_reward")
        ][done].mean()
    
    return episode_reward_mean_list, policy
```

---

## Key Technical Achievements

✅ **6 Custom Blindness Scenarios** - Probabilistic, duration-based, single/multi-agent  
✅ **MAPPO Implementation** - Centralized critic with decentralized execution  
✅ **Advantage Normalization** - Critical for handling observation variance  
✅ **Extensive Hyperparameter Study** - 50+ experiments across 7 dimensions  
✅ **Vectorized Simulation** - 30 parallel environments for efficient data collection  
✅ **Parameter Sharing** - Single policy shared across all agents (swarm behavior)  
✅ **GAE Value Estimation** - Smooth advantage calculation for uncertain states  
✅ **Automated Experimentation** - Configurable training function for rapid iteration  

---

## Experimental Results

### Experiment 1: Baseline Comparison Across Blindness Scenarios

**Setup:** 7 environments, 25 iterations, no normalization, clip=0.2

**Results:**

| Environment | Final Reward | Difficulty |
|-------------|--------------|------------|
| Normal (no blindness) | ~450 | Baseline |
| Blind 1 agent randomly (1 step) | ~380 | Easy |
| Blind 1 agent every step | ~320 | Medium |
| Blind 1 agent random duration | ~280 | Hard |
| Blind random agents randomly | ~240 | Very Hard |
| Blind random agents random duration | ~180 | Extreme |
| Blind all agents every step | ~50 | Nearly Impossible |

**Key Finding:** Agents can learn despite blindness, but performance degrades with:
- Increased blindness frequency
- Longer blindness duration
- More agents affected simultaneously

**Surprising Result:** Random 1-step blindness performs reasonably well (380/450 = 84% of baseline) → suggests robust cooperative strategies

---

### Experiment 2: Normalization Impact

**Setup:** All 7 scenarios × 2 conditions (normalized vs. non-normalized)

**Results:**

**Normalized Advantage (True):**
- Blind 1 agent randomly (1 step): **~420** (↑ 40 from baseline)
- Blind 1 agent random duration: **~340** (↑ 60 from baseline)
- Blind random agents random duration: **~220** (↑ 40 from baseline)

**Non-Normalized Advantage (False):**
- Much higher variance in training curves
- Lower final performance across all scenarios
- More unstable learning (sudden drops)

**Conclusion:** **Normalization is essential** for blindness scenarios
- Deals with variance from agents experiencing different observation states
- Prevents blinded agents from dominating gradient updates with large errors
- Stabilizes training even in extreme conditions

**All future experiments use normalization enabled.**

---

### Experiment 3: PPO Clipping Value

**Setup:** Clip values = [0.01, 0.1, 0.2, 0.3, 0.5, 0.75]  
**Environment:** Blind 1 random agent for random duration (hardest single-agent scenario)

**Results:**

| Clip Value | Final Reward | Observations |
|------------|--------------|--------------|
| 0.01 | ~180 | Too restrictive, can't learn |
| 0.1 | ~220 | Slow learning |
| 0.2 | ~280 | Balanced (default) |
| **0.3** | **~360** | **Best performance** |
| 0.5 | ~310 | Unstable, overshoots |
| 0.75 | ~290 | Too aggressive updates |

**Key Insight:** clip=0.3 provides optimal balance
- Not too conservative (can learn important features)
- Not too aggressive (maintains stability)
- **+28% improvement over default 0.2**

**Why It Works:**
- Allows larger policy updates when agents discover how to compensate for blindness
- Still prevents catastrophic updates from outlier experiences
- Sweet spot for this level of environment stochasticity

---

### Experiment 4: Minibatch Size

**Setup:** Batch sizes = [10, 50, 100, 200, 300, 500, 1000]

**Results:**

| Batch Size | Final Reward | Training Speed |
|------------|--------------|----------------|
| 10 | ~340 | Fast epochs, noisy |
| 50 | ~380 | Good balance |
| **100** | **~400** | **Optimal** |
| 200 | ~360 | Default (good) |
| 300 | ~310 | Sample reuse issues |
| 500 | ~280 | Overfitting to batches |
| 1000 | ~260 | Severe overfitting |

**Key Finding:** Smaller batches (50-100) outperform default (200)

**Explanation:**
- **Too large:** Samples reused too often within epoch → overfitting
- **Too small:** High variance, unstable gradients
- **Sweet spot (50-100):** Fresh samples, stable gradients

**Trade-off:** 
- Batch=100: Slightly slower per iteration but better final performance
- Batch=200: Faster iteration but lower asymptotic reward

---

### Experiment 5: Number of Epochs

**Setup:** Epochs = [5, 10, 15, 20, 30, 50]

**Results:**

| Epochs | Final Reward | Observations |
|--------|--------------|--------------|
| 5 | ~240 | Underutilizes data |
| 10 | ~300 | Improvement |
| 15 | ~340 | Good |
| 20 | ~360 | Default (good) |
| **30** | **~420** | **Optimal** |
| 50 | ~425 | Marginal gain (+5) |

**Key Finding:** 30 epochs hits diminishing returns

**Explanation:**
- More epochs = more optimization steps per batch
- Complex blindness scenarios benefit from thorough learning
- 30 vs. 50: Only +5 reward but +66% training time

**Recommendation:** Use 30 epochs for best time/performance tradeoff

---

### Experiment 6: Blindness Probability Sensitivity

**Setup:** Probability = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]  
**Environment:** Blind 1 random agent for random duration

**Results (Normalized by Blindness Probability):**

```
Reward_normalized = Reward × P(blind) × E[blind_steps]
```

| P(blind) | Raw Reward | Normalized Reward | Relative Performance |
|----------|------------|-------------------|----------------------|
| 0.01 | ~440 | ~22 | 100% (baseline) |
| 0.05 | ~420 | ~105 | 95% |
| **0.10** | ~380 | ~190 | 86% |
| 0.20 | ~320 | ~320 | 73% |
| 0.30 | ~260 | ~390 | 59% |
| 0.50 | ~180 | ~450 | 41% |

**Key Insight:** Agents maintain proportional performance until P(blind) > 0.2
- Up to 20% blindness probability: Graceful degradation
- Beyond 30%: Sharp performance cliff
- Suggests learned strategies have robustness threshold

**Interpretation:** 
- Agents learn to **anticipate and compensate** for occasional blindness
- When blindness becomes dominant (>30%), cooperation breaks down
- Similar to human teams: tolerate occasional member unavailability, fail under chronic absences

---

### Experiment 7: Maximum Blindness Duration

**Setup:** Max blind steps = [1, 2, 3, 5, 7, 10, 20]  
**Environment:** P(blind) = 0.1 fixed

**Results (Normalized):**

| Max Steps | Raw Reward | Normalized Reward | Performance |
|-----------|------------|-------------------|-------------|
| 1 | ~420 | ~21 | 100% |
| 2 | ~400 | ~40 | 95% |
| 3 | ~380 | ~57 | 90% |
| **5** | ~350 | ~87.5 | 83% |
| 7 | ~320 | ~112 | 76% |
| **10** | ~280 | ~140 | 67% |
| 20 | ~180 | ~180 | 43% |

**Key Finding:** Performance degrades linearly with duration up to ~10 steps, then crashes

**Explanation:**
- Short blindness (1-5 steps): Agents maintain momentum, other agents compensate
- Medium blindness (7-10 steps): Coordination suffers, recovery possible
- Long blindness (20 steps): 10% of episode—too disruptive for recovery

**Real-World Parallel:** 
- Brief sensor glitch (1-5s): Vehicle can coast safely
- Extended failure (10-20s): Requires pulling over or emergency protocols

---

### Experiment 8: Number of Agents

**Setup:** n_agents = [2, 3, 4, 5, 7, 10]  
**Environment:** Blind 1 random agent randomly (10% prob, 1-10 steps)

**Results:**

| Agents | Final Reward | Observations |
|--------|--------------|--------------|
| 2 | ~280 | Hard (50% capacity loss when 1 blinded) |
| **3** | ~380 | Baseline (33% loss) |
| 4 | ~420 | Better (25% loss) |
| 5 | ~460 | Good (20% loss) |
| **7** | **~510** | **Optimal** |
| 10 | ~490 | Coordination overhead |

**Key Insight:** More agents improve robustness—to a point

**Explanation:**
- **2-3 agents:** One blind = significant capacity loss
- **4-7 agents:** Redundancy allows compensation
- **10 agents:** Coordination complexity outweighs redundancy benefits

**Sweet Spot:** 5-7 agents
- Enough redundancy to handle blindness
- Not so many that coordination becomes bottleneck

---

### Experiment 9: Best Combined Hyperparameters

**Setup:** Combine all improvements from previous experiments

**Configurations Tested:**

| Config | Norm | Clip | Batch | Epochs | Final Reward |
|--------|------|------|-------|--------|--------------|
| Original | False | 0.2 | 200 | 20 | ~280 |
| **Best** | True | 0.3 | 100 | 30 | **~520** |
| Best - Clip | True | 0.2 | 100 | 30 | ~480 |
| Best - Batch | True | 0.3 | 200 | 30 | ~490 |
| Best - Epochs | True | 0.3 | 100 | 20 | ~470 |

**Key Finding:** Combined improvements yield **+86% performance** over baseline

**Optimal Hyperparameters:**
```python
normalize_advantage = True
clip_epsilon = 0.3
minibatch_size = 100
num_epochs = 30
```

**Video Evidence:** Trained agents successfully balance platform and move ball to goal despite random multi-step blindness events

---

## Challenges & Solutions

### Challenge: VMAS Documentation Nightmare

**Problem:** "VMAS is one of the worst to work in"

**Specific Issues:**
- Sparse, incomplete documentation
- Conflicting library syntaxes (BenchMARL, TorchRL, VMAS)
- Examples use different APIs (can't copy-paste)
- Hours spent restructuring code to match VMAS syntax

**Example Frustration:**
```python
# BenchMARL syntax (doesn't work in TorchRL)
env = make_vmas_env(scenario="balance", ...)

# TorchRL syntax (required)
env = VmasEnv(scenario="balance", ...)
env = TransformedEnv(env, RewardSum(...))

# Different observation keys between libraries!
```

**Impact:** 60%+ of project time spent on framework compatibility vs. actual research

**Solution:**
- Found working TorchRL MAPPO tutorial
- Built everything from that single example
- Avoided BenchMARL/MADDPG code (incompatible)

**Lesson Learned:** Framework maturity matters—prioritize well-documented tools

---

### Challenge: Failed MADDPG/QMIX Implementation

**Problem:** Wanted to compare MAPPO to other algorithms

**Attempted:**
- **QMIX:** Couldn't translate from different library syntax
- **MADDPG:** Ran without errors but no learning occurred

**MADDPG Symptoms:**
- Rewards oscillate wildly
- No convergence after 25 iterations
- Likely implementation bug in state/action handling

**Decision:** Abandon other algorithms, focus on MAPPO mastery

**Trade-off:**
- Lost algorithmic comparison
- Gained depth in single algorithm with extensive hyperparameter study

**Future Work:** Use BenchMARL directly (if compatible with blindness transforms)

---

### Challenge: Partner Abandonment

**Problem:** Group project became solo project mid-semester

**Timeline:**
- Week 1-3: Collaborated on proposal
- Week 4-8: Minimal communication
- Week 9+: Complete radio silence

**Impact:**
- Reduced scope (no multi-algorithm comparison)
- Increased workload
- Stress of solo research project

**Coping Strategy:**
- Focused on depth over breadth
- Hyperparameter study replaces multi-algorithm study
- Quality over quantity

**Silver Lining:** Full ownership of codebase, complete understanding of implementation

---

### Challenge: Training Time Constraints

**Problem:** 25 iterations × 6,000 frames × 30 envs = slow

**Training Times:**
- Single run: ~15-20 minutes
- Full experiment (7 scenarios): ~2 hours
- Hyperparameter sweep (6 values): ~12 hours

**Solution:**
- Ran experiments overnight
- Prioritized most impactful hyperparameters
- Used GPU acceleration (when available)

**Limitation:** Couldn't test extreme scales (50+ iterations, 100+ agents)

---

## Code Architecture

### Main Components

```
Morgans4900Project.py (1,500+ lines)
├── Imports & Setup
│   ├── PyTorch, TorchRL, VMAS
│   ├── Device configuration (GPU/CPU)
│   └── Random seed (reproducibility)
│
├── Hyperparameters
│   ├── Sampling (frames, iterations, envs)
│   ├── Training (epochs, batch size, learning rate)
│   ├── PPO (clip, gamma, lambda, entropy)
│   └── Environment (max steps, agents)
│
├── Environment Creation
│   ├── VmasEnv (base Balance scenario)
│   ├── TransformedEnv (add RewardSum)
│   └── 6 Blindness Variants (env2-env7)
│
├── Custom Transforms (6 classes)
│   ├── BlindOneRandomAgentEveryStep
│   ├── BlindAllAgentsEveryStep
│   ├── BlindOneRandomAgentIfProbability
│   ├── BlindRandomAgentsIfProbability
│   ├── BlindOneRandomAgentIfProbabilityForJSteps
│   └── BlindRandomAgentsIfProbabilityForJSteps
│
├── Training Function
│   ├── train_environment_variables()
│   │   ├── Network setup (policy + critic)
│   │   ├── Collector & replay buffer
│   │   ├── PPO loss module
│   │   ├── Training loop
│   │   └── Return rewards & policy
│
└── Experiments (10 total)
    ├── Exp 1: Baseline comparison (7 scenarios)
    ├── Exp 2: Normalization (True/False)
    ├── Exp 3: Clipping (6 values)
    ├── Exp 4: Batch size (7 values)
    ├── Exp 5: Epochs (6 values)
    ├── Exp 6: Blindness probability (6 values)
    ├── Exp 7: Max blind steps (7 values)
    ├── Exp 8: Number of agents (6 values)
    ├── Exp 9: Other scenarios (5 scenarios)
    └── Exp 10: Best combined parameters
```

### Design Patterns Used

**1. Transform Pattern (VMAS/TorchRL):**
```python
class BlindTransform(Transform):
    def _step(self, tensordict, next_tensordict):
        # Modify observations mid-rollout
        next_tensordict[("agents", "observation")][...] = 0
        return next_tensordict
```

**2. Functional Training:**
```python
def train_environment_variables(env, description, **kwargs):
    # Configurable training function
    # Returns: rewards_list, trained_policy
```

**3. Experiment Loop:**
```python
for env, desc in environments:
    for hyperparam in hyperparams:
        rewards, policy = train_environment_variables(...)
        results.append({...})
# Batch plotting after all experiments
```

**4. Automated Visualization:**
- Every experiment generates matplotlib plot
- Rollout visualization with trained policies
- Consistent color coding across experiments

---

## What I Learned

This project taught me:

**Multi-Agent Reinforcement Learning**
- MAPPO algorithm and centralized training, decentralized execution
- Parameter sharing for swarm-like homogeneous policies
- Advantage normalization critical for variance management
- GAE for smooth advantage estimation in stochastic environments

**Hyperparameter Sensitivity**
- Clipping value has non-monotonic relationship with performance
- Batch size sweet spot (too large = overfitting, too small = noise)
- Epochs: diminishing returns after threshold
- Normalization: not optional for partial observability

**VMAS Framework (Despite Frustration)**
- Vectorized simulation for parallel data collection
- Custom transforms for environment modifications
- TorchRL integration (after much struggle)
- GPU acceleration for faster training

**Research Methodology**
- Ablation studies (change one variable at a time)
- Baseline comparisons (always test default)
- Normalized metrics (account for scenario difficulty)
- Visual evidence (rollout videos validate learning)

**Robustness & Real-World AI**
- Partial observability is realistic and challenging
- Systems need redundancy (more agents help)
- Performance gracefully degrades until threshold
- Generalization doesn't come free (scenario-specific policies)

**Soft Skills**
- Working solo on group project
- Adapting scope when constraints appear
- Prioritizing experiments with limited time
- Honest acknowledgment of limitations

---

## Future Improvements

If I were to extend this research, I would:

1. **Fix Other Scenarios** - Debug blindness implementation for Wheel, Give Way, etc.
2. **Implement MADDPG/QMIX Properly** - Use BenchMARL for fair algorithm comparison
3. **Communication Protocols** - Allow agents to signal "I'm blind, help!"
4. **Partial Blindness** - Noisy observations instead of complete zeros
5. **Dynamic Blindness Models** - Learn to predict when blindness will occur
6. **Multi-Task Training** - Train on multiple scenarios simultaneously
7. **Longer Training** - 100+ iterations to see if performance plateaus
8. **Attention Mechanisms** - Let agents attend to non-blind teammates
9. **Curriculum Learning** - Gradually increase blindness difficulty
10. **Real Robot Deployment** - Test on actual drone/robot swarms
11. **Adversarial Blindness** - Opponent strategically blinds agents
12. **Meta-Learning** - Train to adapt quickly to new blindness patterns

---

## Research Impact & Applications

### Key Findings Summary

**1. Robustness is Learnable:**
- Agents can cooperate despite 10-20% blindness probability
- Performance degrades gracefully until ~30% threshold
- Suggests potential for real autonomous systems

**2. Hyperparameters Matter More Than Expected:**
- Optimal config: +86% performance over default
- Normalization absolutely essential (not optional)
- Clipping sweet spot (0.3) specific to stochasticity level

**3. Redundancy Helps (Up to a Point):**
- 5-7 agents optimal for Balance scenario
- Too few = no redundancy, too many = coordination overhead
- Real-world implication: design for sweet spot

**4. Generalization Fails:**
- Balance-trained policies don't transfer to other tasks
- Suggests need for multi-task training or meta-learning

### Real-World Applications

**1. Autonomous Vehicle Fleets:**
- Handle camera/sensor failures gracefully
- Maintain collision avoidance despite partial blindness
- Coordinate lane changes when one car loses GPS

**2. Drone Swarms:**
- Continue formation flight during communication blackouts
- Compensate for drones with malfunctioning sensors
- Search-and-rescue with unreliable equipment

**3. Satellite Networks:**
- Maintain orbit coordination during solar flares (sensor noise)
- Handle antenna failures in constellation
- Redundant communication pathways

**4. Industrial Robot Teams:**
- Factory robots handle vision system glitches
- Warehouse robots coordinate despite localization errors
- Construction robots compensate for failed team members

**5. Underwater/Space Exploration:**
- Robots in harsh environments with frequent sensor failures
- Redundant sensing across team members
- Mission completion despite partial information loss

---

## Files & Resources

**Project Files:**
- `Morgans4900Project.py` - Complete implementation (1,500+ lines)
- `COMP 4900 Project doc.pdf` - Presentation slides and documentation
- Training logs and plots (generated during experiments)

**Framework Requirements:**
- **Python** 3.8+
- **PyTorch** 1.13+
- **TorchRL** 0.2.0
- **VMAS** (Vectorized Multi-Agent Simulator)
- **TensorDict** (data structure for RL)
- **Matplotlib** (visualization)

**Installation:**
```bash
# Install PyTorch (with CUDA if available)
pip install torch torchvision torchaudio

# Install TorchRL and dependencies
pip install torchrl
pip install tensordict

# Install VMAS
pip install vmas

# Additional dependencies
pip install matplotlib tqdm
```

**Running Experiments:**
```python
# Single training run
rewards, policy = train_environment_variables(
    env=env6,  # Blind 1 random agent for random steps
    description="Blindness Test",
    norm=True,
    clipVal=0.3,
    batchSize=100,
    numEpochs=30
)

# Visualize trained policy
with torch.no_grad():
    env.rollout(
        max_steps=200,
        policy=policy,
        callback=lambda env, _: env.render()
    )
```

**Key Hyperparameters (Optimal):**
```python
# Sampling
frames_per_batch = 6_000
n_iters = 25
max_steps = 200
num_vmas_envs = 30

# Training
num_epochs = 30          # ← Increased from 20
minibatch_size = 100     # ← Decreased from 200
lr = 3e-4

# PPO
clip_epsilon = 0.3       # ← Increased from 0.2
normalize_advantage = True  # ← Critical!
gamma = 0.9
lmbda = 0.9
```

---

### References & Prior Work

**Key Papers:**

1. **Bettini et al. (2023)** - "VMAS: A Vectorized Multi-Agent Simulator for Collective Robot Learning"
   - Framework foundation
   - Balance scenario design

2. **Yu et al. (2022)** - "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
   - Justification for MAPPO baseline
   - Benchmark performance comparisons

3. **Lowe et al. (2017)** - "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
   - MADDPG algorithm (attempted implementation)
   - Centralized training framework

4. **Lin et al. (2020)** - "On the Robustness of Cooperative Multi-Agent Reinforcement Learning"
   - Inspiration for robustness investigation
   - Adversarial MARL concepts

**Full citations in project documentation.**

---

## Takeaway

This project demonstrates that Multi-Agent Reinforcement Learning systems can be trained to handle realistic sensor failures through careful environment design and hyperparameter optimization. While agents don't achieve perfect robustness, they learn cooperative strategies that gracefully degrade under partial observability—exhibiting resilience patterns similar to human teams.

The research highlights critical gaps in MARL tooling (VMAS documentation, library compatibility) that consume researcher time, underscoring the need for better infrastructure in the MARL community. Despite these challenges and the constraint of working solo on a group project, the systematic hyperparameter study provides actionable insights for building robust multi-agent systems.

Most importantly, this work validates that **robustness can be learned, not just engineered**—agents discover emergent compensation strategies when teammates go blind, suggesting promising avenues for deploying cooperative AI in real-world scenarios where sensor reliability cannot be guaranteed.

The 86% performance improvement through hyperparameter optimization (normalization + optimal clip/batch/epochs) proves that careful tuning is not just beneficial but **essential** for MARL in stochastic environments with partial observability. This finding has direct implications for practitioners building real autonomous systems.
