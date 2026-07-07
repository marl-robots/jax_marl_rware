# 📘 Multi‑Agent RL Metrics Reference (Complete)

## A complete reference for all metrics used in the multi‑agent reinforcement learning system.  

---

# 📑 Table of Contents

1. [Overview](#overview)  
2. [Core Data Structures](#core-data-structures)  
3. [Global Utility Functions](#global-utility-functions)  
4. [Metric Groups](#metric-groups)  
   - [Performance](#performance)  
   - [Task Success](#task-success)  
   - [Coordination](#coordination)  
   - [Behaviour](#behaviour)  
   - [Temporal Efficiency](#temporal-efficiency)  
   - [Completion Curves](#completion-curves)  
   - [Fairness Outcomes](#fairness-outcomes)  
   - [Fairness Distributions per Agent](#fairness-distributions-per-agent)  
   - [Credit Correlations](#credit-correlations)  
   - [Credit Shapley](#credit-shapley)  
   - [Advantage Statistics](#advantage-statistics)  
   - [Value Statistics](#value-statistics)  
   - [Loss Statistics](#loss-statistics)  
   - [Loss Dynamics per Epoch](#loss-dynamics-per-epoch)  
   - [Entropy Global per Epoch](#entropy-global-per-epoch)  
   - [Entropy per Agent](#entropy-per-agent)  
   - [Action Distribution per Agent](#action-distribution-per-agent)  
   - [Action Similarity Agents](#action-similarity-agents)  
   - [KL Global](#kl-global)  
   - [KL per Parameter](#kl-per-parameter)  
   - [Gradient Global](#gradient-global)  
   - [Gradient per Agent](#gradient-per-agent-only-if-centralised-critic)  
   - [Rewards per Agent](#rewards-per-agent)  
   - [Reward Standardise](#reward-standardise)  
   - [Environment Metrics](#environment-metrics)  
   - [System Performance](#system-performance)  
   - [System Timing](#system-timing)  
5. [Event Definitions](#event-definitions)  
6. [Reward System](#reward-system)  
7. [Loss Definitions](#loss-definitions)  
8. [Temporal & Fraction Metrics](#temporal--fraction-metrics)  

---

# Overview

Metrics are computed:

- **Per‑episode** — one value per episode  
- **Per‑epoch** — aggregated across episodes in an epoch  
- **Per‑agent** — reveals imbalance and coordination  
- **Per‑environment** — reveals instability and heterogeneity  

> *"Episode return — sum of rewards per episode; shape: scalar."*  
> *"Team success rate — fraction of envs where all agents succeeded at least once."*

---

# Core Data Structures

### Axes
- **T** — time steps  
- **E** — environments  
- **N** — agents  

### Step‑level tensors
- `deliv_t[T,E,N]` — delivery indicator  
- `blocked_t[T,E,N]` — contention/collision proxy  
- `noop_t[T,E,N]` — idle/no‑op  
- `pickup_t[T,E,N]` — pickup action  
- `drop_t[T,E,N]` — drop action  
- `distance_traveled_t[T,E,N]` — forward movement count  
- `entropy_t[T,E,N]` — per‑agent policy entropy  
- `rstd_t[T,E,N]` — Welford‑standardised rewards  

### Timing tensors
- `step_time_t[T,E]` — elapsed time per env step  
- `step_count[E]` — number of steps per env  
- `episode_time[chunk]` — episode duration in update steps  

---

# Global Utility Functions

### frac_mean / frac_std

> *"Fraction = proportion / rate. Calculated as count / total values in range [0,1]."*

- **frac_mean** — mean proportion of event indicator over `[T,E,N]`  
- **frac_std** — variability of event rate  

### E_per_T_mean / E_per_T_std

> *"Measures the mean over time for each environment… reports temporal dispersion/instability."*

- **E_per_T_mean** — time‑averaged per‑env mean  
- **E_per_T_std** — temporal instability per env  

---

# Metric Groups

---

# Performance

### episode_returns_mean  
Mean episode return (sum of rewards per episode).  
> *"Episode return — sum of rewards per episode; shape: scalar."*

### episode_returns_std  
Std of episode returns; stability across episodes/envs.  
> *"Shows stability/variance across episodes."*

### returns_mean  
Mean returns across episodes.  
(Same definition as episode return but aggregated differently.)

### returns_std  
Std of returns; variability in performance.

### returns_percentile_stats_p10  
10th percentile return; worst‑case tail.  
> *"Percentiles (10/50/90)… expose tails/outliers."*

### returns_percentile_stats_p50  
Median return; robust central tendency.

### returns_percentile_stats_p90  
90th percentile return; best‑case tail.

### returns_skew  
Skewness of return distribution.  
> *"Skewness reveals asymmetric performance, often caused by unstable exploration or environment stochasticity."*

---

# Task Success

### success_mean  
Mean number of successful agents per env.  
> *"Mean number of agents that achieved at least one delivery."*

### success_std  
Variability in number of successful agents.

### success_rate_mean  
Fraction of envs where **all agents** succeeded.  
> *"Team success rate — fraction of envs where all agents succeeded at least once."*

### success_rate_std  
Variability in full‑team success.

### deliveries_mean  
Mean total deliveries per episode.  
> *"Throughput of the system; direct task performance measure."*

### deliveries_std  
Std of total deliveries; inconsistency across envs.

---

# Coordination

### deliveries_early_mean  
Early‑phase deliveries; fast coordination.  
> *"Early deliveries indicate rapid task allocation and initial coordination."*

### deliveries_early_std  
Variability in early coordination.

### deliveries_mid_mean  
Mid‑phase deliveries; sustained coordination.

### deliveries_mid_std  
Variability in mid coordination.

### deliveries_late_mean  
Late‑phase deliveries; finishing ability.

### deliveries_late_std  
Variability in late completion.

---

# Behaviour

### block_rate_mean  
Mean fraction of blocked steps.  
> *"Blocked_t… agent tried to step forward, lost movement contention — collision/contention proxy."*

### block_rate_std  
Variability in contention.

### block_early_mean  
Early blocking rate.

### block_early_std  
Variability in early blocking.

### block_mid_mean  
Mid blocking rate.

### block_mid_std  
Variability in mid blocking.

### block_late_mean  
Late blocking rate.

### block_late_std  
Variability in late blocking.

### idle_rate_mean  
Mean noop rate.  
> *"Noop_t… agent chose noop action."*

### idle_rate_std  
Variability in idleness.

### pickup_rate_mean  
Mean pickup rate.  
> *"Pickup_t… agent chose TOGGLE_LOAD on real shelf when not carrying."*

### pickup_rate_std  
Variability in pickup rate.

### distance_traveled_mean  
Mean movement load.  
> *"Distance_traveled_t… accumulative count of forward steps."*

### distance_traveled_std  
Variability in movement.

### distance_traveled_agent  
Per‑agent movement (N,).  
Reveals workload imbalance.

---

# Temporal Efficiency

### time_to_completion_mean  
Mean episode completion time.  
> *"Time to completion — identifies latency problems."*

### time_to_completion_std  
Variability in completion time.

### time_to_completion_p10  
Fastest 10% completion.

### time_to_completion_p50  
Median completion.

### time_to_completion_p90  
Slowest 10% completion.

### time_to_first_delivery_mean  
Mean latency to first delivery.  
> *"Time to first delivery — identifies early latency issues."*

### time_to_first_delivery_std  
Variability in first delivery latency.

### time_to_first_delivery_p10  
Fastest first deliveries.

### time_to_first_delivery_p50  
Median first delivery.

### time_to_first_delivery_p90  
Slowest first deliveries.

---

# Completion Curves

### episode_time  
Episode duration in update steps.

### step_count_mean  
Mean number of steps per env.

### time_to_completion_cdf_grid  
CDF grid of completion times.
> "*Used to evaluate distributional performance and robustness."*

### time_to_first_delivery_cdf_grid  
CDF grid of first delivery times.

---

# Fairness Outcomes

### fairness_deliveries_gini_mean  
Mean Gini of deliveries.  
> *"Fairness / imbalance — Gini fairness on per‑agent deliveries."*

### fairness_deliveries_gini_std  
Variability in delivery fairness.

### fairness_rewards_gini_mean  
Mean Gini of rewards.

### fairness_rewards_gini_std  
Variability in reward fairness.

---

# Fairness Distributions per Agent

### fairness_deliveries_lorenz_x_agent  
Lorenz curve X for deliveries (N,).  
> *"Used to visualize inequality in agent workload."*

### fairness_rewards_lorenz_x_agent  
Lorenz curve X for rewards (N,).

---

# Credit Correlations

### credit_correlations_mean  
Mean action‑to‑outcome correlation.  
> *"Credit‑assignment proxies — correlation between actions and outcome."*

### credit_correlations_std  
Variability in credit signal.

### credit_agent_correlations  
Per‑agent credit correlation.

---

# Credit Shapley

### credit_shapley_mean  
Mean Shapley value; average contribution.  
> *"Shapley‑style proxies for credit assignment."*

### credit_shapley_std  
Variability in contribution.

### credit_shapley_loo_mean  
Mean leave‑one‑out Shapley.

### credit_shapley_loo_std  
Variability in LOO Shapley.

### credit_agent_shapley  
Per‑agent Shapley.

### credit_agent_shapley_loo  
Per‑agent LOO Shapley.

---

# Advantage Statistics

### advantage_mean  
Mean advantage; average learning signal.  
> *"Per‑agent advantage / TD error stats — mean/std/skew; important for A2C‑type."*

### advantage_std  
Variability in advantage.

### advantage_p10  
10th percentile advantage.

### advantage_p50  
Median advantage.

### advantage_p90  
90th percentile advantage.

### advantage_skew  
Skewness of advantage distribution.  
> *"Skewness indicates imbalance in TD error distribution."*

---

# Value Statistics

### q_value_magnitude_mean  
Mean Q‑value magnitude.  
> *"Indicates scale of critic outputs."*

### q_value_magnitude_std  
Variability in Q‑value magnitude.

### q_value_trend  
Q‑value trend over epochs.

### returns_agent_mean  
Per‑agent returns (N,).  
> *"Per‑agent total return — reveals imbalance between agents."*

### returns_agent_std  
Variability per agent.

---

# Loss Statistics

### loss_mean  
Mean total loss.  
> *"Total loss = actor_loss + λ * value_loss."*

### loss_std  
Variability in total loss.

### loss_percentile_stats_p10  
10th percentile loss.

### loss_percentile_stats_p50  
Median loss.

### loss_percentile_stats_p90  
90th percentile loss.

### loss_skew  
Skewness of loss distribution.  
> *"Indicates heavy‑tail instability in optimization."*

---

# Loss Dynamics per Epoch

### actor_loss_mean  
Mean actor loss per epoch.  
> *"Actor loss measures how much policy should change according to advantage."*

### actor_loss_std  
Variability in actor loss.

### actor_loss_trend  
Actor loss trend.

### value_loss_mean  
Mean value loss per epoch.  
> *"High value loss = critic inaccurate; causes shaky advantage."*

### value_loss_std  
Variability in value loss.

### value_loss_trend  
Value loss trend.

### ratio_mean  
Mean PPO ratio.

### ratio_std  
Variability in ratio.

### ratio_trend  
Ratio trend.

### cumulative_ratio_mean  
Mean cumulative ratio.

### cumulative_ratio_std  
Variability in cumulative ratio.

### UTD  
Update‑to‑Data ratio.  
> *"Number of critic updates vs actor updates — affects value_loss."*

---

# Entropy Global per Epoch

### entropy_mean  
Mean entropy; global exploration.  
> *"Entropy = measure of policy uncertainty… high entropy = exploratory."*

### entropy_std  
Variability in entropy.

### entropy_p10  
Low‑entropy tail.

### entropy_p50  
Median entropy.

### entropy_p90  
High‑entropy tail.

### entropy_trend  
Entropy trend over training.

---

# Entropy per Agent

### entropy_agent_mean  
Mean entropy per agent (N,).

### entropy_agent_std  
Variability per agent.

### entropy_agent_p10  
10th percentile entropy per agent.

### entropy_agent_p50  
Median entropy per agent.

### entropy_agent_p90  
90th percentile entropy per agent.

### entropy_agent_trend  
Entropy trend per agent.

---

# Action Distribution per Agent

### action_histogram_flat_agent  
Action histogram per agent.  
> *"Per‑agent action distribution / diversity index — coordinated or competing."*

### action_entropy_agent  
Action entropy per agent.

---

# Action Similarity Agents

### action_jsd  
Jensen‑Shannon divergence between agents’ action distributions.  
> *"Measures similarity/coordination between agents’ policies."*

---

# KL Global

### kl_over_P_mean  
Mean KL divergence.  
> *"KL between old/new policy — measure policy update; track large values."*

### kl_over_P_std  
Variability in KL.

### kl_p90_over_P  
90th percentile KL.

---

# KL per Parameter

### kl_per_P_mean_mean  
Mean of per‑parameter KL means.

### kl_per_P_std_std  
Std of per‑parameter KL stds.

### kl_per_P_p90_mean  
Mean of 90th percentile KL per parameter.

### kl_per_P_p90_max  
Max of 90th percentile KL per parameter.

### kl_per_P_p90_median  
Median of 90th percentile KL per parameter.

---

# Gradient Global

### grad_norm_mean  
Mean gradient norm.  
> *"Per‑agent gradient norms — detect agents receiving too large updates."*

### grad_norm_std  
Variability in gradient norm.

### grad_norm_p90  
90th percentile gradient norm.

---

# Gradient per Agent (Only if centralised critic)

### grad_norms_per_agent_mean  
Mean gradient norm per agent.

### grad_norms_per_agent_std  
Variability per agent.

### grad_norms_per_agent_p90_mean  
Mean of per‑agent 90th percentile gradient norms.

### grad_norms_per_agent_p90_max  
Max of per‑agent 90th percentile gradient norms.

### grad_norms_per_agent_p90_median  
Median of per‑agent 90th percentile gradient norms.

### grad_norms_agent_p90  
90th percentile gradient norm per agent.

### grad_norms_agent_mean  
Mean gradient norm per agent.

### grad_norms_agent_std  
Std gradient norm per agent.

---

# Rewards per Agent

### reward_agent_mean  
Mean reward per agent.

### reward_agent_std  
Variability per agent.

### reward_agent_p10  
10th percentile reward per agent.

### reward_agent_p50  
Median reward per agent.

### reward_agent_p90  
90th percentile reward per agent.

### reward_agent_trend  
Reward trend per agent.

---

# Reward Standardise

### reward_std_mean  
Mean of Welford‑standardised rewards.  
> *"Provides real-time rewards normalization without storing full history."*

### reward_std_std  
Variability in standardised rewards.

---

# Environment Metrics

### rware_mean  
Mean reward‑weighted metric per env.  
> *"Reward‑weighted metrics — weighted by team success or number of successful agents."*

### rware_std  
Variability in rware.

### rware_p10  
10th percentile rware.

### rware_p50  
Median rware.

### rware_p90  
90th percentile rware.

---

# System Performance

### FPS_mean  
Frames per second; throughput of environment steps.  
> *"FPS = frac(step_count / E_per_T_mean(step_time_t))."*

---

# System Timing

### step_time_mean  
Mean step time per env.

---

# Event Definitions

- **blocked_t** — agent attempted forward movement but lost contention  
- **noop_t** — agent chose no‑op  
- **pickup_t** — agent toggled load on shelf  
- **drop_t** — agent dropped shelf in non‑highway area  
- **distance_traveled_t** — forward movement count  

---

# Reward System

> *"Welford standardised rewards provide real-time normalization without storing full history."*

Reward components:

- Stage 1: +0.5 for previous‑step delivery  
- Stage 2: +0.5 for current‑step delivery  
- Global: +1 for all agents when one delivers  
- Individual: +1 for delivering agent  

---

# Loss Definitions

Actor loss:  
`-(log π(a|s) * A)`  

Value loss:  
`(R − V)^2`  

Total loss:  
`actor_loss + λ * value_loss`

---

# Temporal & Fraction Metrics

- **frac_mean** — mean proportion of events  
- **frac_std** — variability of event rate  
- **E_per_T_mean** — time‑averaged per‑env mean  
- **E_per_T_std** — temporal instability  

---