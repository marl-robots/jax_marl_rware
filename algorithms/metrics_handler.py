import math

import jax.numpy as jnp
import numpy as np

import algorithms.metrics_funcs as funcs


def process_raw(raw: dict, upd: int, batch_steps: int, actionDim: int):
    """
    T=env time_limit
    E= paralle envs
    N= n_agents
    P=PPO epoch=num_epochs is A2C P=1
    Episode = what happens between reset and done
    If you reset at the beginning of each U
    and each environment runs until the end of the episode
    Then each environment = episode
    When is this not true?
    If:
    a rollout is cut off in the middle of an episode
    or an episode spans multiple rollouts
    or there are episodes of varying lengths
    At each epoch (each U):
    You do env.reset() for each of the E environments

    One epoch=(U, T, E, N) |U|=k episodes=(T, E, N)
        episode_time shape:(U,)
        step_count_tensor_raw shape:(U, E)
        epoch_loss_tensor_raw shape:(U, P)
        step_time_tensor_raw shape:(U, T, E)
        block_tensor_raw shape:(U, T, E, N)
        rewards_tensor_raw shape:(U, T, E, N)
        actions_tensor_raw shape: (U, T, E, N)
        observation_tensor_raw shape: (U, T, E, N, O)
        deliveries_tensor_raw shape:(U, T, E, N)
        distance_traveled_tensor_raw shape:(U, T, E, N)
        epoch_old_logp_tensor_raw shape:(U, T, E, N)
        no need epoch_returns_tensor_raw shape:(U, T, E, N)
        idle_tensor_raw shape:(U, T, E, N)
        pickup_tensor_raw shape:(U, T, E, N)
        no need logits_tensor_raw shape:(U, T, 1, E*N, A)
        epoch_grads_tensor_raw (gradients dict need flat):(U,P,....)
        rewards_std_tensor_raw shape:(U, T, E, N)
        epoch_actor_loss_tensor_raw shape:(U, P, T, E)
        epoch_value_loss_tensor_raw shape:(U, P, T, E)
        epoch_advantage_tensor_raw shape:(U, P, T, E, N)
        epoch_entropy_tensor_raw shape:(U, P, T, E, N)
        epoch_logp_tensor_raw shape:(U, P, T, E, N)
        epoch_values_tensor_raw shape:(U, P, T, E, N)
        epoch_returns_tensor_raw shape:(U, P, T, E, N)
    epoch_grads_tensor(gradients dict)
    U=updates P=ppo_epoch H0,H1=hiddens layers A=action space O=observation space N=num agents
    HD=net out heads count
    RNN
    (actor->Dense_0->bias):val_shape(U, P, H0)
    (actor->Dense_0->kernel):val_shape(U, P, O, H0)
    (actor->Dense_1->bias):val_shape(U, P, A)
    (actor->Dense_1->kernel):val_shape(U, P, H0, A)
    (actor->ScannedGRU_0->GRUCell_0->hn->bias):val_shape(U, P, H0)
    (actor->ScannedGRU_0->GRUCell_0->hn->kernel):val_shape(U, P, H0, H1)
    (actor->ScannedGRU_0->GRUCell_0->hr->kernel):val_shape(U, P, H0, H1)
    (actor->ScannedGRU_0->GRUCell_0->hz->kernel):val_shape(U, P, H0, H1)
    (actor->ScannedGRU_0->GRUCell_0->in->bias):val_shape(U, P, H0)
    (actor->ScannedGRU_0->GRUCell_0->in->kernel):val_shape(U, P, H0, H1)
    (actor->ScannedGRU_0->GRUCell_0->ir->bias):val_shape(U, P, H0)
    (actor->ScannedGRU_0->GRUCell_0->ir->kernel):val_shape(U, P, H0, H1)
    (actor->ScannedGRU_0->GRUCell_0->iz->bias):val_shape(U, P, H0)
    (actor->ScannedGRU_0->GRUCell_0->iz->kernel):val_shape(U, P, H0, H1)
    (critic->Dense_0->bias):val_shape(U, P, H0)
    (critic->Dense_0->kernel):val_shape(U, P, O, H0) if centralised val_shape(U, P, N*O, H0)
    (critic->Dense_1->bias):val_shape(U, P, HD)
    (critic->Dense_1->kernel):val_shape(U, P, H0, HD)
    (critic->ScannedGRU_0->GRUCell_0->hn->bias):val_shape(U, P, H0)
    (critic->ScannedGRU_0->GRUCell_0->hn->kernel):val_shape(U, P, H0, H1)
    (critic->ScannedGRU_0->GRUCell_0->hr->kernel):val_shape(U, P, H0, H1)
    (critic->ScannedGRU_0->GRUCell_0->hz->kernel):val_shape(U, P, H0, H1)
    (critic->ScannedGRU_0->GRUCell_0->in->bias):val_shape(U, P, H0)
    (critic->ScannedGRU_0->GRUCell_0->in->kernel):val_shape(U, P, H0, H1)
    (critic->ScannedGRU_0->GRUCell_0->ir->bias):val_shape(U, P, H0)
    (critic->ScannedGRU_0->GRUCell_0->ir->kernel):val_shape(U, P, H0, H1)
    (critic->ScannedGRU_0->GRUCell_0->iz->bias):val_shape(U, P, H0)
    (critic->ScannedGRU_0->GRUCell_0->iz->kernel):val_shape(U, P, H0, H1)

    FCN
    (actor->Dense_0->bias):val_shape(U,P,H0)
    (actor->Dense_0->kernel):val_shape(U,P, O, H0)
    (actor->Dense_1->bias):val_shape(U,P,H0)
    (actor->Dense_1->kernel):val_shape(U,P,H0,H1)
    (actor->Dense_2->bias):val_shape(U,P, A)
    (actor->Dense_2->kernel):val_shape(U,P, H0, A)
    (critic->Dense_0->bias):val_shape(U,P,H0)
    (critic->Dense_0->kernel):val_shape(U,P, O, H0) if centralised val_shape(U, P, N*O, H0)
    (critic->Dense_1->bias):val_shape(U,P,H0)
    (critic->Dense_1->kernel):val_shape(U,P,H0,H1)
    (critic->Dense_2->bias):val_shape(U,P, HD)
    (critic->Dense_2->kernel):val_shape(U,P, H0, HD)
    """
    actions_tensor_raw = jnp.array(raw["actions_tensor_raw"])
    observation_tensor_raw = jnp.array(raw["observation_tensor_raw"])
    block_tensor_raw = jnp.array(raw["block_tensor_raw"])
    deliveries_tensor_raw = jnp.array(raw["deliveries_tensor_raw"])
    distance_traveled_tensor_raw = jnp.array(raw["distance_traveled_tensor_raw"])
    episode_time = jnp.array(raw["episode_time"])
    epoch_actor_loss_tensor_raw = jnp.array(raw["epoch_actor_loss_tensor_raw"])
    epoch_advantage_tensor_raw = jnp.array(raw["epoch_advantage_tensor_raw"])
    epoch_entropy_tensor_raw = jnp.array(raw["epoch_entropy_tensor_raw"])
    epoch_logp_tensor_raw = jnp.array(raw["epoch_logp_tensor_raw"])
    epoch_loss_tensor_raw = jnp.array(raw["epoch_loss_tensor_raw"])
    epoch_old_logp_tensor_raw = jnp.array(raw["epoch_old_logp_tensor_raw"])
    epoch_value_loss_tensor_raw = jnp.array(raw["epoch_value_loss_tensor_raw"])
    epoch_values_tensor_raw = jnp.array(raw["epoch_values_tensor_raw"])
    epoch_returns_tensor_raw = jnp.array(raw["epoch_returns_tensor_raw"])
    epoch_grads_dict_raw = dict(raw["epoch_grads_tensor_raw"])
    idle_tensor_raw = jnp.array(raw["idle_tensor_raw"])
    pickup_tensor_raw = jnp.array(raw["pickup_tensor_raw"])
    rewards_std_tensor_raw = jnp.array(raw["rewards_std_tensor_raw"])
    rewards_tensor_raw = jnp.array(raw["rewards_tensor_raw"])
    step_count_tensor_raw = jnp.array(raw["step_count_tensor_raw"])
    step_time_tensor_raw = jnp.array(raw["step_time_tensor_raw"])
    U = int(raw["epoch_values_tensor_raw"].shape[0])
    P = int(raw["epoch_values_tensor_raw"].shape[1])
    T = int(raw["epoch_values_tensor_raw"].shape[2])
    E = int(raw["epoch_values_tensor_raw"].shape[3])
    N = int(raw["epoch_values_tensor_raw"].shape[4])
    O = int(observation_tensor_raw.shape[-1])
    A = actionDim
    # Episode is split into thirds to show how behavior shifts within the 500-step episode.
    # [0-(t1-1),E,N,]
    # [t1-(t2-1),E,N,]
    # [t2-(T-1),E,N,]
    t1, t2 = T // 3, 2 * (T // 3)
    flatten_grads_dict = funcs.extract_flat_grads(epoch_grads_dict_raw)

    logger_metrics_list: list[dict[str, float]] = []
    for i in range(U):

        deliveries_fairness_metrics = funcs.fairness_metrics(deliveries_tensor_raw[i])
        rewards_fairness_metrics = funcs.fairness_metrics(rewards_tensor_raw[i])

        loss_stats = funcs.loss_stats(epoch_loss_tensor_raw[i])

        returns_stats = funcs.returns_stats(epoch_returns_tensor_raw[i])
        returns_stats_metrics = {
            **funcs.per_agent_dict(
                returns_stats["returns_per_agent_mean"],  # [N]
                "returns_per_agent_mean",
            ),
            **funcs.per_agent_dict(
                returns_stats["returns_per_agent_std"],  # [N]
                "returns_per_agent_std",
            ),
            "returns_mean": returns_stats["returns_mean"],
            "returns_std": returns_stats["returns_std"],
            "returns_percentile_stats_p10": returns_stats["returns_p10"],
            "returns_percentile_stats_p50": returns_stats["returns_p50"],
            "returns_percentile_stats_p90": returns_stats["returns_p90"],
            "returns_skew": returns_stats["returns_skew"],
        }

        kl_m = funcs.kl_divergence(
            epoch_old_logp_tensor_raw[i], epoch_logp_tensor_raw[i]
        )

        grads_u = {
            name: arr[i] for name, arr in flatten_grads_dict.items()
        }  # dict[param_name] with shape (P, ...)
        gn = funcs.gradient_norms(grads_u)
        gna = funcs.gradient_norms_per_agent(grads_u, N, O)

        per_agent_gna = funcs.per_agent_dict(gna)
        per_agent_gna_metric = {}
        for k, v in per_agent_gna.items():
            if v.ndim > 0:
                per_agent_gna_metric[k + "_mean"] = v.mean()
                per_agent_gna_metric[k + "_std"] = v.std()
            else:
                per_agent_gna_metric[k] = v

        rware_metric = funcs.rware_metric(
            rewards_tensor_raw[i], deliveries_tensor_raw[i]
        )

        credit_assignment_proxies_metric = funcs.credit_assignment_proxies(
            actions_tensor_raw[i], rewards_tensor_raw[i]
        )

        per_agent_credit_assignment_proxies_metric = funcs.per_agent_dict(
            credit_assignment_proxies_metric
        )

        action_distribution = funcs.action_distribution_metrics(
            actions_tensor_raw[i], A
        )

        action_distribution_metrics = {
            **funcs.flat_action_histogram(action_distribution["action_histogram"]),
            **funcs.per_agent_dict(
                action_distribution["action_entropy_per_agent"],
                "action_entropy_per_agent",
            ),
            **funcs.flat_action_jsd_metric(action_distribution["action_jsd_matrix"]),
        }

        advantage_stats = funcs.advantage_stats(epoch_advantage_tensor_raw[i])

        analyze_update = funcs.analyze_update(
            epoch_actor_loss_tensor_raw[i],
            epoch_value_loss_tensor_raw[i],
            epoch_entropy_tensor_raw[i],
            epoch_values_tensor_raw[i],
        )

        per_agent_entropy = funcs.per_agent_entropy(epoch_entropy_tensor_raw[i])
        per_agent_entropy_metrics = funcs.per_agent_dict(per_agent_entropy)

        entropy_Per_epoch_metrics = funcs.entropy_Per_epoch(epoch_entropy_tensor_raw[i])

        reward_per_agent = funcs.reward_per_agent(rewards_tensor_raw[i])
        reward_per_agent_metrics = funcs.per_agent_dict(reward_per_agent)
        # per-episode
        ret_ep = rewards_tensor_raw[i].sum(axis=0).sum(axis=-1)  # [E]
        step_time_ep = step_time_tensor_raw[i].sum(axis=0)  # [E]

        # percentiles
        ret_ep_p10, ret_ep_p50, ret_ep_p90 = funcs.percentile_stats(ret_ep)
        step_time_ep_p10, step_time_ep_p50, step_time_ep_p90 = funcs.percentile_stats(
            step_time_ep
        )
        add_metrics_mean, add_metrics_std = funcs.additional_metrics(
            deliveries_tensor_raw[i]
        )
        gradient_norms_metrics = {
            "grad_norms_per_agent_mean": gna["grad_norms_per_agent_mean"].mean(),
            "grad_norms_per_agent_std": gna["grad_norms_per_agent_std"].std(),
            "grad_norms_per_agent_p90_mean": gna["grad_norms_per_agent_p90"].mean(),
            "grad_norms_per_agent_p90_median": jnp.median(
                gna["grad_norms_per_agent_p90"]
            ),
            "grad_norms_per_agent_p90_max": gna["grad_norms_per_agent_p90"].max(),
            "grad_norm_mean": gn["grad_norm_mean"],
            "grad_norm_std": gn["grad_norm_std"],
            "grad_norm_p90": gn["grad_norm_p90"],
        }
        credit_assignment_proxies_metrics = {
            "credit_correlations_mean": credit_assignment_proxies_metric[
                "credit_per_agent_correlations"
            ].mean(),
            "credit_shapley_mean": credit_assignment_proxies_metric[
                "credit_per_agent_shapley"
            ].mean(),
            "credit_shapley_loo_mean": credit_assignment_proxies_metric[
                "credit_per_agent_shapley_loo"
            ].mean(),
            "credit_correlations_std": credit_assignment_proxies_metric[
                "credit_per_agent_correlations"
            ].std(),
            "credit_shapley_std": credit_assignment_proxies_metric[
                "credit_per_agent_shapley"
            ].std(),
            "credit_shapley_loo_std": credit_assignment_proxies_metric[
                "credit_per_agent_shapley_loo"
            ].std(),
        }
        advantage_stats_metrics = {
            "advantage_mean": advantage_stats["advantage_mean"],
            "advantage_std": advantage_stats["advantage_std"],
            "advantage_p10": advantage_stats["advantage_p10"],
            "advantage_p50": advantage_stats["advantage_p50"],
            "advantage_p90": advantage_stats["advantage_p90"],
            "advantage_skew": advantage_stats["advantage_skew"],
        }
        actor_loss_vs_critic_loss_metrics = {
            "actor_loss_mean": analyze_update["actor_loss_per_epoch"].mean(),  # (P,)
            "actor_loss_std": analyze_update["actor_loss_per_epoch"].std(),  # (P,)
            "actor_loss_trend": analyze_update["actor_loss_trend"],  # scalar
            "value_loss_mean": analyze_update["critic_loss_per_epoch"].mean(),  # (P,)
            "value_loss_std": analyze_update["critic_loss_per_epoch"].std(),  # (P,)
            "value_loss_trend": analyze_update["critic_loss_trend"],  # scalar
            "ratio_mean": analyze_update["per_epoch_ratio"].mean(),  # (P,)
            "ratio_std": analyze_update["per_epoch_ratio"].std(),  # (P,)
            "ratio_trend": analyze_update["per_epoch_ratio_trend"],  # scalar
            "cumulative_ratio_mean": analyze_update["cumulative_ratio"].mean(),  # (P,)
            "cumulative_ratio_std": analyze_update["cumulative_ratio"].std(),  # (P,)
            "UTD": analyze_update["UTD"],  # scalar
            "q_value_magnitude_mean": analyze_update[
                "q_value_magnitude"
            ].mean(),  # (P,)
            "q_value_magnitude_std": analyze_update["q_value_magnitude"].std(),  # (P,)
            "q_value_trend": analyze_update["q_value_trend"],  # scalar
        }

        episode_metrics_std = {
            "episode_returns_std": funcs.team_per_ep("std", rewards_tensor_raw[i]),
            "loss_std": loss_stats["loss_std"],
            "reward_std_std": rewards_std_tensor_raw[i].std(),
            "deliveries_std": funcs.team_per_ep("std", deliveries_tensor_raw[i]),
            "block_rate_std": funcs.frac("std", block_tensor_raw[i]),
            "idle_rate_std": funcs.frac("std", idle_tensor_raw[i]),
            "pickup_rate_std": funcs.frac("std", pickup_tensor_raw[i]),
            "deliveries_early_std": funcs.team_per_ep(
                "std", deliveries_tensor_raw[i][:t1]
            ),
            "deliveries_mid_std": funcs.team_per_ep(
                "std", deliveries_tensor_raw[i][t1:t2]
            ),
            "deliveries_late_std": funcs.team_per_ep(
                "std", deliveries_tensor_raw[i][t2:]
            ),
            "block_early_std": funcs.frac("std", block_tensor_raw[i][:t1]),
            "block_mid_std": funcs.frac("std", block_tensor_raw[i][t1:t2]),
            "block_late_std": funcs.frac("std", block_tensor_raw[i][t2:]),
            "distance_traveled_std": funcs.team_per_ep(
                "std", distance_traveled_tensor_raw[i]
            ),
            "step_time_std": funcs.E_per_T("std", step_time_tensor_raw[i]),
            "step_count_std": step_count_tensor_raw[i].std(),  # [i,E,]
            "success_std": funcs.num_successful("std", deliveries_tensor_raw[i]),
            "success_rate_std": funcs.team_success_rate(
                "std", deliveries_tensor_raw[i]
            ),
            "FPS_std": (step_count_tensor_raw[i] / (step_time_tensor_raw[i] + 1e-12))
            .mean(axis=0)
            .std(),
        }
        kl_metrics = {
            "kl_over_P_mean": kl_m["kl_mean_over_P"],
            "kl_over_P_std": kl_m["kl_std_over_P"],
            "kl_p90_over_P": kl_m["kl_p90_over_P"],
            "kl_per_P_mean_mean": kl_m["kl_per_P_mean"].mean(),
            "kl_per_P_std_std": kl_m["kl_per_P_std"].std(),
            "kl_per_P_p90_mean": kl_m["kl_per_P_p90"].mean(),
            "kl_per_P_p90_median": jnp.median(kl_m["kl_per_P_p90"]),
            "kl_per_P_p90_max": kl_m["kl_per_P_p90"].max(),
        }

        per_agent_distance_traveled_metric = {
            **funcs.per_agent_dict(
                distance_traveled_tensor_raw[i].sum(axis=(0, 1)), "distance_traveled"
            ),
        }

        episode_metrics_mean = {
            "episode_time": episode_time[i],
            "episode_returns_mean": funcs.team_per_ep("mean", rewards_tensor_raw[i]),
            "loss_mean": loss_stats["loss_mean"],
            "reward_std_mean": rewards_std_tensor_raw[i].mean(),
            "deliveries_mean": funcs.team_per_ep("mean", deliveries_tensor_raw[i]),
            "block_rate_mean": funcs.frac("mean", block_tensor_raw[i]),
            "idle_rate_mean": funcs.frac("mean", idle_tensor_raw[i]),
            "pickup_rate_mean": funcs.frac("mean", pickup_tensor_raw[i]),
            "deliveries_early_mean": funcs.team_per_ep(
                "mean", deliveries_tensor_raw[i][:t1]
            ),
            "deliveries_mid_mean": funcs.team_per_ep(
                "mean", deliveries_tensor_raw[i][t1:t2]
            ),
            "deliveries_late_mean": funcs.team_per_ep(
                "mean", deliveries_tensor_raw[i][t2:]
            ),
            "block_early_mean": funcs.frac("mean", block_tensor_raw[i][:t1]),
            "block_mid_mean": funcs.frac("mean", block_tensor_raw[i][t1:t2]),
            "block_late_mean": funcs.frac("mean", block_tensor_raw[i][t2:]),
            "distance_traveled_mean": funcs.team_per_ep(
                "mean", distance_traveled_tensor_raw[i]
            ),
            "step_time_mean": funcs.E_per_T("mean", step_time_tensor_raw[i]),
            "step_count_mean": step_count_tensor_raw[i].mean(),  # [E,]
            "success_mean": funcs.num_successful("mean", deliveries_tensor_raw[i]),
            "success_rate_mean": funcs.team_success_rate(
                "mean",
                deliveries_tensor_raw[i],
            ),
            "FPS_mean": (step_count_tensor_raw[i] / (step_time_tensor_raw[i] + 1e-12))
            .mean(axis=0)
            .mean(),
            "step_time_percentile_stats_p10": step_time_ep_p10,
            "step_time_percentile_stats_p50": step_time_ep_p50,
            "step_time_percentile_stats_p90": step_time_ep_p90,
            "loss_percentile_stats_p10": loss_stats["loss_p10"],
            "loss_percentile_stats_p50": loss_stats["loss_p50"],
            "loss_percentile_stats_p90": loss_stats["loss_p90"],
            "loss_skew": loss_stats["loss_skew"],
        }

        fairness_metrics = {
            "fairness_deliveries_gini_mean": deliveries_fairness_metrics[
                "fairness_gini_mean"
            ],
            "fairness_deliveries_gini_std": deliveries_fairness_metrics[
                "fairness_gini_std"
            ],
            **funcs.per_agent_dict(
                deliveries_fairness_metrics["fairness_lorenz_curve_x"],
                "fairness_deliveries_lorenz_x",
            ),
            **funcs.flat_fairness_metrics_lorenz_y(
                deliveries_fairness_metrics["fairness_lorenz_curve_y"],
                "fairness_deliveries_lorenz_y",
            ),
            "fairness_rewards_gini_mean": rewards_fairness_metrics[
                "fairness_gini_mean"
            ],
            "fairness_rewards_gini_std": rewards_fairness_metrics["fairness_gini_std"],
            **funcs.per_agent_dict(
                rewards_fairness_metrics["fairness_lorenz_curve_x"],
                "fairness_rewards_lorenz_x",
            ),
            **funcs.flat_fairness_metrics_lorenz_y(
                rewards_fairness_metrics["fairness_lorenz_curve_y"],
                "fairness_rewards_lorenz_y",
            ),
        }
        rware_metrics = {
            "rware_mean": rware_metric["rware_mean"],
            "rware_std": rware_metric["rware_std"],
            "rware_p10": rware_metric["rware_p10"],
            "rware_p50": rware_metric["rware_p50"],
            "rware_p90": rware_metric["rware_p90"],
        }
        done_count = upd + i + 1
        env_steps = done_count * batch_steps

        metrics = {
            "environment_steps": env_steps,
            "updates": done_count,
            **episode_metrics_mean,
            **episode_metrics_std,
            **add_metrics_mean,
            **add_metrics_std,
            # NEW
            **fairness_metrics,
            **kl_metrics,
            **gradient_norms_metrics,
            **per_agent_gna_metric,
            **credit_assignment_proxies_metrics,
            **per_agent_credit_assignment_proxies_metric,
            **entropy_Per_epoch_metrics,
            **per_agent_entropy_metrics,
            **rware_metrics,
            **advantage_stats_metrics,
            **actor_loss_vs_critic_loss_metrics,
            **reward_per_agent_metrics,
            **action_distribution_metrics,
            **per_agent_distance_traveled_metric,
            **returns_stats_metrics,
        }

        only_floats_metrics = {}
        for key, val in metrics.items():
            if isinstance(val, float):
                if math.isinf(val):
                    only_floats_metrics[key] = jnp.finfo(jnp.float32).max
                else:
                    only_floats_metrics[key] = jnp.float32(val)

            elif isinstance(val, jnp.ndarray):
                if val.size == 0:
                    only_floats_metrics[key] = jnp.float32(0.0)
                elif val.size == 1:
                    if jnp.isinf(val):
                        only_floats_metrics[key] = jnp.finfo(jnp.float32).max
                    else:
                        only_floats_metrics[key] = jnp.float32(val)

            else:
                only_floats_metrics[key] = val

        # all per-update metrics for this chunk, as np arrays of shape [k]
        m = {key: np.asarray(val) for key, val in only_floats_metrics.items()}
        m = {key: float(val) for key, val in m.items()}
        m["updates"] = int(m["updates"])
        m["environment_steps"] = int(m["environment_steps"])

        logger_metrics_list.append(m)
    return logger_metrics_list
