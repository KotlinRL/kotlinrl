package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * Represents a function for calculating the n-step temporal difference (TD) error
 * for Q-functions in reinforcement learning.
 *
 * This interface defines the core logic to compute the TD error for a given Q-function,
 * trajectory, policy, and other relevant parameters. The n-step TD error calculation
 * is an essential part of algorithms such as n-step SARSA and n-step Q-learning, which
 * use it to update action-value estimates and improve policies over time.
 *
 * The function considers the trajectory of states, actions, and rewards, and optionally
 * the tail action for incomplete trajectories, in addition to incorporating a discount
 * factor for future rewards.
 *
 * @param State The type representing the environment's state space.
 * @param Action The type representing the actions available in the environment.
 */
fun interface NStepTDQError<State, Action> {
    /**
     * Computes the n-step temporal difference error for a given Q-function
     * based on the provided trajectory, policy, tail action, and discount factor.
     *
     * This operator function calculates the error by considering the rewards
     * collected over a series of steps and the expected future rewards given
     * by the Q-function. It supports reinforcement learning algorithms such as
     * n-step SARSA and n-step Q-learning.
     *
     * @param Q The Q-function that estimates the action-value for state-action pairs.
     * @param t The trajectory containing a series of states, actions, and rewards
     *          that occurred during an agent's interaction with the environment.
     * @param policy The policy that guides the agent's behavior. Used to calculate
     *               the expected Q-values for non-deterministic policies.
     *               If null, a default behavior may be applied.
     * @param tailAction The action taken after the last step in the trajectory.
     *                   This is used for incomplete trajectories to handle edge cases.
     * @param gamma The discount factor applied to future rewards, influencing
     *              how much weight is given to immediate versus future rewards.
     *              The value must be in the range [0, 1].
     * @return The computed n-step temporal difference error as a Double.
     */
    operator fun invoke(
        Q: QFunction<State, Action>,
        t: Trajectory<State, Action>,
        policy: QFunctionPolicy<State, Action>?,
        tailAction: Action?,
        gamma: Double,
    ): Double
}
