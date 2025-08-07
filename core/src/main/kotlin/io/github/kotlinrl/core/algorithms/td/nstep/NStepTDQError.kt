package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * Functional interface representing the computation of n-step temporal difference (TD) error
 * used in reinforcement learning algorithms. This abstraction is utilized to calculate the
 * error component necessary for updating Q-functions in various n-step TD methods,
 * including n-step SARSA and n-step Q-learning.
 *
 * The n-step TD error evaluates the discrepancy between the expected cumulative reward
 * over a trajectory of `n` steps (predicted by the Q-function) and the actual observed
 * rewards in the environment. The goal is to minimize this error during the learning process
 * to achieve an accurate estimation of the value function.
 *
 * @param State The type representing the states within the environment.
 * @param Action The type representing the actions taken by the agent.
 */
fun interface NStepTDQError<State, Action> {
    /**
     * Computes the n-step temporal difference (TD) error for a given trajectory, using a
     * Q-function and an optional policy to determine the expected return. This method
     * is typically used in reinforcement learning algorithms to update the Q-function
     * based on the discrepancy between expected and actual rewards over a trajectory segment.
     *
     * @param Q the Q-function that estimates the value of state-action pairs.
     * @param t the trajectory consisting of a sequence of state-action transitions.
     * @param policy the policy used to determine the probability distribution over actions,
     *               which may be null for off-policy updates.
     * @param tailAction an optional terminal action used for bootstrapping incomplete trajectories.
     *                   This can be null if no tail action is necessary.
     * @param gamma the discount factor applied to future rewards, controlling the trade-off
     *              between immediate and delayed rewards.
     * @return the computed n-step temporal difference error as a double.
     */
    operator fun invoke(
        Q: QFunction<State, Action>,
        t: Trajectory<State, Action>,
        policy: Policy<State, Action>?,
        tailAction: Action?,
        gamma: Double,
    ): Double
}
