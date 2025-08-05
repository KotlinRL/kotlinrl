package io.github.kotlinrl.core.algorithms.td.nstep

import kotlin.math.*

/**
 * Provides implementations of n-step temporal difference (TD) errors used in reinforcement
 * learning algorithms such as Expected SARSA, SARSA, and Q-learning. These TD error functions
 * calculate the discrepancy between the current Q-values and the target Q-values derived from
 * n-step transitions in an agent's trajectory.
 *
 * The implementations support on-policy and off-policy learning, leveraging the action-value
 * function (Q-function), sampled trajectories, and optional policy information or tail actions.
 * They account for both immediate rewards and expected future rewards by incorporating the
 * discount factor.
 */
object NStepTDQErrors {
    /**
     * Computes the n-step Expected SARSA Temporal Difference (TD) error for a Q-function.
     *
     * The function calculates the TD error using trajectories of transitions and incorporates the expected
     * value of future rewards based on the given policy. Expected SARSA is used to update action-value
     * estimates by combining stochastic policies with discounted future rewards.
     *
     * The TD error is calculated as:
     *  - The sum of discounted rewards up to the n-th step
     *  - Plus the bootstrapped expected Q-value at the n-th step (if the episode is not done)
     *  - Minus the current Q-value of the initial state and action in the trajectory.
     *
     * @param State The type representing the environment's state space.
     * @param Action The type representing the actions available in the environment.
     * @return An implementation of NStepTDQError<State, Action> that computes the n-step Expected SARSA TD error.
     */
    @Suppress("DuplicatedCode")
    fun <State, Action> nStepExpectedSARSA(): NStepTDQError<State, Action> =
        NStepTDQError { Q, traj, policy, _, gamma ->
            require(policy != null) { "Policy required for n-step Expected SARSA." }
            val (s0, a0) = traj.first().state to traj.first().action

            var G = 0.0
            traj.forEachIndexed { i, t -> G += gamma.pow(i) * t.reward }

            val last = traj.last()
            if (!last.done) {
                val sT = last.nextState                  // <-- bootstrap at S_{t+n}
                val exp = policy.probabilities(sT).entries
                    .sumOf { (a, p) -> p * Q[sT, a] }
                G += gamma.pow(traj.size) * exp
            }

            G - Q[s0, a0]
        }

    /**
     * Implements the n-step SARSA algorithm to compute the temporal difference (TD) error
     * for state-action pairs in reinforcement learning.
     *
     * This function calculates the n-step TD error using an on-policy approach, incorporating
     * the actual action taken at time t+n to bootstrap the future value estimation. The
     * computation builds on the rewards accumulated over the trajectory and the Q-value of
     * the subsequent state-action pair if the trajectory has not ended.
     *
     * The algorithm computes:
     * - G (Return) as the sum of discounted rewards over the trajectory.
     * - Optionally, includes the discounted Q-value for the tail action if the trajectory is
     *   incomplete.
     * - Subtracts the current Q-value estimate from the calculated return G to produce the TD error.
     *
     * The function is utilized in policy evaluation and improvement procedures within
     * reinforcement learning systems.
     *
     * @param State The type representing the state of the environment.
     * @param Action The type representing the actions possible within the environment.
     * @return An `NStepTDQError<State, Action>` instance encapsulating the logic for
     *         calculating the n-step temporal difference error for SARSA updates.
     */
    // n-step SARSA (on-policy): bootstrap with the action actually taken at time t+n
    @Suppress("DuplicatedCode")
    fun <State, Action> nStepSARSA(): NStepTDQError<State, Action> =
        NStepTDQError { Q, traj, _, tailAction, gamma ->
            val (s0, a0) = traj.first().state to traj.first().action

            val n = traj.size
            var G = 0.0
            for (i in 0 until n) {
                G += gamma.pow(i) * traj[i].reward
            }

            val last = traj.last()
            if (!last.done && tailAction != null) {
                val sT = last.nextState
                G += gamma.pow(n) * Q[sT, tailAction]
            }

            G - Q[s0, a0]
        }

    /**
     * Implements the n-step Q-learning algorithm for temporal difference (TD) reinforcement learning.
     *
     * This method computes the n-step TD error for a Q-function by bootstrapping with the maximum
     * Q-value of the state reached after n steps, incorporating rewards from the trajectory and
     * applying a discount factor to future rewards. It is an off-policy approach that updates
     * the Q-function by comparing the estimated value with the observed value derived from the
     * trajectory and the maximum future Q-value.
     *
     * The returned function (of type `NStepTDQError`) defines the logic for error calculation,
     * given a Q-function, a trajectory of transitions, and relevant parameters like the discount factor.
     * It balances exploration and exploitation by using the maximum action-value for future states.
     *
     * @return A function that calculates the n-step TD error for Q-learning based on a trajectory
     *         of state-action transitions and the associated rewards.
     */
    // n-step Q-learning (off-policy): bootstrap with max_a Q(S_{t+n}, a)
    @Suppress("DuplicatedCode")
    fun <State, Action> nStepQLearning(): NStepTDQError<State, Action> =
        NStepTDQError { Q, traj, _, _, gamma ->
            val (s0, a0) = traj.first().state to traj.first().action

            var G = 0.0
            traj.forEachIndexed { i, t -> G += gamma.pow(i) * t.reward }

            val last = traj.last()
            if (!last.done) {
                val sT = last.nextState
                G += gamma.pow(traj.size) * Q.maxValue(sT)
            }

            G - Q[s0, a0]
        }
}