package io.github.kotlinrl.core.algorithms.td.nstep

import kotlin.math.*

/**
 * Object containing implementations of n-step Temporal Difference (TD) error computation methods for
 * reinforcement learning. These methods, including variations of n-step Expected SARSA, SARSA, and Q-learning,
 * calculate the TD error used for updating Q-functions.
 *
 * The n-step TD methods use sequences of state-action transitions and potentially a policy
 * to compute the discrepancy between predicted Q-values and the observed trajectory rewards. Different
 * algorithms bootstrap future reward estimations in unique ways:
 * - Expected SARSA incorporates action probabilities from the policy.
 * - SARSA uses the actual action taken.
 * - Q-learning uses the highest-valued action (off-policy update).
 */
object NStepTDQErrors {
    /**
     * Computes the n-step Expected SARSA temporal difference (TD) error for a given trajectory.
     * This method calculates the TD error as the difference between the expected cumulative reward
     * over a trajectory of `n` steps and the Q-value of the starting state-action pair.
     * It incorporates bootstrapping if the trajectory is incomplete by estimating the value of the
     * final state using the given policy and Q-function.
     *
     * The method assumes that the policy provides probabilities of selecting actions and uses these
     * to compute the expected return in incomplete trajectories.
     *
     * @return the n-step TD error calculation as an instance of `NStepTDQError<State, Action>`,
     *         which evaluates the error for updating the Q-function.
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
     * Computes the n-step Temporal Difference (TD) error using the SARSA (State-Action-Reward-State-Action) algorithm.
     * This method calculates the TD error by summing up the discounted rewards over an n-step transition trajectory
     * and bootstrapping with the action-value of the final state-action pair in the trajectory, if applicable.
     * It is an on-policy method, meaning the update is performed using the action actually taken at each step.
     *
     * @return an instance of NStepTDQError that, when invoked, computes the n-step SARSA error for a given trajectory
     * based on the provided Q-function, trajectory, discount factor, and optional tail action.
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
     * Constructs the n-step Q-learning temporal difference (TD) error computation strategy.
     *
     * This method implements the off-policy n-step Q-learning algorithm by using the maximum
     * Q-value over actions at the n-th step for bootstrapping. It calculates the TD error by
     * comparing the observed cumulative reward and the predicted value from the Q-function.
     *
     * The computation considers:
     * - The cumulative discounted reward over the trajectory of n steps.
     * - The maximum estimated Q-value for the state at the end of the trajectory for bootstrapping, if the final state is non-terminal.
     * - The TD error as the difference between this target value and the Q-value of the initial state-action pair.
     *
     * @return A strategy function of type `NStepTDQError` that calculates the TD error for the n-step Q-learning algorithm.
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