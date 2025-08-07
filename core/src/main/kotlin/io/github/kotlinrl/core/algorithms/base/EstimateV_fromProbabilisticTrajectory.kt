package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * A functional interface that defines a strategy for estimating the value function (V)
 * of states in a reinforcement learning context, based on a probabilistic trajectory.
 *
 * This interface is primarily designed to process a probabilistic trajectory, which consists
 * of states, actions, and probabilities, and use it in conjunction with an existing
 * value function (V) to generate an updated value function. The probabilistic trajectory
 * encapsulates the dynamics of the environment in terms of likelihoods and transitions, enabling
 * more refined estimations of state values.
 *
 * Implementations of this interface should define a specific mechanism for incorporating
 * the probabilistic dynamics of the trajectory into the value function estimation. This
 * could involve techniques such as Monte Carlo evaluation, Temporal-Difference learning,
 * or other statistical approaches tailored for probabilistic trajectories.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions that can be performed in the environment.
 */
fun interface EstimateV_fromProbabilisticTrajectory<State, Action> {
    /**
     * Invokes the estimation strategy to generate an updated value function (V)
     * based on the initial value function and a provided probabilistic trajectory.
     *
     * @param V the initial value function, representing the current estimation
     *          of the value for each state in the environment.
     * @param trajectory the probabilistic trajectory containing states, actions,
     *                   and probabilities, representing the environment's dynamics.
     * @return a value function updated by incorporating information from the
     *         provided probabilistic trajectory.
     */
    operator fun invoke(
        V: ValueFunction<State>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): ValueFunction<State>
}
