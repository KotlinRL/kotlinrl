package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

/**
 * Abstract base class for algorithms implementing Dynamic Programming (DP) iterations
 * in reinforcement learning. It provides a foundation for constructing decision-making
 * policies through iterative planning methods.
 *
 * Dynamic Programming algorithms iteratively evaluate and improve policies to find
 * an optimal solution for Markov Decision Processes (MDPs). Subclasses of this class
 * are expected to implement the specific DP planning strategy.
 *
 * @param State the type representing states in the environment.
 * @param Action the type representing actions that can be performed in the environment.
 *
 * @see Planner
 */
abstract class DPIteration<State, Action> : Planner<State, Action> {

    /**
     * Invokes the dynamic programming iteration process to compute and return
     * a decision-making policy. The method implements the logic defined by the
     * `plan` function in the subclass and acts as an entry point for executing
     * policy planning in reinforcement learning algorithms.
     *
     * @return A `Policy` that maps states to actions, representing the result
     *         of the planning process.
     */
    override operator fun invoke(): Policy<State, Action> = plan()

    /**
     * Plans and computes a decision-making policy for a given Markov Decision Process (MDP).
     * The method relies on the specific dynamic programming iteration strategy
     * implemented in the subclass to generate a policy.
     *
     * @return A `Policy` that maps states to actions, representing the outcome
     *         of the dynamic programming iteration process.
     */
    abstract fun plan(): Policy<State, Action>
}
