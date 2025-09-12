package io.github.kotlinrl.core.algorithm

import io.github.kotlinrl.core.api.*
import kotlin.random.*

/**
 * Abstract base class for a learning algorithm implementing functionality for managing
 * and invoking policies. This class defines a common foundation for reinforcement learning
 * algorithms that operate using policies to decide actions based on states.
 *
 * @param State the type representing the state of the environment.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param initialPolicy the initial policy used by the algorithm for deciding actions based on states.
 * @param onPolicyUpdate a callback invoked every time the policy is updated, with the new policy as the argument.
 * @param rng the random number generator to be used throughout the algorithm's operations.
 */
abstract class BaseAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    protected val onPolicyUpdate: PolicyUpdate<State, Action> ,
    protected val rng: Random,
) : LearningAlgorithm<State, Action> {

    /**
     * Represents the current policy being used by the algorithm. This policy
     * determines the actions to be taken based on the given states within
     * the environment. The policy can be updated during the algorithm's operation,
     * reflecting changes in strategy or learning progress.
     *
     * The setter for this variable is protected, allowing it to be modified
     * only within subclasses or the defining class. Upon updating the policy,
     * the `onPolicyUpdate` callback is invoked to handle any logic required
     * to respond to the policy change.
     */
    var policy: Policy<State, Action> = initialPolicy
        protected set(value) {
            field = value
            onPolicyUpdate(value)
        }

    /**
     * Invokes the policy using the given state to determine the corresponding action.
     *
     * @param state the current state of the environment.
     * @return the action decided by the policy for the given state.
     */
    override fun invoke(state: State): Action = policy(state)
}