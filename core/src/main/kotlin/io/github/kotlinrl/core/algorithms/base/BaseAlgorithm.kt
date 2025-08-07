package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * An abstract base class for reinforcement learning algorithms that defines
 * and manages a policy and Q-function. It provides basic functionalities shared
 * among different learning algorithms, such as invoking the policy to determine actions
 * and handling updates to the policy and Q-function.
 *
 * This class extends the `LearningAlgorithm` interface, defining a contract
 * for algorithms that can process transitions and trajectories for learning.
 *
 * State and Action are generic types that represent the environment's state and
 * action spaces, respectively.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions that can be performed.
 * @param initialPolicy the initial policy used by the algorithm to map states to actions.
 * @param onPolicyUpdate a callback invoked whenever the policy is updated.
 * @param onQFunctionUpdate a callback invoked whenever the Q-function is updated.
 */
abstract class BaseAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    private val onPolicyUpdate: PolicyUpdate<State, Action> = { },
    private val onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
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
     * Represents the current Q-function, which is used to evaluate the quality of
     * different state-action pairs within a reinforcement learning algorithm.
     *
     * The Q-function plays a critical role in determining the optimal policy by
     * encoding the expected future rewards for each possible action in a given state.
     * This variable is initialized using the Q-function associated with the
     * initial policy and can be updated during the learning process.
     *
     * When the Q-function is updated, the associated update listener (`onQFunctionUpdate`)
     * is automatically invoked to notify other components of the change, enabling
     * synchronization across the algorithm's components.
     */
    var Q = initialPolicy.Q
        protected set(value) {
            field = value
            onQFunctionUpdate(value)
        }

    /**
     * Invokes the policy using the given state to determine the corresponding action.
     *
     * @param state the current state of the environment.
     * @return the action decided by the policy for the given state.
     */
    override fun invoke(state: State): Action = policy(state)
}