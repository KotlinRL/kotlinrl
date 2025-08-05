package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Represents an abstract base class for reinforcement learning algorithms. This class defines a contract
 * for algorithms that learn and adapt policies based on observed states, actions, and transitions within
 * a given environment.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param initialPolicy the initial policy to be used by the algorithm.
 * @param onPolicyUpdate a callback function invoked upon policy updates.
 */
abstract class LearningAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    protected val onPolicyUpdate: PolicyUpdate<State, Action> = { }
) {

    var policy: Policy<State, Action> = initialPolicy
        protected set(value) {
            field = value
            onPolicyUpdate(value)
        }

    /**
     * Invokes the underlying policy with the provided state to determine the corresponding action.
     *
     * @param state the current state of the environment.
     * @return the action determined by the policy for the given state.
     */
    operator fun invoke(state: State): Action = policy(state)

    /**
     * Updates the algorithm's state or model based on the provided transition.
     *
     * This method processes the observed transition, which encapsulates a state,
     * an action taken in that state, a reward received for that action, and the
     * resulting new state. Implementations of this method may involve updating
     * internal representations, such as policies or value functions, to reflect the
     * information contained within the transition.
     *
     * @param transition the observed transition consisting of the initial state,
     * the action taken, the obtained reward, and the resulting state.
     */
    fun update(transition: Transition<State, Action>) = observe(transition)

    /**
     * Updates the underlying model or policy based on the observed trajectory and episode information.
     *
     * This method delegates the update process to the `observe` function, allowing implementations
     * to handle trajectories and episode details appropriately. The update typically involves
     * processing reward signals, adjusting policies, or modifying internal representations.
     *
     * @param trajectory The sequence of transitions observed during an episode, containing
     * states, actions, and rewards.
     * @param episode The index or identifier of the episode associated with the trajectory.
     */
    fun update(trajectory: Trajectory<State, Action>, episode: Int) = observe(trajectory, episode)

    /**
     * Observes and processes a single transition within the learning algorithm.
     *
     * This method is designed to handle an individual transition, which includes the
     * current state, an action performed, the reward received, and the resulting new state.
     * Subclasses can override this function to implement specific handling or updating logic
     * based on the observed transition.
     *
     * @param transition the transition containing the initial state, action taken,
     * reward received, and the resulting state after the action.
     */
    protected open fun observe(transition: Transition<State, Action>) {}

    /**
     * Observes and processes a trajectory within the learning algorithm.
     *
     * This method is designed to process a sequence of transitions associated with a specific episode.
     * Subclasses can override this function to implement behavior that updates the state or model
     * of the algorithm based on the observed trajectory and episode details.
     *
     * @param trajectory The sequence of transitions observed during an episode. Each transition encapsulates
     * states, actions, and rewards encountered over the episode.
     * @param episode The index or identifier of the episode associated with the provided trajectory.
     */
    protected open fun observe(trajectory: Trajectory<State, Action>, episode: Int) {}
}