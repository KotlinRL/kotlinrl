package io.github.kotlinrl.core.agent

/**
 * Represents a functional interface to observe state-action transitions in a reinforcement learning environment.
 *
 * The `TransitionObserver` is designed to handle and process transitions, enabling actions such as logging,
 * learning updates, or other custom behaviors. It acts as a callback for monitoring and managing how transitions
 * are handled when they occur.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
fun interface TransitionObserver<State, Action> {
    /**
     * Handles the invocation of a `Transition` representing a state-action transition in a reinforcement
     * learning environment.
     *
     * This method can be used to process, observe, or respond to a specific transition event,
     * allowing for tasks such as logging, learning updates, or custom behaviors.
     *
     * @param transition The transition object containing the current state, action taken, resulting reward,
     * next state, termination or truncation status, and any additional metadata.
     */
    operator fun invoke(transition: Transition<State, Action>)
}