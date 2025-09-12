package io.github.kotlinrl.core.agent

/**
 * Represents a transition in a reinforcement learning environment, which encapsulates the interaction
 * between an agent and the environment within a single step.
 *
 * A transition contains information about the current state, the action taken,
 * the reward received, the resulting state, and termination or truncation status. It also includes
 * a flag to indicate whether the transition marks the end of an episode, as well as additional metadata.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @property state The state of the environment before the action was taken.
 * @property action The action chosen by the agent.
 * @property reward The reward received as a result of the action taken in the current state.
 * @property nextState The state of the environment resulting from the action.
 * @property terminated A flag indicating whether the episode has ended due to a terminal state.
 * @property truncated A flag indicating whether the episode was prematurely ended, such as due
 * to time limits or constraints.
 * @property done A computed flag that is true if the episode is either terminated or truncated.
 * @property info Additional metadata related to the transition.
 */
data class Transition<State, Action>(
    val state: State,
    val action: Action,
    val reward: Double,
    val nextState: State,
    val terminated: Boolean,
    val truncated: Boolean,
    val done: Boolean = terminated || truncated,
    val info: Map<String, Any?> = emptyMap()
)