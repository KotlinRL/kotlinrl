package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.api.*
import kotlin.random.*

/**
 * Represents an agent that follows a defined policy for decision-making in a reinforcement learning setting.
 *
 * The PolicyAgent encapsulates a policy for selecting actions based on observed states. It also supports
 * observing transitions and trajectories, allowing for external logging or learning processes.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @property id A unique identifier for this agent.
 * @property policy The policy that determines the actions to be taken for a given state.
 * @property onTransition A callback for handling individual state-action transitions.
 * @property onTrajectory A callback for handling trajectories, which are sequences of transitions across episodes.
 */
class PolicyAgent<State, Action>(
    override val id: String,
    val policy: Policy<State, Action>,
    val onTransition: TransitionObserver<State, Action> = TransitionObserver { },
    val onTrajectory: TrajectoryObserver<State, Action> = TrajectoryObserver { _, _ -> }
) : Agent<State, Action> {

    /**
     * Determines the action to take based on the provided state using the agent's predefined policy.
     *
     * This method delegates the decision-making process to the associated policy,
     * which computes the appropriate action for the given state.
     *
     * @param state The current state of the environment for which an action is to be determined.
     * @return The action chosen by the policy based on the provided state.
     */
    override fun act(state: State): Action = policy(state)

    /**
     * Observes a single transition in the environment and delegates it to the onTransition callback.
     *
     * This method processes the transition, which includes the current state, chosen action,
     * resulting reward, next state, and additional metadata. It facilitates external handling
     * of the transition, such as logging, learning updates, or other custom behaviors defined
     * in the onTransition callback.
     *
     * @param transition The transition object containing the observed state, action, reward,
     * next state, and additional metadata.
     */
    override fun observe(transition: Transition<State, Action>) = onTransition(transition)

    /**
     * Observes a trajectory of transitions across an episode and delegates the processing to the onTrajectory callback.
     *
     * This method is typically used for recording or processing a sequence of transitions that represent an episode's
     * dynamics. The callback can handle tasks such as logging, policy updates, or statistical analysis.
     *
     * @param trajectory The list of transitions observed during the episode. Each transition contains a state,
     * action, reward, next state, termination status, and additional metadata.
     * @param episode The numeric identifier or index representing the episode.
     */
    override fun observe(trajectory: List<Transition<State, Action>>, episode: Int) = onTrajectory(trajectory, episode)
}
