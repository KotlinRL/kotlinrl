package io.github.kotlinrl.core.agent

/**
 * Represents an abstraction for agents interacting with environments in a reinforcement learning setup.
 *
 * An agent is an entity capable of observing its environment's state, deciding actions to take based on its policy or logic,
 * and learning or adapting from feedback (e.g., transitions or trajectories).
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
interface Agent<State, Action> {
    val id: String
    /**
     * Determines the action to take given the current state.
     *
     * This method encapsulates the agent's policy or logic for decision-making,
     * allowing it to interact with an external environment by selecting appropriate actions.
     *
     * @param state The current state of the environment.
     * @return The action chosen by the agent based on the provided state.
     */
    fun act(state: State): Action

    /**
     * Observes a single transition in the environment, typically for the purpose of learning or logging.
     *
     * This method allows the agent to receive feedback comprising the current state, action taken, resulting reward,
     * next state, and termination status of the environment, enabling reinforcement learning or other update mechanisms.
     *
     * @param transition The transition object containing the observed state, action, reward, next state, and additional metadata.
     */
    fun observe(transition: Transition<State, Action>)

    /**
     * Observes a trajectory of transitions across an episode.
     *
     * This method is used for processing or learning from a sequence of transitions that represent an episode.
     * It provides insights into the agent's performance over the entire episode and allows for updating policies or statistics.
     *
     * @param trajectory The sequence of transitions observed during the episode.
     * @param episode The identifier or number of the current episode.
     */
    fun observe(trajectory: Trajectory<State, Action>, episode: Int)
}