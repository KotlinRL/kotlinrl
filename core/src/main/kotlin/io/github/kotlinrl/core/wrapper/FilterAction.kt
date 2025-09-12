package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

/**
 * A wrapper class for filtering actions in a reinforcement learning environment.
 *
 * This class restricts the actions visible to the agent to a specified set of keys while maintaining
 * the full set of actions required by the underlying environment. It allows the agent to focus
 * on a subset of the action space and automatically fills in the remaining keys with default values
 * or samples from the action space as required.
 *
 * @param State The type of the state used in the environment.
 * @param ObservationSpace The type of the observation space, which must extend `Space<State>`.
 * @param env The underlying environment being wrapped. The environment specifies the full action
 *            and observation spaces and defines the step and reset logic.
 * @param keys A set of keys specifying the subset of actions exposed to the agent.
 * @param default An optional map of default values for any keys not included in the agent's actions.
 *                If not provided, values will be sampled from the action space as needed.
 */
class FilterAction<State, ObservationSpace : Space<State>>(
    env: Env<State, Map<String, Any>, ObservationSpace, Dict>,
    private val keys: Set<String>,
    private val default: Map<String, Any>? = null
) : Wrapper<
        State,
        Map<String, Any>, // Agent provides only selected keys
        ObservationSpace,
        Dict,
        State,
        Map<String, Any>, // Underlying env expects full Dict action
        ObservationSpace,
        Dict
        >(env) {

    /**
     * Represents the action space filtered for the selected keys. This property is lazily initialized
     * to expose only a subset of the environment's action space to the `policyAgent`, defined by the keys
     * specified in the `keys` property of the containing class.
     *
     * The filtered action space utilizes the `Dict` class, ensuring only the permissible keys and their
     * associated subspaces are included.
     *
     * @see Dict
     */
    override val actionSpace: Dict by lazy {
        // Only expose the filtered part to the policyAgent
        val filteredSpaces = env.actionSpace.spaces.filterKeys { it in keys }
        Dict(filteredSpaces)
    }

    /**
     * The observation space of the environment.
     *
     * This property retrieves the observation space of the wrapped environment. The observation space
     * defines the structure and bounds of the observations that can be obtained from the environment.
     */
    override val observationSpace: ObservationSpace
        get() = env.observationSpace

    /**
     * Resets the environment to its initial state, providing a starting point for a new sequence of actions.
     * It can optionally utilize a random seed and configuration options to customize the reset process.
     *
     * @param seed An optional random seed used for deterministic initialization of the environment.
     *             If `null`, the environment will use its default random generator.
     * @param options An optional map of configuration options influencing the reset behavior of the environment.
     *                The expected keys and values depend on the specific environment implementation.
     * @return An `InitialState` object that encapsulates the starting state of the environment along with additional metadata.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> =
        env.reset(seed, options)

    /**
     * Executes a step in the environment using the provided action.
     * If the action is missing required keys, they are filled in with default values
     * or samples from the action space.
     *
     * @param action A map representing the action to be performed in the environment.
     *               The map can contain partial actions, and missing keys will be filled automatically.
     * @return A `StepResult` object containing the resulting state of the environment,
     *         the reward obtained, termination status, truncation status, and additional metadata.
     */
    override fun step(action: Map<String, Any>): StepResult<State> {
        // Fill missing keys using default (or actionSpace.sample() if not provided)
        val fullAction = buildMap<String, Any> {
            // Agent-controlled keys
            putAll(action)
            // Fill in others
            val allSpaces = env.actionSpace.spaces
            val fillDefault = default ?: env.actionSpace.sample()
            for (k in allSpaces.keys) {
                if (k !in action) {
                    put(k, fillDefault[k] ?: error("No default value for action key $k"))
                }
            }
        }
        return env.step(fullAction)
    }
}
