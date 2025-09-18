package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.*

/**
 * A wrapper that filters the observations from the environment to include only a specified subset
 * of keys. This class is useful when only certain parts of the observation data are relevant
 * for an agent's decision-making process, reducing the dimensionality and complexity of the
 * observation space.
 *
 * @param Action The type representing the actions that can be performed in the environment.
 * @param ActionSpace The type of the action space of the environment, specifying valid actions.
 * @param env The environment to be wrapped, providing the complete observations before filtering.
 * @param keys The set of keys to retain in the observations. Keys not included in this set
 * will be excluded from the filtered observations.
 */
class FilterObservation<Action, ActionSpace : Space<Action>>(
    env: Env<Map<String, Any>, Action, Dict, ActionSpace>,
    private val keys: Set<String>
) : Wrapper<
        Map<String, Any>, // Output obs is a Map<String, Any>
        Action,
        Dict,            // ObservationSpace is a Dict space
        ActionSpace,
        Map<String, Any>, // Wrapped obs
        Action,
        Dict,
        ActionSpace
        >(env) {

    /**
     * A filtered representation of the `observationSpace` where only the specified keys are retained.
     *
     * This property lazily initializes a `Dict` object based on the `observationSpace` of the underlying environment.
     * It filters the keys in the environment's `observationSpace` to include only those specified in the `keys` field.
     * The resulting `Dict` object reflects the subset of spaces corresponding to the retained keys.
     *
     * The filtering operation ensures that irrelevant or unwanted keys are excluded from the observation space,
     * allowing for a more focused and efficient representation of the environment.
     */
    override val observationSpace: Dict by lazy {
        // Only retain specified keys in observationSpace
        val filteredSpaces = env.observationSpace.spaces.filterKeys { it in keys }
        Dict(filteredSpaces)
    }

    /**
     * Specifies the action space for the environment, representing all possible actions
     * that an agent can perform. This property is overridden to provide a custom
     * implementation based on the environment's action space.
     *
     * The action space defines the structure and constraints of the permissible actions,
     * which is crucial for enabling the agent to interact meaningfully within the environment.
     */
    override val actionSpace: ActionSpace
        get() = env.actionSpace

    /**
     * Resets the environment to its initial state with filtered observations.
     *
     * This method reinitializes the environment, applies a filter to the observations based
     * on predefined keys, and starts a new episode. It allows for optional random seed
     * initialization and configuration through additional options.
     *
     * @param seed An optional random seed for reproducing deterministic behavior.
     *             If `null`, the environment will use its default random generator.
     * @param options An optional map of configuration options to customize the reset process.
     *                The specific keys and values depend on the environment implementation.
     * @return The initial state of the environment after reset, wrapped in an `InitialState` object,
     *         with filtered observations in the `state` and any associated metadata in the `info` map.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<Map<String, Any>> {
        val initial = env.reset(seed, options)
        return InitialState(
            state = filter(initial.state),
            info = initial.info
        )
    }

    /**
     * Executes a filtered step in the environment based on the provided action.
     *
     * This method applies an action to the environment, retrieves the resulting step data,
     * and then filters the state to retain only the specified keys before returning the result.
     *
     * @param action The action to be performed in the environment.
     * @return A `StepResult` containing the updated filtered state, reward, termination status,
     * truncation status, and additional information after the action is applied.
     */
    override fun step(action: Action): StepResult<Map<String, Any>> {
        val t = env.step(action)
        return t.copy(state = filter(t.state))
    }

    /**
     * Filters the input map to retain only the entries whose keys are part of the predefined key set.
     *
     * @param obs The input map containing key-value pairs to be filtered.
     * @return A filtered map containing only the key-value pairs where the keys exist in the predefined set.
     */
    private fun filter(obs: Map<String, Any>): Map<String, Any> =
        obs.filterKeys { it in keys }
}
