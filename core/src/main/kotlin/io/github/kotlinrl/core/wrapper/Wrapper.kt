package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import kotlin.random.*

/**
 * Represents an abstract wrapper around an environment, enabling transformations
 * or extensions to be applied to the underlying environment. The wrapper provides
 * a way to modify or augment the behavior, spaces, or representations of the
 * underlying environment by overriding specific methods or spaces.
 *
 * @param State The type representing the abstracted state in the wrapped environment.
 * @param Action The type representing the abstracted action in the wrapped environment.
 * @param ObservationSpace The type of space specifying the abstract observation space for the state.
 * @param ActionSpace The type of space specifying the abstract allowable actions.
 * @param WrappedState The type representing the state in the original wrapped environment.
 * @param WrappedAction The type representing the actions in the original wrapped environment.
 * @param WrappedObservationSpace The type of space specifying the original observation space.
 * @param WrappedActionSpace The type of space specifying the original allowable actions.
 * @param env The underlying environment to be wrapped. All non-abstract methods
 *            of this class delegate functionality to the corresponding methods
 *            of this wrapped environment.
 */
abstract class Wrapper<
        State,
        Action,
        ObservationSpace : Space<State>,
        ActionSpace : Space<Action>,
        WrappedState,
        WrappedAction,
        WrappedObservationSpace : Space<WrappedState>,
        WrappedActionSpace : Space<WrappedAction>
        >(
    protected val env: Env<WrappedState, WrappedAction, WrappedObservationSpace, WrappedActionSpace>
) : Env<State, Action, ObservationSpace, ActionSpace> {

    /**
     * Executes a single step in the environment using the provided action and returns the result.
     *
     * @param action The action to be executed within the environment. This determines
     *               the interaction with the current state of the environment.
     * @return The result of the step, encapsulating the new state of the environment,
     *         the reward obtained, whether the episode has terminated or been truncated,
     *         and any additional metadata.
     */
    abstract override fun step(action: Action): StepResult<State>

    /**
     * Resets the environment to its initial state and returns the initial state of the environment,
     * optionally applying a specific random seed and additional configuration options.
     *
     * @param seed The seed for the environment's random number generator, or null to use a random seed.
     *             This parameter controls the randomness of the environment's behavior and ensures
     *             reproducibility.
     * @param options A map of additional options or configurations for resetting the environment.
     *                These options can include settings like custom initial states or environment-specific
     *                parameters.
     * @return The initial state of the environment encapsulated in an `InitialState<State>` object,
     *         which contains both the state and additional metadata describing the reset state.
     */
    abstract override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State>

    /**
     * Renders the current state of the environment.
     *
     * This method provides a representation of the environment's current state by delegating
     * to the `env` object. The result can be an empty rendering or a rendered frame with
     * specific data, dependent on the environment's implementation.
     *
     * @return A `Rendering` instance representing the current state of the environment.
     */
    override fun render(): Rendering = env.render()

    /**
     * Closes the environment and releases any allocated resources.
     *
     * This method is an override that delegates the closing operation to the encapsulated `env` object.
     * It ensures that any resources held by the environment, such as file handles, network connections,
     * or memory allocations, are properly released.
     *
     * Once this method is invoked, the environment should no longer be used.
     */
    override fun close() = env.close()

    /**
     * Provides metadata for the environment, represented as a map of key-value pairs.
     *
     * The keys in the map are strings representing metadata attributes or identifiers,
     * and the values are of type `Any?`, allowing flexibility in the type of data stored.
     *
     * This property is an override, delegating to the `metadata` property of the encapsulated `env` object.
     * It contains environment-specific details or additional information about the current instance of the environment.
     */
    override val metadata: Map<String, Any?>
        get() = env.metadata

    /**
     * Represents the space that defines the structure, shape, and bounds of possible observations
     * in the environment. This is an abstract property that must be overridden to specify the
     * observation space corresponding to the specific implementation of the environment.
     *
     * The observation space essentially describes the possible observations the environment can
     * produce during interactions, providing constraints such as dimensionality and value ranges.
     */
    abstract override val observationSpace: ObservationSpace

    /**
     * Represents the action space of the environment.
     *
     * This property defines the set of all possible actions that can be performed in the environment.
     * It typically outlines the structure, shape, and type of the actions that are valid, enabling the
     * user or agent interacting with the environment to understand the permissible operations.
     *
     * The implementation of this property is environment-specific and abstracts the underlying
     * configurations or constraints of the action domain.
     */
    abstract override val actionSpace: ActionSpace

    /**
     * Provides access to the random number generator used by the environment.
     *
     * This property delegates to the `random` instance within the encapsulated `env` object.
     * It is utilized to generate random numbers, ensuring consistent and reproducible
     * behavior when required, such as during environment resets or step executions.
     */
    override val random: Random
        get() = env.random
}
