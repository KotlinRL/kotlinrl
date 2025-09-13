package io.github.kotlinrl.core

/**
 * A type alias for the `ClipAction` class, simplifying its usage and reference in the codebase.
 *
 * Represents a wrapper for modifying actions within an environment by constraining them to lie
 * within the predefined bounds of the environment's action space. The alias preserves the generic
 * signature of the class, supporting a broad range of environments with varying state types,
 * numeric types, dimensions, and observation space configurations.
 *
 * @param State The type representing the state of the environment.
 * @param Num The numeric type used for the actions and observations within the environment.
 * @param D The dimension type for the environment's action and observation spaces.
 * @param ObservationSpace The type representing the observation space of the environment.
 */
typealias ClipAction<State, Num, D, ObservationSpace> = io.github.kotlinrl.core.wrapper.ClipAction<State, Num, D, ObservationSpace>
/**
 * A type alias for `io.github.kotlinrl.core.wrapper.FilterAction`, simplifying the reference
 * to the wrapper that filters visible actions in a reinforcement learning environment.
 *
 * `FilterAction` restricts the agent to a subset of the environment's action space, allowing
 * a more focused interaction by providing only the selected keys while still maintaining
 * the underlying environment's complete action set.
 *
 * @param State The type representing the environment's state structure.
 * @param ObservationSpace The type representing the observation space of the environment.
 */
typealias FilterAction<State, ObservationSpace> = io.github.kotlinrl.core.wrapper.FilterAction<State, ObservationSpace>
/**
 * Typealias for the `FilterObservation` wrapper in the KotlinRL library.
 *
 * This alias represents a wrapper that filters observations from an environment by retaining
 * only a specified subset of keys. It is useful for scenarios where the entire observation data
 * is not relevant to the agent's decision-making process, allowing for reduced dimensionality
 * and complexity of the observation space.
 *
 * @param Action The type representing actions that can be performed in the environment.
 * @param ActionSpace The type of the environment's action space, specifying valid actions.
 */
typealias FilterObservation<Action, ActionSpace> = io.github.kotlinrl.core.wrapper.FilterObservation<Action, ActionSpace>
/**
 * A type alias for the `FlattenObservation` class, which wraps an existing environment to provide
 * a flattened one-dimensional array (NDArray) representation of its observations.
 *
 * This type alias simplifies references to the `FlattenObservation` class by reducing the verbosity of its
 * generic parameter specification.
 *
 * @param Num The numeric type for the elements of the flattened observation (e.g., Double, Float, Int, Long).
 * @param WrappedState The original, untransformed state type of the underlying wrapped environment.
 * @param Action The type of actions supported by the environment.
 * @param WrappedObservationSpace The original observation space type of the wrapped environment.
 * @param ActionSpace The action space type defining the allowable actions in the environment.
 */
typealias FlattenObservation<Num, WrappedState, Action, WrappedObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.FlattenObservation<Num, WrappedState, Action, WrappedObservationSpace, ActionSpace>
/**
 * Typealias for `io.github.kotlinrl.core.wrapper.Monitor`.
 *
 * Represents a wrapper for environments to monitor and log episode statistics, including rewards and lengths,
 * during reinforcement learning tasks. This alias allows for simpler references to the Monitor class.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing actions that can be performed within the environment.
 * @param ObservationSpace The type of space defining the structure of valid observations in the environment.
 * @param ActionSpace The type of space defining the structure of valid actions in the environment.
 */
typealias Monitor<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.Monitor<State, Action, ObservationSpace, ActionSpace>
/**
 * Typealias for `io.github.kotlinrl.core.wrapper.NormalizeReward`.
 *
 * Represents an environment wrapper that normalizes rewards based on the running mean and standard deviation
 * of observed rewards. This normalization helps stabilize reward distribution, which is beneficial for
 * training reinforcement learning algorithms. The normalization process avoids division by zero by including an
 * epsilon parameter as a safeguard.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing the actions available in the environment.
 * @param ObservationSpace The type of space describing the observation space for the environment's state.
 * @param ActionSpace The type of space describing the possible actions in the environment.
 */
typealias NormalizeReward<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.NormalizeReward<State, Action, ObservationSpace, ActionSpace>
/**
 * A type alias for `io.github.kotlinrl.core.wrapper.NormalizeState`, which represents a wrapper
 * that normalizes state observations in an environment. It is particularly useful when states
 * need to be scaled or standardized for consistent processing by agents.
 *
 * By using this alias, the full qualifier can be avoided while retaining a clear reference to
 * the `NormalizeState` class.
 *
 * @param Num The numeric type of the observations, extending `Number`.
 * @param D The dimensionality of the `NDArray`.
 * @param Action The type of actions executable in the environment.
 * @param ObservationSpace The structure or constraints defining the valid observation space.
 * @param ActionSpace The structure or constraints defining the valid actions in the environment.
 */
typealias NormalizeState<Num, D, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.NormalizeState<Num, D, Action, ObservationSpace, ActionSpace>
/**
 * A type alias for `io.github.kotlinrl.core.wrapper.OrderEnforcing`.
 *
 * Represents a wrapper for enforcing the correct order of operations within an environment.
 * This utility ensures that `reset` is called before any `step` operations, adding a layer of
 * control to prevent invalid operation sequences. It throws an exception if `step` is called
 * before resetting the environment or after an episode ends (terminated or truncated).
 *
 * @param State The type representing the environment's state.
 * @param Action The type representing actions that can be performed in the environment.
 * @param ObservationSpace The structured space or constraints defining observations.
 * @param ActionSpace The structured space or constraints defining valid actions.
 */
typealias OrderEnforcing<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.OrderEnforcing<State, Action, ObservationSpace, ActionSpace>
/**
 * A type alias for `io.github.kotlinrl.core.wrapper.RecordEpisodeStatistics`.
 *
 * This alias simplifies the reference to the `RecordEpisodeStatistics` wrapper, which tracks
 * cumulative episode statistics such as total reward and episode length. The statistics are
 * included in the `info` map upon completion of an episode. It is utilized in reinforcement
 * learning environments for monitoring and analysis of episode-level performance.
 *
 * @param State The type representing the environment's state.
 * @param Action The type representing the actions that can be performed in the environment.
 * @param ObservationSpace The type representing the observation space structure of the environment.
 * @param ActionSpace The type representing the action space structure of the environment.
 */
typealias RecordEpisodeStatistics<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.RecordEpisodeStatistics<State, Action, ObservationSpace, ActionSpace>
/**
 * Typealias for the `RecordVideo` class in the `io.github.kotlinrl.core.wrapper` package.
 *
 * `RecordVideo` is a wrapper for environments that enables video recording of episodes. This class
 * captures frames rendered by the environment and saves them as video files. It is useful for
 * visualizing agent behavior over the course of multiple episodes.
 *
 * @param State The type representing the environment's state.
 * @param Action The type representing actions performed in the environment.
 * @param ObservationSpace The space type representing the observation space of the environment.
 * @param ActionSpace The space type representing the action space of the environment.
 */
typealias RecordVideo<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.RecordVideo<State, Action, ObservationSpace, ActionSpace>
/**
 * Type alias for the `RescaleAction` wrapper.
 *
 * Represents an environment wrapper that rescales the action space of a given environment.
 * The rescaling adjusts the action range provided by an agent to align with the acceptable
 * action range required by the underlying environment. This is particularly useful in cases
 * where agents operate within normalized action ranges but need to interact with environments
 * that have different action space bounds.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing actions performed within the environment.
 * @param ObservationSpace The type extending `Space<State>`, representing the observation space of the environment.
 * @param ActionSpace The type extending `Space<Action>`, representing the action space of the environment.
 */
typealias RescaleAction<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.RescaleAction<State, Action, ObservationSpace, ActionSpace>
/**
 * Type alias for `io.github.kotlinrl.core.wrapper.RunningStats`, which represents a utility
 * for maintaining running statistics like mean and standard deviation in a numerically stable manner.
 *
 * The `RunningStats` class is used to track statistics incrementally, and it adjusts
 * its internal state through updates for new data points. It ensures efficiency and stability
 * for dynamically computed statistical measures.
 */
typealias RunningStats = io.github.kotlinrl.core.wrapper.RunningStats
/**
 * A type alias for the `TimeLimit` wrapper class.
 *
 * This alias simplifies access to the `TimeLimit` wrapper, which is responsible for
 * enforcing a time limit on episodes in an environment. The wrapper terminates an episode
 * after a predefined maximum number of steps, even if the underlying environment itself
 * has not naturally ended. It is commonly used in reinforcement learning to restrict
 * episode length and avoid indefinite agent-environment interactions.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing actions executable in the environment.
 * @param ObservationSpace The type of space defining the structure of observation states.
 * @param ActionSpace The type of space defining the structure of allowable actions.
 */
typealias TimeLimit<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.TimeLimit<State, Action, ObservationSpace, ActionSpace>
/**
 * A type alias for the `TransformReward` class in the `io.github.kotlinrl.core.wrapper` package.
 *
 * Represents a customizable wrapper for an environment where the reward values are
 * transformed using a user-defined function. This allows for dynamic modification
 * of reward structures without altering the original environment.
 *
 * @param State The type representing the states of the environment.
 * @param Action The type representing the possible actions in the environment.
 * @param ObservationSpace The type defining the observation space of the environment.
 * @param ActionSpace The type defining the action space of the environment.
 */
typealias TransformReward<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.TransformReward<State, Action, ObservationSpace, ActionSpace>
/**
 * A type alias representing the `TransformState` wrapper class.
 *
 * This alias simplifies the reference to the `TransformState` class, which wraps an
 * existing environment and applies a transformation to its states. The transformed
 * states are exposed to the user while maintaining the original action and observation
 * spaces of the environment.
 *
 * @param State The type of the transformed state.
 * @param Action The type of actions supported by the environment.
 * @param ObservationSpace The observation space type for the transformed states.
 * @param ActionSpace The action space type of the environment.
 * @param WrappedState The raw state type of the underlying environment.
 * @param WrappedObservationSpace The observation space type for the raw states.
 */
typealias TransformState<State, Action, ObservationSpace, ActionSpace, WrappedState, WrappedObservationSpace> = io.github.kotlinrl.core.wrapper.TransformState<State, Action, ObservationSpace, ActionSpace, WrappedState, WrappedObservationSpace>
/**
 * Type alias for the `TransformAction` class.
 *
 * Represents a wrapper that modifies the action space and the behavior of the `step` function
 * by applying a transformation to actions before they are executed in a wrapped environment.
 * It enables mapping abstract actions to the specific actions understood by the wrapped environment.
 *
 * This transformation maintains the observation space and other properties of the wrapped environment
 * while introducing customization in the action execution process.
 *
 * @param State The type representing the state in the environment.
 * @param Action The type representing the abstract action to be transformed.
 * @param ObservationSpace The type representing the observation space of the environment.
 * @param ActionSpace The type representing the space of abstract actions in the transformed environment.
 * @param WrappedAction The type of action used by the wrapped environment after transformation.
 * @param WrappedActionSpace The type representing the space of actions specific to the wrapped environment.
 */
typealias TransformAction<State, Action, ObservationSpace, ActionSpace, WrappedAction, WrappedActionSpace> = io.github.kotlinrl.core.wrapper.TransformAction<State, Action, ObservationSpace, ActionSpace, WrappedAction, WrappedActionSpace>
