package io.github.kotlinrl.core

import io.github.kotlinrl.core.env.Env
import io.github.kotlinrl.core.env.InitialState
import io.github.kotlinrl.core.env.ModelBasedEnv
import io.github.kotlinrl.core.env.Rendering
import io.github.kotlinrl.core.env.StepResult

/**
 * A type alias for `io.github.kotlinrl.core.env.InitialState`.
 *
 * Represents the initial state of an environment along with any auxiliary metadata
 * necessary for proper initialization or reset of the environment. Encapsulates both
 * the state and additional configuration or contextual details in a structured format.
 *
 * Provides a simpler reference to the `io.github.kotlinrl.core.env.InitialState` type
 * within the codebase.
 *
 * @param State The type parameter representing the state of the environment.
 */
typealias InitialState<State> = InitialState<State>
/**
 * Type alias for `StepResult`, representing the result of a single step in an environment.
 *
 * This alias provides a more concise way to refer to the `StepResult` data class, which encapsulates
 * information about the state transition, reward, termination, episode truncation, and additional metadata
 * from executing an action in an environment.
 *
 * @param State The type representing the environment state.
 */
typealias StepResult<State> = StepResult<State>
/**
 * Provides a type alias for the `Rendering` class in the `io.github.kotlinrl.core.env` package.
 *
 * This alias simplifies the reference to the `Rendering` class, which represents the visual
 * or data representation of an environment's state. It is used to encapsulate rendering results
 * that may include empty renderings or frames with specific dimensions and raw data.
 */
typealias Rendering = Rendering
/**
 * A type alias for the `RenderFrame` class from the `io.github.kotlinrl.core.env.Rendering` package.
 *
 * `RenderFrame` represents a single rendered frame in the rendering process.
 * It encapsulates the width, height, and raw byte data of the frame, providing
 * a structured representation of the rendering output within an environment.
 */
typealias RenderFrame = Rendering.RenderFrame
/**
 * A type alias for `io.github.kotlinrl.core.env.ModelBasedEnv`, representing an environment
 * with explicit transition dynamics that can be simulated. This type alias provides a simplified
 * way to work with environments whose behavior can be predicted or simulated without directly
 * modifying their internal state.
 *
 * The model-based environment extends the generic environment model, adding functionality to
 * hypothetically evaluate the outcome of state-action pairs, including resulting states, rewards,
 * and termination conditions.
 *
 * @param State The type representing the states of the environment.
 * @param Action The type representing the actions available in the environment.
 * @param ObservationSpace The structure or constraints defining the valid states in the environment.
 * @param ActionSpace The structure or constraints defining the valid actions in the environment.
 */
typealias ModelBasedEnv<State, Action, ObservationSpace, ActionSpace> = ModelBasedEnv<State, Action, ObservationSpace, ActionSpace>
/**
 * Typealias for `io.github.kotlinrl.core.env.Env`, representing a generic interface
 * for environments in reinforcement learning or simulation contexts.
 *
 * This abstraction serves as the foundation for agent-environment interaction,
 * encompassing functionalities such as stepping through actions, resetting the state,
 * rendering, and managing spaces for observations and actions. It provides
 * standardized methods and properties to model a variety of environments.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing actions that can be executed within the environment.
 * @param ObservationSpace The type representing the space structure for observations.
 * @param ActionSpace The type representing the space structure for allowable actions.
 */
typealias Env<State, Action, ObservationSpace, ActionSpace> = Env<State, Action, ObservationSpace, ActionSpace>
