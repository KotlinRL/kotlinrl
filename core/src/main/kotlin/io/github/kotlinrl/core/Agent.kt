package io.github.kotlinrl.core

import java.util.*

/**
 * Represents a trajectory in a reinforcement learning context, which is a sequence of transitions
 * observed during an episode of interaction between an agent and the environment.
 *
 * A trajectory comprises a list of transitions, where each transition contains information about
 * the state, action, reward, next state, and termination or truncation status.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias Trajectory<State, Action> = List<io.github.kotlinrl.core.agent.Transition<State, Action>>
/**
 * A type alias for the `Agent` interface, representing an abstraction for agents interacting with environments
 * in a reinforcement learning setup. This alias serves to simplify usage references within the primary codebase.
 *
 * An agent observes its environment's state, decides on actions based on its policy or logic, and has the ability
 * to adapt or learn from feedback like state transitions or trajectories. This abstraction is foundational for
 * implementing reinforcement learning solutions.
 *
 * @param State The type parameter representing the state space of the environment.
 * @param Action The type parameter representing the action space of the environment.
 */
typealias Agent<State, Action> = io.github.kotlinrl.core.agent.Agent<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.agent.Transition`, representing a single step
 * interaction between an agent and the environment in reinforcement learning.
 *
 * This alias is used to simplify the reference to the `Transition` class, which provides
 * structured information about the state, action, reward, next state, and termination
 * details of a transition within an environment.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias Transition<State, Action> = io.github.kotlinrl.core.agent.Transition<State, Action>
/**
 * Type alias for `TransitionObserver`, representing a functional interface used to observe
 * state-action transitions in reinforcement learning environments.
 *
 * Provides a mechanism for receiving and processing transition events, enabling actions such
 * as logging, learning updates, or customizing behaviors during transitions.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias TransitionObserver<State, Action> = io.github.kotlinrl.core.agent.TransitionObserver<State, Action>
/**
 * A type alias for the `TrajectoryObserver` functional interface from the KotlinRL library.
 *
 * This alias provides a shorthand for referencing the observer, responsible for processing
 * trajectories in reinforcement learning environments. It observes sequences of transitions
 * during an episode, including state-action interactions, rewards, and resulting states.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias TrajectoryObserver<State, Action> = io.github.kotlinrl.core.agent.TrajectoryObserver<State, Action>
/**
 * A type alias representing an agent that follows a specific policy for decision-making
 * in a reinforcement learning environment.
 *
 * The `PolicyAgent` type encapsulates logic for selecting actions based on states, observing
 * transitions, and managing complete trajectories. This alias simplifies the reference to
 * the `PolicyAgent` class within the codebase.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias PolicyAgent<State, Action> = io.github.kotlinrl.core.agent.PolicyAgent<State, Action>
/**
 * Represents a type alias for the `LearningAgent` class from the `io.github.kotlinrl.core.agent` package.
 *
 * A `LearningAgent` is an abstraction for agents in reinforcement learning systems that can
 * interact with an environment, learn from feedback, and adapt their behavior over time.
 *
 * @param State The type that defines the state space of the environment in which the agent operates.
 * @param Action The type that defines the action space of the agent to interact with the environment.
 */
typealias LearningAgent<State, Action> = io.github.kotlinrl.core.agent.LearningAgent<State, Action>

/**
 * Creates a policy-based agent that selects actions based on a given policy and optionally observes
 * transitions and trajectories in a reinforcement learning environment.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param id A unique identifier for the agent. Defaults to a randomly generated UUID string.
 * @param policy The policy the agent uses to select actions based on observed states.
 * @param onTransition An optional callback invoked for each individual state-action transition.
 *                     Defaults to an empty observer that performs no action.
 * @param onTrajectory An optional callback invoked for processing trajectories (sequences of transitions)
 *                     across episodes. Defaults to an empty observer that performs no action.
 * @return An instance of the agent with the specified policy and observation capabilities.
 */
fun <State, Action> policyAgent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<State, Action>,
    onTransition: TransitionObserver<State, Action> = TransitionObserver { },
    onTrajectory: TrajectoryObserver<State, Action> = TrajectoryObserver { _, _ -> }
): Agent<State, Action> = PolicyAgent(
    id = id,
    policy = policy,
    onTransition = onTransition,
    onTrajectory = onTrajectory,
)

/**
 * Creates a learning agent that can interact with an environment, adapt its behavior through
 * feedback, and make decisions using a specified learning algorithm.
 *
 * @param id The unique identifier of the learning agent. Defaults to a randomly generated UUID.
 * @param algorithm The learning algorithm used by the agent to determine actions and update its behavior.
 * @return An instance of a learning agent configured with the specified identifier and algorithm.
 */
fun <State, Action> learningAgent(
    id: String = UUID.randomUUID().toString(),
    algorithm: LearningAlgorithm<State, Action>,
): Agent<State, Action> = LearningAgent(
    id = id,
    algorithm = algorithm
)

/**
 * Extends the functionality of an Agent by allowing observation of transitions
 * to be intercepted and handled with a custom callback. The callback function
 * is executed whenever a transition is observed.
 *
 * @param observer A lambda function invoked with each observed transition.
 *                 The transition contains details about the state, action,
 *                 reward, next state, and termination/truncation status.
 * @return A new Agent that delegates all its operations to the current Agent
 *         while intercepting and invoking the callback on each observed transition.
 */
inline fun <State, Action> Agent<State, Action>.onTransition(
    crossinline observer: (Transition<State, Action>) -> Unit
): Agent<State, Action> =
    object : Agent<State, Action> by this {
        override fun observe(transition: Transition<State, Action>) {
            this@onTransition.observe(transition)
            observer(transition)
        }
    }

/**
 * Enhances an agent to invoke a callback whenever it observes a trajectory of transitions during an episode.
 *
 * This function wraps the original agent and calls the provided callback whenever the `observe` method
 * is triggered with a trajectory and episode. The callback allows for additional processing or side effects
 * based on the observed trajectory and episode number.
 *
 * @param callback A lambda function invoked when a trajectory is observed. It takes a `Trajectory` object
 * and the episode index as parameters.
 * @return A new agent that extends the original functionality by invoking the provided callback on trajectory observations.
 */
inline fun <State, Action> Agent<State, Action>.onTrajectory(
    crossinline callback: (Trajectory<State, Action>, Int) -> Unit
): Agent<State, Action> =
    object : Agent<State, Action> by this {
        override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
            this@onTrajectory.observe(trajectory, episode)
            callback(trajectory, episode)
        }
    }

/**
 * Registers a callback to handle state-action transitions in a reinforcement learning environment.
 * This function creates an observer for transitions, which can be used for tasks such as logging,
 * updating learning models, or monitoring the behavior of the environment.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param callback The function to be invoked for each transition. The callback will receive a `Transition`
 * object containing the current state, action taken, reward received, resulting state, and other metadata.
 * @return A `TransitionObserver` that observes and processes each transition using the provided callback.
 */
inline fun <State, Action> onTransition(
    crossinline callback: (Transition<State, Action>) -> Unit
): TransitionObserver<State, Action> = TransitionObserver { callback(it) }

/**
 * Creates a trajectory observer that invokes the provided callback function.
 *
 * This function allows monitoring and processing of trajectories in reinforcement learning environments
 * by creating a `TrajectoryObserver` instance that delegates to the provided callback.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param callback A function that is called with a trajectory and its associated episode index.
 *                 The trajectory represents a sequence of transitions, while the episode index
 *                 identifies the episode being observed.
 * @return A `TrajectoryObserver` that uses the provided callback for processing trajectories.
 */
inline fun <State, Action> onTrajectory(
    crossinline callback: (Trajectory<State, Action>, Int) -> Unit
): TrajectoryObserver<State, Action> = TrajectoryObserver { t, c -> callback(t, c) }