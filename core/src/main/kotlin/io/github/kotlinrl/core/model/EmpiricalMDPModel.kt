package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.*

/**
 * A model for handling empirical Markov Decision Processes (MDPs) using a sampling-based strategy.
 * This class works with a model-based environment and approximates transition dynamics and expected rewards
 * based on sampled trajectories from the environment.
 *
 * @param State The type representing states in the MDP.
 * @param Action The type representing actions in the MDP.
 * @param env The underlying model-based environment used for simulating steps in the MDP.
 * @param allStates A predefined list of all possible states. Defaults to an empty list if not provided.
 * @param allActions A predefined list of all possible actions. Defaults to an empty list if not provided.
 * @param numSamples The number of samples to use for simulating steps to compute transitions and rewards.
 *                   A higher value provides more accurate estimates at the cost of increased computational effort.
 */
class EmpiricalMDPModel<State, Action>(
    private val env: ModelBasedEnv<State, Action, *, *>,
    private val allStates: List<State> = emptyList(),
    private val allActions: List<Action> = emptyList(),
    private val numSamples: Int = 100
) : MDPModel<State, Action> {

    /**
     * Computes the probabilistic transitions for a given state and action based on simulated outcomes.
     * This method collects the probabilities of possible transitions by sampling the outcomes of the action
     * in a specific state a predetermined number of times.
     *
     * @param state The current state from which the action is executed.
     * @param action The action to simulate in the provided state.
     * @return A list of probabilistic transitions, where each transition includes the next state,
     * the reward, the probability of the transition, and whether the resulting state is terminal.
     */
    override fun transitions(state: State, action: Action): ProbabilisticTrajectory<State, Action> {
        val outcomes = mutableMapOf<Transition<State, Action>, Int>()

        repeat(numSamples) {
            val (nextState, reward, terminated, truncated) = env.simulateStep(state, action)
            val key = Transition(state, action, reward, nextState, terminated, truncated)
            outcomes[key] = outcomes.getOrDefault(key, 0) + 1
        }

        return outcomes.entries.map { (key, count) ->
            val transition = key
            ProbabilisticTransition(
                state = transition.state,
                action = transition.action,
                reward = transition.reward,
                nextState = transition.nextState,
                probability = count.toDouble() / numSamples,
                done = transition.done
            )
        }
    }

    /**
     * Estimates the expected reward for performing a given action in a specific state by simulating
     * the action multiple times and averaging the rewards obtained.
     *
     * @param state The current state from which the action is executed.
     * @param action The action to simulate in the given state.
     * @return The average reward computed based on multiple simulated outcomes of the action in the state.
     */
    override fun expectedReward(state: State, action: Action): Double {
        var totalReward = 0.0

        repeat(numSamples) {
            val (_, reward, _, _) = env.simulateStep(state, action)
            totalReward += reward
        }

        return totalReward / numSamples
    }

    /**
     * Retrieves a list of all states discovered or modeled within the current MDP model.
     *
     * @return A list containing all states represented in the MDP model.
     */
    override fun allStates(): List<State> = allStates

    /**
     * Retrieves a list of all actions available in the model.
     *
     * @return A list containing all actions represented in the MDP model.
     */
    override fun allActions(): List<Action> = allActions
}
