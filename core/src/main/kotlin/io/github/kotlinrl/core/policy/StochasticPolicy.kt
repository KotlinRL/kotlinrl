package io.github.kotlinrl.core.policy

import kotlin.random.*

/**
 * A stochastic policy implementation that selects actions according to a probability distribution
 * derived from a scoring mechanism. This class defines the framework for evaluating actions and
 * their scores, calculating probabilities, and selecting actions based on stochastic sampling.
 *
 * @param State the type representing the state of the environment.
 * @param Action the type representing the available actions in the environment.
 * @param rng a `Random` instance used for stochastic sampling of actions.
 */
abstract class StochasticPolicy<State, Action>(
    protected val rng: Random
) : Policy<State, Action> {

    /**
     * Computes a list of actions and their associated scores for a given state.
     * This method evaluates potential actions that can be taken in the provided state
     * and assigns a score to each action based on the policy's underlying scoring mechanism.
     *
     * @param state the current state of the environment for which actions and scores
     * are to be determined.
     * @return a list of pairs where each pair consists of an action and its corresponding
     * score, representing the action's quality or suitability in the given state.
     */
    abstract fun actionScores(state: State): List<Pair<Action, Double>>

    /**
     * Determines the action to be taken for the given state by processing the associated
     * action-scores pairs and sampling based on their computed probabilities.
     *
     * @param state the current state of the environment for which the action is to be selected.
     * @return the sampled action to be performed in the given state.
     */
    override fun invoke(state: State): Action {
        val (actions, scores) = actionScores(state).unzip()
        return calculateAndSample(scores, actions)
    }

    /**
     * Calculates the probability of selecting a specific action in a given state
     * based on the underlying scoring mechanism and probability distribution computed
     * by the policy.
     *
     * @param state the current state of the environment for which the probability of the action
     * is to be determined.
     * @param action the action whose probability of being selected is to be calculated.
     * @return the probability of selecting the specified action in the given state,
     * or 0.0 if the action is not available in the list of scored actions.
     */
    override fun probability(state: State, action: Action): Double {
        val actionScoreList = actionScores(state)
        val actions = actionScoreList.map { it.first }
        val scores = actionScoreList.map { it.second }
        val probs = computeProbabilities(scores)
        return actions.indexOf(action).let { if (it >= 0) probs[it] else 0.0 }
    }

    /**
     * Computes the normalized probabilities for a list of scores such that each score is divided
     * by the sum of all scores, resulting in a probability distribution.
     *
     * @param scores a list of non-negative scores representing the unnormalized weights used to compute probabilities.
     *               The list should not be empty, and all elements must be non-negative.
     * @return a list of probabilities corresponding to the input scores. The probabilities will sum to 1.0.
     */
    protected fun computeProbabilities(scores: List<Double>): List<Double> {
        val sum = scores.sum()
        return scores.map { it / sum }
    }

    /**
     * Calculates the probabilities for a list of scores and samples an action
     * based on the corresponding probabilities.
     *
     * @param scores a list of scores representing the weights for calculating
     *               probabilities. Each score corresponds to an action.
     * @param actions a list of actions from which one will be sampled based on the
     *                calculated probabilities.
     * @return the sampled action based on the calculated probabilities.
     */
    private fun calculateAndSample(scores: List<Double>, actions: List<Action>): Action {
        val probs = computeProbabilities(scores)
        return sample(actions, probs)
    }

    /**
     * Samples an action from a given list of actions based on their associated probabilities.
     *
     * The method assumes that the `probabilities` list contains values that sum
     * to 1.0 and corresponds to the `actions` list in a one-to-one mapping.
     * A random number is generated and used to determine which action
     * to select based on the cumulative sum of probabilities.
     *
     * @param actions the list of actions to sample from.
     * @param probabilities the list of probabilities corresponding to each action, which should sum to 1.0.
     * @return the sampled action based on the probabilities.
     */
    private fun sample(actions: List<Action>, probabilities: List<Double>): Action {
        val rand = rng.nextDouble()
        var cumulative = 0.0
        for (i in actions.indices) {
            cumulative += probabilities[i]
            if (rand < cumulative) return actions[i]
        }
        return actions.last()
    }
}