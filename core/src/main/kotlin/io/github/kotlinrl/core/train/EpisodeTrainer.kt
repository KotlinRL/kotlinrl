package io.github.kotlinrl.core.train

import io.github.kotlinrl.core.*

class EpisodeTrainer<State, Action>(
    private val env: Env<State, Action, *, *>,
    private val agent: Agent<State, Action>,
    private val maxStepsPerEpisode: Int = 10_000,
    private val callbacks: List<EpisodeCallback<State, Action>> = emptyList()
) : Trainer {

    override fun train(episodes: Int): TrainingResult {
        val episodeRewards = mutableListOf<Double>()

        for(episode in 1 until episodes + 1) {
            callbacks.forEach { it.onEpisodeStart(episode) }

            val transitions = mutableListOf<Transition<State, Action>>()
            val actions = mutableListOf<Action>()

            var state = env.reset().state
            var totalReward = 0.0
            var done = false

            while (!done && transitions.size < maxStepsPerEpisode) {
                val action = agent.act(state)
                val stepResult = env.step(action)
                totalReward += stepResult.reward

                val transition = Transition(
                    state = state,
                    action = action,
                    nextState = stepResult.state,
                    reward = stepResult.reward,
                    terminated = stepResult.terminated,
                    truncated = stepResult.truncated,
                    info = stepResult.info
                )

                agent.observe(transition)

                transitions += transition
                actions += action

                state = transition.nextState
                done = transition.done
            }

            val stats = EpisodeStats(
                episode = episode,
                totalReward = totalReward,
                steps = transitions.size,
                transitions = transitions
            )

            callbacks.forEach { it.onEpisodeEnd(stats) }
            episodeRewards += totalReward
        }

        return TrainingResult(episodeRewards)
    }
}
