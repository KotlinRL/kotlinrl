package io.github.kotlinrl.core.train

import io.github.kotlinrl.core.*

class EpisodeTrainer<State, Action>(
    private val env: Env<State, Action, *, *>,
    private val agent: Agent<State, Action>,
    private val maxStepsPerEpisode: Int = 10_000,
    private val successfulTermination: SuccessfulTermination<State, Action>,
    private val callbacks: List<EpisodeCallback<State, Action>> = emptyList(),
) : Trainer {

    override fun train(stopCondition: TrainingStopCondition): TrainingResult {
        val episodeStats = mutableListOf<EpisodeStats<State, Action>>()
        var episode = 1

        while(true) {
            callbacks.forEach { it.onEpisodeStart(episode) }

            val transitions = mutableListOf<Transition<State, Action>>()
            val actions = mutableListOf<Action>()

            var state = env.reset().state
            var totalReward = 0.0
            var done = false
            var exception: Throwable? = null

            while (!done && transitions.size < maxStepsPerEpisode) {
                try {
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
                } catch (e: Exception) {
                    done = true
                    exception = e
                }
            }

            val stats = EpisodeStats(
                trajectory = transitions,
                episode = episode,
                steps = transitions.size,
                reachedGoal = transitions.lastOrNull()?.let { successfulTermination(it) } ?: false,
                info = exception?.let { mapOf("exception" to it) } ?: emptyMap()
            )
            agent.observe(transitions, episode)
            callbacks.forEach { it.onEpisodeEnd(stats) }
            episodeStats += stats

            val result = TrainingResult(episodeStats)
            if (stopCondition(result)) {
                return result
            }
            episode++
        }
    }
}
