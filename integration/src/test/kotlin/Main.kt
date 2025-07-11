import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.*
import io.github.kotlinrl.core.learn.mcc.*
import io.github.kotlinrl.core.policy.*
import io.github.kotlinrl.core.train.*
import io.github.kotlinrl.integration.gymnasium.*
import io.github.kotlinrl.integration.gymnasium.GymnasiumEnvs.*

fun main() {
    val env = gymnasium.make<BlackjackEnv>(Blackjack_v1, seed = 123, render = true)

    val episodeLogger = object : EpisodeCallback<List<Any>, Int> {
        override fun onEpisodeStart(episode: Int) {
            if (episode % 10_000 == 0) println("Starting episode $episode")
        }
    }
    val actionListProvider = StateActionListProvider<List<Any>, Int>{ listOf(0, 1) }

    val qTable = QTable(
        shape = intArrayOf(32, 10, 2, 2),
        indexFor = { state: List<Any>, action: Int ->
            val (playerSum, dealerSum, usableAce) = state.map { it as Int }
            intArrayOf(playerSum, dealerSum - 1, usableAce, action)
        },
        actionListProvider = actionListProvider
    )

    val trainingAgent = monteCarloAgent(
        id = "training",
        policy = epsilonGreedyPolicy(
            stateActionListProvider = actionListProvider,
            explorationFactor = constantEpsilon(0.1),
            qTable = qTable
        )
    )
    val trainer = trainer(
        env = env,
        agent = trainingAgent,
        maxStepsPerEpisode = 1,
        callbacks = listOf(
            MonteCarloControl(qTable, 0.99),
            episodeLogger
        )
    )
    println("Starting training")
    val training = trainer.train(100_000)

    val testingAgent = agent(
        id = "testing",
        policy = greedyPolicy(
            stateActionListProvider = actionListProvider,
            qTable = qTable
        )
    )
    val tester = trainer(
        env = env,
        agent = testingAgent,
        maxStepsPerEpisode = 1,
        callbacks = listOf(episodeLogger)
    )
    println("Starting testing")
    val test = tester.train(100_000)

    println("Training Results: ${training.episodeRewards.sum() / training.episodeRewards.size}")
    println("Test Results: ${test.episodeRewards.sum() / test.episodeRewards.size}")

    println("Q-value for [20, 10, 1, Stick]: ${qTable[listOf(20, 10, 1), 0]}")
    println("Q-value for [20, 10, 1, Hit]:   ${qTable[listOf(20, 10, 1), 1]}")
}