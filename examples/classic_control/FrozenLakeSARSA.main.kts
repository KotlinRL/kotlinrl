#!/usr/bin/env kotlin -Djava.awt.headless=false -Dprism.order=sw

@file:Repository("https://repo1.maven.org/maven2")
@file:Repository("https://central.sonatype.com/repository/maven-snapshots/")

@file:DependsOn("io.github.kotlinrl:integration:0.1.0-SNAPSHOT")
@file:DependsOn("io.github.kotlinrl:tabular:0.1.0-SNAPSHOT")
@file:DependsOn("io.github.kotlinrl:envs:0.1.0-SNAPSHOT")
@file:DependsOn("io.github.kotlinrl:rendering:0.1.0-SNAPSHOT")

import io.github.kotlinrl.core.*
import io.github.kotlinrl.integration.gymnasium.*
import io.github.kotlinrl.integration.gymnasium.GymnasiumEnvs.*
import io.github.kotlinrl.rendering.*
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.kandy.letsplot.export.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.io.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import java.io.*
import kotlin.math.abs

val maxStepsPerEpisode = 200
val trainingEpisodes = 50_000
val testEpisodes = 50
val initialEpsilon = 0.6
val minEpsilon = 0.0
val epsilonDecayRate = (initialEpsilon - minEpsilon) / (trainingEpisodes * 0.9)
val alpha = ParameterSchedule.constant(0.2)
val minAlpha = ParameterSchedule.constant(0.02)
val gamma = 0.99
val fileName = "FrozenLakeSARSA.npy"
val actionSymbols = mapOf(
    0 to "←",
    1 to "↓",
    2 to "→",
    3 to "↑"
)

val env = gymnasium.make<FrozenLakeEnv>(FrozenLake_v1, render = true, options = mapOf(
    "is_slippery" to false,
    "map_name" to "8x8"
))

val trainingQtable: QTable = mk.rand<Double, D2>(from = 0.24, until = 0.26, dims = intArrayOf(64, 4))

val (epsilonSchedule, epsilonDecrement) = ParameterSchedule.linearDecay(
    initialValue = initialEpsilon,
    minValue = minEpsilon,
    decayRate = epsilonDecayRate,
    callback = { episode, parameter ->
        if (episode % 1000 == 0) {
            println("Episode: $episode, Epsilon: $parameter")
        }
    })
val phi = mk.d1array(64) { state ->
    (1.0 - gamma) * -(abs(7 - state / 8) + abs(7 - state % 8)).toDouble()
}

var currentState = 0
val trainer = episodicTrainer(
    env = TransformReward(env, transform = {
        val state = currentState
        val nextState = it.state
        currentState = if (it.terminated || it.truncated) 0 else nextState
        val isGoal = it.terminated && it.reward == 1.0
        val isHole = it.terminated && it.reward == 0.0
        val isSameState = state == nextState
        when {
            isGoal -> 1.0
            isHole -> -2.0
            isSameState -> 0.0
            else -> (50 * (gamma * phi[nextState] - phi[state])).coerceIn(-0.5, 0.5)
        }
    }
    ),
    agent = learningAgent(
        id = "training",
        algorithm = SARSA(
            Q = trainingQtable,
            epsilon = epsilonSchedule,
            alpha = { if (epsilonSchedule().decayStep > 8000) minAlpha() else alpha() },
            gamma = gamma,
        )
    ),
    maxStepsPerEpisode = maxStepsPerEpisode,
    warnOnTruncationOrMax = false,
    successfulTermination = { it.reward == 1.0 },
    callbacks = listOf(
        printEpisodeStart(1000),
        onEpisodeEnd {
            epsilonDecrement()
            if (it.totalEpisodes % 1_000 == 0) {
                val goalSuccessCount = TrainingResult(it.episodeStats.takeLast(1_000)).totalGoalSuccessCount
                println("Current goal success count: $goalSuccessCount, over the last 1000 episodes")
            }
        })
)
println("Starting training")
val training = trainer.train(maxEpisodes(trainingEpisodes).or {
    it.totalEpisodes >= 1000 && it.takeLast(1000).totalGoalSuccessCount == 1000
})
mk.writeNPY(fileName, trainingQtable)

val testingQtable = mk.readNPY<Double, D2>(fileName).asD2Array()
val recordEnv = RecordVideo(env = env, folder = "videos/frozen_lake_sarsa", testEpisodes / 3)
val tester = episodicTrainer(
    env = recordEnv,
    agent = policyAgent(id = "testing", policy = testingQtable.greedy()),
    maxStepsPerEpisode = maxStepsPerEpisode,
    successfulTermination = { it.reward == 1.0 },
    callbacks = listOf(printEpisodeStart(10))
)
println("Starting testing")
val test = tester.train(maxEpisodes(testEpisodes))

println("Training average reward: ${training.totalAverageReward}")
println("Test average reward: ${test.totalAverageReward}")

displayVideos(recordEnv.folder)
printQTable(testingQtable, 8, 8, actionSymbols = actionSymbols)
val plot = plotPolicyActionValueGrid(testingQtable, 8, 8, actionSymbols)
saveBufferedImageAsPng(plot.toBufferedImage(), File(recordEnv.folder, "policy_grid.png"))
