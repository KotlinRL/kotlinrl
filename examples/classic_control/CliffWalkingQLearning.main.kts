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

val maxStepsPerEpisode = 600
val trainingEpisodes = 500
val testEpisodes = 50
val initialEpsilon = 0.3
val epsilonDecayRate = 0.000625
val minEpsilon = 0.05
val alpha = 0.5
val gamma = 0.99
val fileName = "CliffWalkingQLearning.npy"
val actionSymbols = mapOf(
    3 to "←",
    2 to "↓",
    1 to "→",
    0 to "↑"
)

val env = gymnasium.make<CliffWalkingEnv>(CliffWalking_v0, render = true, options = mapOf(
    "is_slippery" to false
))

var trainingQtable: QTable = mk.d2array(48, 4) { 0.0 }

val (epsilonSchedule, epsilonDecrement) = ParameterSchedule.linearDecay(
    initialValue = initialEpsilon,
    minValue = minEpsilon,
    decayRate = epsilonDecayRate,
    callback = { episode, parameter ->
        if (episode % 100 == 0) {
            println("Episode: $episode, Epsilon: $parameter")
        }
    }
)

val trainer = episodicTrainer(
    env = env,
    agent = learningAgent(
        id = "training",
        algorithm = QLearning(
            Q = trainingQtable,
            epsilon = epsilonSchedule,
            alpha = ParameterSchedule.constant(alpha),
            gamma = gamma,
        )
    ),
    maxStepsPerEpisode = maxStepsPerEpisode,
    successfulTermination = { it.done },
    callbacks = listOf(
        printEpisodeStart(100),
        onEpisodeEnd { epsilonDecrement() }
    )
)
println("Starting training")
val training = trainer.train(maxEpisodes(trainingEpisodes))
mk.writeNPY(fileName, trainingQtable)

val testingQtable = mk.readNPY<Double, D2>(fileName).asD2Array()

val recordEnv = RecordVideo(env = env, folder = "videos/cliff_walking_q_learning", testEpisodes / 3)
val tester = episodicTrainer(
    env = recordEnv,
    agent = policyAgent(
        id = "testing",
        policy = testingQtable.greedy()
    ),
    maxStepsPerEpisode = maxStepsPerEpisode,
    successfulTermination = { it.done },
    callbacks = listOf(
        printEpisodeStart(10)
    )
)
println("Starting testing")
val test = tester.train(maxEpisodes(testEpisodes))
println("Training average reward: ${training.totalAverageReward}")
println("Test average reward: ${test.totalAverageReward}")

printQTable(testingQtable, 4, 12, actionSymbols = actionSymbols)
displayVideos(recordEnv.folder)
val plot = plotPolicyActionValueGrid(testingQtable, 4, 12, actionSymbols )
saveBufferedImageAsPng(plot.toBufferedImage(), File(recordEnv.folder, "policy_grid.png"))
