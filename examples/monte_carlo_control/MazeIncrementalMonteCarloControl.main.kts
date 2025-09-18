#!/usr/bin/env kotlin -Djava.awt.headless=false -Dprism.order=sw

@file:Repository("https://repo1.maven.org/maven2")
@file:Repository("https://central.sonatype.com/repository/maven-snapshots/")

@file:DependsOn("io.github.kotlinrl:integration:0.1.0-SNAPSHOT")
@file:DependsOn("io.github.kotlinrl:tabular:0.1.0-SNAPSHOT")
@file:DependsOn("io.github.kotlinrl:envs:0.1.0-SNAPSHOT")
@file:DependsOn("io.github.kotlinrl:rendering:0.1.0-SNAPSHOT")

import io.github.kotlinrl.core.*
import io.github.kotlinrl.envs.*
import io.github.kotlinrl.rendering.*
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.kandy.letsplot.export.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.io.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import java.io.*

val maxStepsPerEpisode = 500
val trainingEpisodes = 1_000
val testEpisodes = 30
val initialEpsilon = 0.8
val linearDecayRate = 0.00089
val minEpsilon = 0.0
val gamma = 0.99
val fileName = "MazeIncrementalMonteCarloControl.npy"

val env = Maze(render = true)
val recordEnv = RecordVideo(env = env, folder = "./videos/maze_incremental_mcc", 10)

val trainingQtable: QTable = mk.d2array(25, 4) { 0.0 }
val (epsilonSchedule, epsilonDecay) = ParameterSchedule.linearDecay(
    initialValue = initialEpsilon,
    minValue = minEpsilon,
    decayRate = linearDecayRate,
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
        algorithm = OnPolicyMonteCarloControl(
            epsilon = epsilonSchedule,
            Q = trainingQtable,
            gamma = gamma,
            firstVisitOnly = true,
        ),
    ),
    maxStepsPerEpisode = maxStepsPerEpisode,
    successfulTermination = { it.done },
    callbacks = listOf(
        printEpisodeStart(10),
        onEpisodeEnd { epsilonDecay() }
    )
)
println("Starting training")
val training = trainer.train(maxEpisodes(trainingEpisodes))
mk.writeNPY(fileName, trainingQtable)

val testingQtable = mk.readNPY<Double, D2>(fileName).asD2Array()
val tester = episodicTrainer(
    env = recordEnv,
    agent = policyAgent(
        id = "testing",
        policy = testingQtable.greedy()
    ),
    maxStepsPerEpisode = maxStepsPerEpisode,
    successfulTermination = { it.reward == 0.0 },
    callbacks = listOf(
        printEpisodeStart(10)
    )
)
println("Starting testing")
val test = tester.train(maxEpisodes(testEpisodes))
println("Training average reward: ${training.totalAverageReward}")
println("Test average reward: ${test.totalAverageReward}")

printQTable(
    testingQtable, rows = 5, columns = 5, actionSymbols = mapOf(
        0 to "↑", 1 to "→", 2 to "↓", 3 to "←"
    )
)
displayVideos(recordEnv.folder)
val plot = plotPolicyActionValueGrid(
    testingQtable, 5, 5, mapOf(
        0 to "↑", 1 to "→", 2 to "↓", 3 to "←"
    )
)
saveBufferedImageAsPng(plot.toBufferedImage(), File(recordEnv.folder, "policy_grid.png"))
