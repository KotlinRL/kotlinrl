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
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.io.*


val maxStepsPerEpisode = 4_500
val trainingEpisodes = 1_000
val testEpisodes = 30
val initialEpsilon = 0.99
val linearDecayRate = 0.001
val minEpsilon = 0.0
val gamma = 0.99
val fileName = "MazeOffPolicyMonteCarloControl.npy"

val env = Maze(render = true)
val recordEnv = RecordVideo(env = env, folder = "./videos/maze_off_policy_mcc", 10)

val policy = mk.d1array(25) { 0 }
val trainingQtable: QTable = mk.d2array(25, 4) { -10.0 }
(0 until trainingQtable.shape[1]).forEach { trainingQtable[24, it] = 0.0 }
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
        algorithm = OffPolicyMonteCarloControl(
            initialTargetPolicy = policy,
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
        policy = policy.pi()
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

val vTable = mk.d1array(25) { testingQtable[it].max() ?: 0.0 }

println("QTable")
printQTable(
    testingQtable, rows = 5, columns = 5, actionSymbols = mapOf(
        0 to "↑", 1 to "→", 2 to "↓", 3 to "←"
    )
)
println("Target Policy")
printPolicyGrid(
    policy, 5, 5, mapOf(
        0 to "↑", 1 to "→", 2 to "↓", 3 to "←"
    )
)
displayVideos(recordEnv.folder)
val plot1 = plotPolicyActionValueGrid(
    testingQtable, 5, 5, mapOf(
        0 to "↑", 1 to "→", 2 to "↓", 3 to "←"
    )
)
saveBufferedImageAsPng(plot1.toBufferedImage(), File(recordEnv.folder, "qtable_grid.png"))
val plot2 = plotPolicyStateValueGrid(
    policy, vTable, 5, 5, mapOf(
        0 to "↑", 1 to "→", 2 to "↓", 3 to "←"
    )
)
saveBufferedImageAsPng(plot2.toBufferedImage(), File(recordEnv.folder, "policy_grid.png"))
