#!/usr/bin/env kotlin -Djava.awt.headless=false -Dprism.order=sw

@file:Repository("https://repo1.maven.org/maven2")
@file:Repository("https://central.sonatype.com/repository/maven-snapshots/")

@file:DependsOn("io.github.kotlinrl:integration:0.1.0-SNAPSHOT")
@file:DependsOn("io.github.kotlinrl:tabular:0.1.0-SNAPSHOT")
@file:DependsOn("io.github.kotlinrl:envs:0.1.0-SNAPSHOT")
@file:DependsOn("io.github.kotlinrl:rendering:0.1.0-SNAPSHOT")

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.RecordVideo
import io.github.kotlinrl.core.api.*
import io.github.kotlinrl.core.wrapper.*
import io.github.kotlinrl.integration.gymnasium.*
import io.github.kotlinrl.integration.gymnasium.GymnasiumEnvs.*
import io.github.kotlinrl.rendering.*
import io.github.kotlinrl.tabular.*
import io.github.kotlinrl.tabular.td.classic.*
import org.jetbrains.kotlinx.kandy.letsplot.export.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.io.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import java.io.*

val maxStepsPerEpisode = 205
val trainingEpisodes = 5_000
val testEpisodes = 50
val initialEpsilon = 0.8
val epsilonDecayRate = 0.000177
val minEpsilon = 0.0
val alpha = 0.5
val gamma = 0.99
val fileName = "TaxiExpectedSARSA.npy"

val env = gymnasium.make<CliffWalkingEnv>(Taxi_v3, render = true)

var trainingQtable: QTable = mk.d2array(500, 6) { 0.0 }

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
        algorithm = SARSA(
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

val recordEnv = RecordVideo(env = env, folder = "videos/taxi_expected_sarsa", testEpisodes / 3)
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

displayVideos(recordEnv.folder)
