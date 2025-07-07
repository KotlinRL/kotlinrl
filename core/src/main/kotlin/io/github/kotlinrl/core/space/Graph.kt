package io.github.kotlinrl.core.space

import kotlin.random.Random

data class GraphObservation<N, E>(
    val nodeFeatures: List<N>,
    val edges: List<Pair<Int, Int>>,
    val edgeFeatures: List<E>? = null
)

class Graph<N, E>(
    val nodeSpace: Space<N>,
    val edgeSpace: Space<E>? = null,
    seed: Int? = null
) : Space<GraphObservation<N, E>> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    override fun sample(): GraphObservation<N, E> {
        // Example: randomly sample 3-10 nodes and 0-2*nodes edges
        val nodeCount = random.nextInt(3, 11)
        val nodes = List(nodeCount) { nodeSpace.sample() }
        val edgeCount = random.nextInt(0, nodeCount * 2)
        val edges = List(edgeCount) {
            Pair(random.nextInt(nodeCount), random.nextInt(nodeCount))
        }
        val edgeFeats = edgeSpace?.let { es -> List(edgeCount) { es.sample() } }
        return GraphObservation(nodes, edges, edgeFeats)
    }

    override fun contains(value: Any?): Boolean {
        if (value !is GraphObservation<*, *>) return false
        if (value.nodeFeatures.any { !nodeSpace.contains(it) }) return false
        if (edgeSpace != null && value.edgeFeatures != null) {
            if (value.edgeFeatures.size != value.edges.size) return false
            if (value.edgeFeatures.any { !edgeSpace.contains(it) }) return false
        }
        val nodeCount = value.nodeFeatures.size
        if (value.edges.any { it.first !in 0 until nodeCount || it.second !in 0 until nodeCount }) return false
        return true
    }
}
