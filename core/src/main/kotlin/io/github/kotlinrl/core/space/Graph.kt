package io.github.kotlinrl.core.space

import kotlin.random.*

/**
 * Represents an observation of a graph, including node features, edges, and optional edge features.
 *
 * @param N The type of the features associated with nodes in the graph.
 * @param E The type of the features associated with edges in the graph.
 * @property nodeFeatures A list where each element represents the feature for a specific node in the graph.
 * The index of the element corresponds to the node's identifier.
 * @property edges A list of pairs of integers, where each pair represents a directed edge in the graph.
 * The first integer in a pair represents the source node, and the second integer represents the target node.
 * @property edgeFeatures An optional list where each element corresponds to the feature for a specific edge,
 * matching the order of the edges in the `edges` property. If null, edges are assumed to have no features.
 */
data class GraphObservation<N, E>(
    val nodeFeatures: List<N>,
    val edges: List<Pair<Int, Int>>,
    val edgeFeatures: List<E>? = null
)

/**
 * A class representing a space of graphs, where nodes and edges can have associated features.
 *
 * @param N The type of features associated with the nodes of the graph.
 * @param E The type of features associated with the edges of the graph.
 * @param nodeSpace The space from which node features can be sampled.
 * @param edgeSpace An optional space for sampling edge features. If null, edges are assumed to have no features.
 * @param seed An optional seed for random number generation. If null, a default random generator is used.
 */
class Graph<N, E>(
    val nodeSpace: Space<N>,
    val edgeSpace: Space<E>? = null,
    seed: Int? = null
) : Space<GraphObservation<N, E>> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    /**
     * Samples a random graph observation, including nodes and edges, from the specified feature spaces.
     *
     * The number of nodes is randomly chosen between 3 and 10 (inclusive). Edges are sampled randomly
     * between these nodes, with the number of edges being in the range from 0 to 2 times the number of nodes.
     * Node features are sampled based on the `nodeSpace`. If an `edgeSpace` is provided, edge features
     * are also sampled; otherwise, edges will have no associated features.
     *
     * @return A randomly sampled graph observation containing nodes, edges, and optionally edge features.
     */
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

    /**
     * Determines whether the specified value is contained within this graph structure.
     *
     * The method checks if the given value:
     * - Is an instance of `GraphObservation`.
     * - Contains node features that are compatible with the `nodeSpace`.
     * - If `edgeSpace` is defined, contains edge features that are compatible with the `edgeSpace` (and matches the number of edges).
     * - Has valid edges where the node indices are within valid bounds.
     *
     * @param value The object to be checked for membership. This is expected to be a `GraphObservation`, but the method will handle cases where the type does not match.
     * @return `true` if the value is a valid `GraphObservation` and satisfies the conditions defined by the graph; `false` otherwise.
     */
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
