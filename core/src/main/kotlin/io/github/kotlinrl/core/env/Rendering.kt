package io.github.kotlinrl.core.env

/**
 * Represents a rendering output for an environment's visual or data representation.
 *
 * This sealed class is used to encapsulate either an empty or non-empty rendering of the
 * current state of an environment. The subclasses determine the specific type of rendering.
 */
sealed class Rendering {
    /**
     * Represents an empty rendering result.
     *
     * This object is returned when the rendering operation does not produce
     * any visual or data output. It indicates the absence of a rendered frame
     * or visual representation in the current context.
     */
    object Empty : Rendering()
    /**
     * Represents a single rendered frame within the rendering process.
     *
     * This data class encapsulates the dimensions and raw byte data of a rendering output.
     * It is typically used to store visual or data representations of a rendered state.
     *
     * @property width The width of the rendered frame in pixels.
     * @property height The height of the rendered frame in pixels.
     * @property bytes The raw byte array representing the rendered frame.
     */
    data class RenderFrame(
        val width: Int,
        val height: Int,
        val bytes: ByteArray
    ) : Rendering() {
        /**
         * Compares this object with another object to determine equality.
         *
         * This method checks whether the given object is equal to the current instance by
         * comparing their types and properties. Specifically, it verifies that the other object is
         * of the same class, and that their `width`, `height`, and `bytes` properties are identical.
         *
         * @param other The object to be compared with the current instance.
         * @return `true` if the objects are equal; `false` otherwise.
         */
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as RenderFrame

            if (width != other.width) return false
            if (height != other.height) return false
            if (!bytes.contentEquals(other.bytes)) return false

            return true
        }

        /**
         * Computes the hash code for this instance.
         *
         * This method generates a hash code based on the `width`, `height`, and `bytes` properties
         * of the object. It ensures consistent hashing for objects with equivalent state.
         *
         * @return A hash code value for this object.
         */
        override fun hashCode(): Int {
            var result = width
            result = 31 * result + height
            result = 31 * result + bytes.contentHashCode()
            return result
        }
    }
}