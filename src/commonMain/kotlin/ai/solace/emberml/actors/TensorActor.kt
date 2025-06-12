package ai.solace.emberml.actors

import ai.solace.emberml.tensor.common.EmberTensor

/**
 * Messages for tensor operations.
 */
sealed class TensorMessage : ActorMessage

/**
 * Message to perform tensor addition.
 */
data class AddTensorMessage(
    val a: EmberTensor,
    val b: EmberTensor,
    val responseChannel: kotlinx.coroutines.channels.SendChannel<TensorResultMessage>? = null
) : TensorMessage()

/**
 * Message to perform tensor multiplication.
 */
data class MultiplyTensorMessage(
    val a: EmberTensor,
    val b: EmberTensor,
    val responseChannel: kotlinx.coroutines.channels.SendChannel<TensorResultMessage>? = null
) : TensorMessage()

/**
 * Message to perform matrix multiplication.
 */
data class MatmulTensorMessage(
    val a: EmberTensor,
    val b: EmberTensor,
    val responseChannel: kotlinx.coroutines.channels.SendChannel<TensorResultMessage>? = null
) : TensorMessage()

/**
 * Message to reshape a tensor.
 */
data class ReshapeTensorMessage(
    val tensor: EmberTensor,
    val newShape: ai.solace.emberml.tensor.common.EmberShape,
    val responseChannel: kotlinx.coroutines.channels.SendChannel<TensorResultMessage>? = null
) : TensorMessage()

/**
 * Result message containing a tensor operation result.
 */
data class TensorResultMessage(val result: EmberTensor) : ActorMessage

/**
 * Error message for tensor operations.
 */
data class TensorErrorMessage(val error: Throwable) : ActorMessage

/**
 * Actor for handling tensor operations in a non-blocking manner.
 */
class TensorActor : BaseActor<TensorMessage>() {
    
    override suspend fun receive(message: TensorMessage) {
        when (message) {
            is AddTensorMessage -> handleAdd(message)
            is MultiplyTensorMessage -> handleMultiply(message)
            is MatmulTensorMessage -> handleMatmul(message)
            is ReshapeTensorMessage -> handleReshape(message)
        }
    }
    
    private suspend fun handleAdd(message: AddTensorMessage) {
        try {
            val result = message.a + message.b
            message.responseChannel?.send(TensorResultMessage(result))
        } catch (e: Exception) {
            message.responseChannel?.send(TensorResultMessage(message.a)) // Fallback
        }
    }
    
    private suspend fun handleMultiply(message: MultiplyTensorMessage) {
        try {
            val result = message.a * message.b
            message.responseChannel?.send(TensorResultMessage(result))
        } catch (e: Exception) {
            message.responseChannel?.send(TensorResultMessage(message.a)) // Fallback
        }
    }
    
    private suspend fun handleMatmul(message: MatmulTensorMessage) {
        try {
            val result = message.a.matmul(message.b)
            message.responseChannel?.send(TensorResultMessage(result))
        } catch (e: Exception) {
            message.responseChannel?.send(TensorResultMessage(message.a)) // Fallback
        }
    }
    
    private suspend fun handleReshape(message: ReshapeTensorMessage) {
        try {
            val result = message.tensor.reshape(message.newShape)
            message.responseChannel?.send(TensorResultMessage(result as EmberTensor))
        } catch (e: Exception) {
            message.responseChannel?.send(TensorResultMessage(message.tensor)) // Fallback
        }
    }
}