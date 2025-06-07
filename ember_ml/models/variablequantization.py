import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
from ember_ml.nn import tensor
from ember_ml.ops import set_backend
from ember_ml.nn.tensor.types import TensorLike
from ember_ml.ops import stats
from ember_ml import ops

@dataclass
class BlockHash:
    block_id: int
    original_hash: str
    quantized_hash: str
    timestamp: float = time.time()

@dataclass
class BlockResult:
    block_id: int
    quantized_data: TensorLike
    original_data: TensorLike
    hash_info: BlockHash
    processing_time_ms: float
    compression_ratio: float
    mse: float

class HashVerifier:
    def __init__(self, verification_precision: int = 10):
        self.verification_precision = verification_precision
    
    def compute_original_block_hash(self, block_id: int, data: TensorLike) -> str:
        """Compute deterministic hash for original float data"""
        # Round the data to specified precision and create string representation
        rounded_data = [round(float(x), self.verification_precision) for x in data]
        data_str = str(rounded_data)  # Uses Python's list string representation
        return self.sha256(data_str)
    
    def compute_quantized_block_hash(self, block_id: int, data: TensorLike) -> str:
        """Compute deterministic hash for quantized data"""
        # Convert quantized data to list and create string representation
        data_list = data.tolist()  # Convert numpy array to list
        return self.sha256(str(data_list))
    
    def verify_block(self, block_id: int, original: TensorLike, quantized: TensorLike) -> BlockHash:
        """Verify quantization result against original data"""
        original_hash = self.compute_original_block_hash(block_id, original)
        quantized_hash = self.compute_quantized_block_hash(block_id, quantized)
        
        return BlockHash(
            block_id=block_id,
            original_hash=original_hash,
            quantized_hash=quantized_hash
        )
    
    @staticmethod
    def sha256(input_str: str) -> str:
        """Compute SHA-256 hash and return first 8 characters"""
        hash_obj = hashlib.sha256(input_str.encode())
        return hash_obj.hexdigest()[:8]

class VerifiedQuantizer:
    def __init__(self, block_size: int = 4096, verification_precision: int = 10):
        self.block_size = block_size
        self.verifier = HashVerifier(verification_precision)
    
    def quantize_block(self, block_id: int, data: TensorLike) -> BlockResult:
        """Quantize a block of data with verification"""
        start_time = time.time()
        
        # Convert input to numpy array if it isn't already
        data = tensor.asarray(data)
        
        # Compute scale based on absolute maximum
        abs_max = stats.max(ops.abs(data)) if len(data) > 0 else 1.0
        
        # Perform quantization
        quantized = self.quantize_8bit(data)
        
        # Compute hashes and metrics
        hash_info = self.verifier.verify_block(block_id, data, quantized)
        compression_ratio, mse = self.compute_metrics(data, quantized, abs_max)
        
        return BlockResult(
            block_id=block_id,
            quantized_data=quantized,
            original_data=data,
            hash_info=hash_info,
            processing_time_ms=(time.time() - start_time) * 1000,
            compression_ratio=compression_ratio,
            mse=mse
        )
    
    def quantize_all_blocks(self, data: TensorLike) -> List[BlockResult]:
        """Quantize all data in blocks"""
        from ember_ml import ops
        data = tensor.convert_to_tensor(data)
        blocks = tensor.convert_to_tensor(data, ops.ceil(len(data) / self.block_size))
        return [self.quantize_block(i, block) for i, block in enumerate(blocks)]
    
    def compute_metrics(self, original: TensorLike, quantized: TensorLike, abs_max: float) -> Tuple[float, float]:
        """Calculate metrics for quantization results"""
        # Compression ratio (32-bit float to 8-bit int)
        compression_ratio = 4.0
        
        # Mean squared error using the same scale as quantization
        scale = abs_max / 127.0
        dequantized = (quantized.astype(float) - 128) * scale
        mse = stats.mean((original - dequantized) ** 2)
        
        return compression_ratio, float(mse)
    
    def quantize_8bit(self, data: TensorLike) -> TensorLike:
        """Quantize float data to 8 bits"""
        abs_max = stats.max(ops.abs(data)) if len(data) > 0 else 1.0
        scale = abs_max / 127.0
        
        # Scale to [-127, 127] range and shift to [0, 255]
        scaled = ops.clip(data / scale, -127, 127)
        return ops.round(scaled + 128).astype(tensor.uint8)

def test_quantization():
    # Generate test data
    data = tensor.random_normal(0, 1, 1000)
    
    # Create quantizer
    quantizer = VerifiedQuantizer(block_size=128)
    
    # Quantize data
    results = quantizer.quantize_all_blocks(data)
    
    print(f"\nProcessed {len(results)} blocks:")
    for block in results[:3]:  # Show first 3 blocks
        print(f"\nBlock {block.block_id}:")
        print(f"Processing Time: {block.processing_time_ms:.2f}ms")
        print(f"Compression Ratio: {block.compression_ratio:.2f}x")
        print(f"Mean Squared Error: {block.mse:.6f}")
        print(f"Original Hash: {block.hash_info.original_hash}")
        print(f"Quantized Hash: {block.hash_info.quantized_hash}")
        print(f"First 5 quantized values: {block.quantized_data[:5]}")
        
        # Verify hash consistency
        verifier = HashVerifier()
        verify_orig = verifier.compute_original_block_hash(block.block_id, block.original_data)
        verify_quant = verifier.compute_quantized_block_hash(block.block_id, block.quantized_data)
        
        print(f"Verification OK: {verify_orig == block.hash_info.original_hash}")

if __name__ == "__main__":
    test_quantization()