# Documentation Cleanup Plan

This document identifies which architecture planning documents should be kept as part of the final documentation and which ones can be deleted as they were just sketches or have been superseded by more comprehensive documents.

## Files to Keep (Final Documentation)

1. **`cuda_kernel_insights.md`**
   - Contains detailed insights from the CUDA kernel implementation in hyena-dna
   - Provides valuable information for backend-specific optimizations

2. **`ember_ml_architecture_summary.md`**
   - Concise summary of the key insights and architecture decisions
   - Serves as a quick reference for the overall architecture

3. **`ember_ml_comprehensive_final_architecture.md`**
   - The comprehensive final architecture plan
   - Incorporates insights from all sources and provides a complete description of the architecture

4. **`ember_ml_implementation_roadmap.md`**
   - Step-by-step roadmap for implementing the architecture
   - Provides a clear timeline and task breakdown for implementation

5. **`fftconv_insights.md`**
   - Detailed analysis of the FFT convolution implementation in hyena-dna
   - Provides valuable information for implementing the FFT convolution sequence mixer

6. **`index.md`**
   - Index document listing all the architecture planning documents
   - Serves as a central reference point for the documentation

7. **`liquid_ai_additional_insights.md`**
   - Additional insights from our interaction with Liquid AI
   - Provides valuable information about self-training and self-tuning approaches

8. **`rbm_save_load_improvement.md`**
   - Plan for improving the save/load functionality for RBM modules
   - Addresses a specific issue with the current implementation

9. **`README.md`**
   - General information about the plans directory
   - Provides context for the documentation

## Files to Delete (Sketches or Superseded)

1. **`ember_ml_architecture_reorganization_updated.md`**
   - Superseded by the comprehensive final architecture
   - Contains earlier ideas that have been refined in later documents

2. **`ember_ml_architecture_reorganization.md`**
   - Superseded by the updated version and the comprehensive final architecture
   - Contains earlier ideas that have been refined in later documents

3. **`ember_ml_architecture_updates.md`**
   - Incorporated into the comprehensive final architecture
   - Contains updates that are now part of the final architecture

4. **`ember_ml_combined_architecture.md`**
   - Superseded by the comprehensive architecture
   - Contains earlier ideas that have been refined in later documents

5. **`ember_ml_comprehensive_architecture.md`**
   - Superseded by the comprehensive final architecture
   - Contains earlier ideas that have been refined in later documents

6. **`ember_ml_final_architecture_plan.md`**
   - Superseded by the comprehensive final architecture
   - Contains earlier ideas that have been refined in later documents

7. **`ember_ml_final_architecture_with_moe.md`**
   - Incorporated into the comprehensive final architecture
   - Contains MoE-specific details that are now part of the final architecture

8. **`ember_ml_final_architecture.md`**
   - Superseded by the comprehensive final architecture
   - Contains earlier ideas that have been refined in later documents

9. **`ember_ml_mad_inspired_architecture.md`**
   - Early sketch incorporated into later documents
   - Contains MAD-Lab-inspired ideas that are now part of the final architecture

10. **`ember_tensor_frontend_backend_separation.md`**
    - Incorporated into the comprehensive final architecture
    - Contains tensor-specific details that are now part of the final architecture

11. **`follow_up_items.md`**
    - Likely superseded by the implementation roadmap
    - Contains follow-up items that are now addressed in the roadmap

12. **`gemini_plan.md`**
    - Likely an early planning document
    - Contains ideas that are now incorporated into the final architecture

13. **`liquid_ai_architecture_insights.md`**
    - Superseded by the additional insights document
    - Contains earlier insights that have been refined in later documents

## Recommendation

1. **Backup**: Before deleting any files, create a backup of the entire `docs/plans` directory to ensure no information is lost.

2. **Update Index**: Update the `index.md` file to reference only the files that are being kept.

3. **Delete**: Delete the files identified for deletion to avoid confusion and maintain a clean documentation structure.

4. **Archive**: Consider moving the deleted files to an archive directory (e.g., `docs/plans/archive`) instead of permanently deleting them, in case they contain any unique insights that might be valuable in the future.

## Next Steps

1. Review this cleanup plan and confirm which files should be kept and which should be deleted.
2. Implement the cleanup according to the recommendation.
3. Proceed with the implementation of the Ember ML architecture according to the roadmap.