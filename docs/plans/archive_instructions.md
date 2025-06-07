# Instructions for Archiving Old Documentation Files

This document provides instructions for archiving old documentation files that have been identified for deletion in the [Documentation Cleanup Plan](documentation_cleanup.md).

## Step 1: Create an Archive Directory

First, create an archive directory to store the old documentation files:

```bash
mkdir -p docs/plans/archive
```

## Step 2: Move Files to Archive

Move the following files to the archive directory:

```bash
# Files to move to archive
mv docs/plans/ember_ml_architecture_reorganization_updated.md docs/plans/archive/
mv docs/plans/ember_ml_architecture_reorganization.md docs/plans/archive/
mv docs/plans/ember_ml_architecture_updates.md docs/plans/archive/
mv docs/plans/ember_ml_combined_architecture.md docs/plans/archive/
mv docs/plans/ember_ml_comprehensive_architecture.md docs/plans/archive/
mv docs/plans/ember_ml_final_architecture_plan.md docs/plans/archive/
mv docs/plans/ember_ml_final_architecture_with_moe.md docs/plans/archive/
mv docs/plans/ember_ml_final_architecture.md docs/plans/archive/
mv docs/plans/ember_ml_mad_inspired_architecture.md docs/plans/archive/
mv docs/plans/ember_tensor_frontend_backend_separation.md docs/plans/archive/
mv docs/plans/follow_up_items.md docs/plans/archive/
mv docs/plans/gemini_plan.md docs/plans/archive/
mv docs/plans/liquid_ai_architecture_insights.md docs/plans/archive/
```

## Step 3: Create an Archive Index

Create an index file in the archive directory to document the archived files:

```bash
cat > docs/plans/archive/index.md << 'EOF'
# Archived Documentation Files

This directory contains archived documentation files that were part of the early planning process for the Ember ML architecture. These files have been superseded by more comprehensive documents in the main plans directory.

## Archived Files

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
    - Superseded by the implementation roadmap
    - Contains follow-up items that are now addressed in the roadmap

12. **`gemini_plan.md`**
    - Early planning document
    - Contains ideas that are now incorporated into the final architecture

13. **`liquid_ai_architecture_insights.md`**
    - Superseded by the additional insights document
    - Contains earlier insights that have been refined in later documents

## Note

These files are kept for historical reference only and should not be used as part of the current architecture documentation.
EOF
```

## Step 4: Verify the Archive

Verify that all files have been moved to the archive directory and that the index file has been created:

```bash
ls -la docs/plans/archive/
```

## Step 5: Update the Main Index

Ensure that the main index file (`docs/plans/index.md`) has been updated to reference only the files that are being kept, as done in the previous step.

## Step 6: Commit Changes

If using version control, commit the changes:

```bash
git add docs/plans/archive/
git add docs/plans/index.md
git commit -m "Archive old documentation files and update index"
```

## Conclusion

By following these steps, you will have successfully archived the old documentation files while maintaining a clean and organized documentation structure. The archived files are still available for reference if needed, but they are clearly separated from the current documentation.