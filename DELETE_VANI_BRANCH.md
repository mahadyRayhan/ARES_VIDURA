# How to Delete vani-branch

This guide explains how to delete the `vani-branch` from the repository.

## Overview

The `vani-branch` exists as a remote branch on the GitHub repository. To completely remove it, you'll need to delete it from the remote repository.

## Prerequisites

- You must have write access to the repository
- You should be on a different branch (not on `vani-branch`)

## Steps to Delete the Branch

### Option 1: Delete Remote Branch Using Git Command

If you have git configured with the repository:

```bash
# Delete the remote branch
git push origin --delete vani-branch
```

### Option 2: Delete via GitHub Web Interface

1. Go to the repository on GitHub: https://github.com/mahadyRayhan/ARES_VIDURA
2. Click on the "branches" link (usually shows "X branches")
3. Find `vani-branch` in the list
4. Click the trash/delete icon next to the branch name
5. Confirm the deletion

### Option 3: Using GitHub CLI

If you have the GitHub CLI (`gh`) installed:

```bash
# Delete the remote branch
gh api repos/mahadyRayhan/ARES_VIDURA/git/refs/heads/vani-branch -X DELETE
```

## Verification

After deletion, you can verify the branch is gone by running:

```bash
# List all remote branches
git ls-remote --heads origin

# Or check on GitHub
# The branch should no longer appear in the branches list
```

## Important Notes

- **This action cannot be undone easily**. Make sure you want to delete the branch before proceeding.
- If the branch contains important work, consider merging it into another branch first.
- Deleting a remote branch does not automatically delete local copies that others may have. Team members will need to clean up their local copies using `git remote prune origin` or `git fetch --prune`.

## Cleaning Up Local References (For Team Members)

After the remote branch is deleted, team members should clean up their local references:

```bash
# Remove stale remote-tracking branches
git remote prune origin

# Or use fetch with prune
git fetch --prune

# To also delete a local copy of the branch (if it exists)
git branch -d vani-branch
```

## Contact

If you encounter any issues or don't have the necessary permissions, please contact the repository administrator.
