# Release Process for GA

## Gitflow

GA tries to follow the gitflow workflow.  A good tutorial on gitflow is [here](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).  Releases are simply tagged commits on the 'master' branch.  All development occurs in the 'develop' branch.  New features are developed in feature branches that started from a branch of 'develop'.  Complete features are merged back into develop.

## The Release Branch

Once enough new features exist, the develop branch is branched into a release branch.  No new features should be added at this point.  Once the release branch is ready, it gets merged into both 'master' and 'develop'.  The 'master' branch is tagged with the new release number.  A new tarball is created for the tag and added to the github release.   These steps are detailed next.

### Create the Branch

Starting with the develop branch, create a new release branch.
```
git checkout develop
git checkout -b release/0.1.0 # <-- use the intended version number
```

### Update the Version and CHANGELOG.md

The version number is in [global/src/gacommon.h](global/src/gacommon.h).  The [CHANGELOG.md](CHANGELOG.md) keeps track of user-visible changes.  Follow the formatting guidelines in the CHANGELOG.md file.

### Merge Back

You must merge the release branch back into develop and master.  If new features were added to develop, you might need to work through merge conflicts.  The merge into master should be a fast forward merge and without conflicts.
```
git checkout develop
git merge release/0.1.0
```
When merging master, you can also tag the release.  You must push the tag in addition to pushing the merged branch back to the origin.
```
git checkout master
git merge release/0.1.0
git push        # pushes the merged master branch to origin
git tag v0.1.0
git push --tags # pushes the tag to origin, creates a github draft release
```

### Finishing the GitHub Release

Pushing a tag creates a draft release.  You must edit the draft release to finish the process.  Select the tag you just created as the tag to build the release around.  Copy-and-paste the latest changes from the CHANGELOG.md file from the earlier release step to create the release notes.

GitHub automatically creates zip and tar.gz archives.  However, these do not contain the configure script.
1. Download the tar.gz archive.
2. Untar the archive.
3. Run autogen.sh inside the new directory.  This creates the missing generated files, e.g., configure, Makefile.in.
4. Remove the created autotools directory. We don't want to bundle the autotools with our releases.
5. Tar the archive back up.
6. Add the new archive as an artifact of the release.

### Miscellaneous

You need to update the Global Arrays web page at
[http://hpc.pnl.gov/globalarrays/](http://hpc.pnl.gov/globalarrays/). The files
for this page can be found at /msrc/webroot/hpc/globalarrays/ on the AFS
filesystem at PNNL. Relevant files for the current release are index.shtml and 
shared/rightnav.inc
