"""
    Remove empty directories (which contains (recursively) no files)
    (for use after move_tensor, move_measure)

    Based on https://gist.github.com/jacobtomlinson/9031697
"""

import os
import argparse

def removeEmptyFolders(path, removeRoot=True, verbose=True):
    """Function to remove empty folders"""
    if not os.path.isdir(path):
        print("Input path is not a dir: {}".format(path))
        return
    # remove empty subfolders
    files = os.listdir(path)
    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                removeEmptyFolders(fullpath)
    # if folder empty, delete it
    files = os.listdir(path)
    if len(files) == 0 and removeRoot:
        if verbose:
            print("Removing empty folder:", path)
        os.rmdir(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively remove empty folders")
    parser.add_argument("rootdir", type=str, 
        help="Root path of search domain")
    parser.add_argument("--keepRoot", action='store_true',
        help="If set, the root folder will be kept even if it contains no contents (recursively).")
    parser.add_argument("--quiet", action='store_true',
        help="If set, removed folder will not be printed. ")
    args = parser.parse_args()
    removeRoot = (not args.keepRoot)
    verbose = (not args.quiet)
    removeEmptyFolders(args.rootdir, removeRoot, verbose)
    