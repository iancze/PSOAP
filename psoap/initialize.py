import sys
import pkg_resources
import psoap


def main():
    """
    Available from command line as ``psoap-initialize``

    """

    import argparse

    parser = argparse.ArgumentParser(description="Initialize a new directory to do inference.")
    parser.add_argument("--check", action="store_true", help="To help folks check whether the package was installed properly.")
    parser.add_argument("--model", choices=["SB1", "SB2", "ST3"], help="Which type of model to use, SB1, SB2, ST1, or SB3.")

    args = parser.parse_args()

    if args.check:

        print("PSOAP successfully installed and linked.")
        print("Using Python Version", sys.version)
        sys.exit()

    else:
        # Initialize the directory based upon the chosen model
        import shutil
        import psoap

        # Copy over the appropriate config.yaml file to current working directory
        masks = pkg_resources.resource_filename("psoap", "data/masks.dat")
        chunks = pkg_resources.resource_filename("psoap", "data/chunks.dat")
        config = pkg_resources.resource_filename("psoap", "data/config.{}.yaml".format(args.model))

        shutil.copy(masks, "masks.dat")
        shutil.copy(chunks, "chunks.dat")
        shutil.copy(config, "config.yaml")

        # import os
        # import inspect
        # basedir = os.path.dirname(inspect.getfile(psoap))
        # shutil.copy(basedir + "/../data/config.{}.yaml".format(args.model), "config.yaml")
        # shutil.copy(basedir + "/../data/chunks.dat", "chunks.dat")
        # shutil.copy(basedir + "/../data/masks.dat", "masks.dat")
        print("Copied config file for {} model to current working directory as config.yaml".format(args.model))
        print("Copied chunks.dat and masks.dat to current working directory.")

if __name__=="__main__":
    main()
