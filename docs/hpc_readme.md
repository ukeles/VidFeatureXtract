# Getting Started with Caltech's Resnick HPC Cluster

*This guide is intended for members of the Adolphs Lab at Caltech. Please ensure you follow all relevant policies and guidelines while utilizing the Resnick HPC Cluster.*

## Accessing the Cluster
To connect to the Resnick HPC Cluster, use a terminal for SSH connection. Replace `username` with your actual Caltech username. Follow the prompts for Duo Push authentication:
```bash
ssh username@login.hpc.caltech.edu
```
**Note**: Connection requires being on the Caltech campus network or connected to Caltech's VPN service. [Here's how to connect to Caltech's VPN](https://www.imss.caltech.edu/services/wired-wireless-remote-access/Virtual-Private-Network-VPN).

## Installing Miniconda [optional / required once]
Miniconda is a minimal installer for Anaconda, useful for running the feature extraction codes in this repository. If you have not already installed Miniconda into your home or group directory, follow these steps to install it:
1. Download the Miniconda installer:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
2. Run the installer:
```bash
sh ./Miniconda3-latest-Linux-x86_64.sh
```
After installation, follow the on-screen instructions to initialize and configure Miniconda. For detailed information on software installation on the cluster, refer to the [Caltech HPC Center's software and modules documentation](https://www.hpc.caltech.edu/documentation/software-and-modules).

## Installing the *vidfeats* Repository for Feature Extraction
To install the `vidfeats` repository for extracting various features from video files, please refer to the installation instructions provided in the repository's [README](../README.md#installation) file.

## Running Jobs
Provide instructions on how to run jobs on the cluster.

## Additional Resources
For a general instruction to using SLURM commands at the HPC cluster, see [the official documentation](https://www.hpc.caltech.edu/documentation/slurm-commands) and [the script generator](https://s3-us-west-2.amazonaws.com/imss-hpc/index.html).



