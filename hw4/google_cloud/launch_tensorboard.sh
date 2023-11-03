#! /bin/bash

# Launches tensorboard on the remote instance, forwarding the port to your local machine.

export ZONE="northamerica-northeast1-c"
export INSTANCE_NAME="instance-1"

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='sudo pkill -f tensorboard'
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='bash -lc "tensorboard --logdir ~/cs285_hw1_dagger/hw4/data --port 6006"' --ssh-flag="-L 6006:localhost:6006"
