docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" -v ~/workspace/isaac/orbit:/data/orbit:rw 134.100.39.10:32000/isaacsim:2023.1.1
