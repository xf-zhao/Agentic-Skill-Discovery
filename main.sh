for i in $(seq 1 10);
do
    # python3 learn.py seed=0 model=gpt-3.5-turbo-0125 &>/dev/null &
    python3 learn.py seed=0 model=gpt-3.5-turbo-0125
    sleep 10
done