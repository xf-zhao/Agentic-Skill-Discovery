for i in $(seq 1 2);
do
    python3 learn.py task=null precedents=null seed=0 &>/dev/null &
    # python3 learn.py seed=0 model=gpt-3.5-turbo-0125
    sleep 10
done