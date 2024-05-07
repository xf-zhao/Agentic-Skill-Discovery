for i in $(seq 1 5);
do
    # python3 learn.py task=null precedents=null seed=0 &>/dev/null &
    python3 learn.py task=null precedents=null seed=0
    sleep 10
done