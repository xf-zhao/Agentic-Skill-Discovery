python propose.py

for i in $(seq 1 3);
do
    python learn.py seed=0 model=gpt-3.5-turbo-0125
    sleep 5
done