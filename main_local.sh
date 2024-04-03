for i in $(seq 1 10);
do
    python3 learn.py seed=0 model=gpt-3.5-turbo-0125 memory_requirement=8 min_gpu_mem=8
    sleep 10
done