# python baselines/run_benchmark.py --model anytext --benchmark CVTG-2K
# python baselines/run_benchmark.py --model qwenedit --benchmark LongText-Bench
# python baselines/run_benchmark.py --model fluxfill --benchmark CVTG-2K
# python baselines/run_benchmark.py --model textcrafter_flux --benchmark LongText-Bench
python baselines/run_parallel_benchmark.py --model qwenimage --benchmark LongText-Bench
