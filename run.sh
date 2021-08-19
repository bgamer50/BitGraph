echo "0" >> benchmark_results.txt
./components_gpu.exe data/twitter.txt 0 2>> benchmark_results.txt

echo "1" >> benchmark_results.txt
./components_gpu.exe data/twitter.txt 1 2>> benchmark_results.txt

echo "2" >> benchmark_results.txt
./components_gpu.exe data/twitter.txt 2 2>> benchmark_results.txt

echo "3" >> benchmark_results.txt
./components_gpu.exe data/twitter.txt 3 2>> benchmark_results.txt

echo "4" >> benchmark_results.txt
./components_gpu.exe data/twitter.txt 4 2>> benchmark_results.txt

echo "5" >> benchmark_results.txt
./components_gpu.exe data/twitter.txt 5 2>> benchmark_results.txt

echo "6" >> benchmark_results.txt
./components_gpu.exe data/twitter.txt 6 2>> benchmark_results.txt
echo "done"
