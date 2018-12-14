pushd ../gremlin++
make clean
make
popd
make clean
make
make test.exe
echo "done"

