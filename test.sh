for f in test/bin/*.exe; do
    echo $f
    ./$f
done