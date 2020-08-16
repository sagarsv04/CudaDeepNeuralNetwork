url="http://yann.lecun.com/exdb/mnist/"
dir="./data/"

mkdir ${dir}

for file in "train-images-idx3-ubyte.gz" "train-labels-idx1-ubyte.gz" "t10k-images-idx3-ubyte.gz" "t10k-labels-idx1-ubyte.gz"
do
		echo "${url}${file}"
		curl "${url}${file}" -o "${dir}${file}"
		gzip -d "${dir}${file}"
done
