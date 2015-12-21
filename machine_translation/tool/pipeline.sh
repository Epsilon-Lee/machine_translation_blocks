preprocess=../tool/preprocess.py
invert=../tool/invert-dict.py
convert=../tool/convert-pkl2hdf5.py
shuffle=../tool/shuffle-hdf5.py
$preprocess -d vocab.$2.pkl -v 30000 -b binarized_text.$2.pkl -p bitext.$2.tok.txt
$preprocess -d vocab.$1.pkl -v 30000 -b binarized_text.$1.pkl -p bitext.$1.tok.txt
python $invert vocab.$2.pkl ivocab.$2.pkl
python $invert vocab.$1.pkl ivocab.$1.pkl
$convert binarized_text.$2.pkl binarized_text.$2.h5
$convert binarized_text.$1.pkl binarized_text.$1.h5
$shuffle binarized_text.$2.h5 binarized_text.$1.h5 binarized_text.$2.shuf.h5 binarized_text.$1.shuf.h5
