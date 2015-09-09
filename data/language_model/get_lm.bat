



echo "Downloading..."

set wget="../../3rdparty/bin/wget.exe"
set do_7za="../../3rdparty/bin/7za.exe"

%wget% --no-check-certificate russellsstewart.com/s/lm/vocab.pkl
%wget% --no-check-certificate russellsstewart.com/s/lm/train_indices.txt
%wget% --no-check-certificate russellsstewart.com/s/lm/valid_indices.txt
%wget% --no-check-certificate russellsstewart.com/s/lm/test_indices.txt

echo "Done."
