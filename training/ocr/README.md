# Custom CTC Loss
CTC Loss but in addition to a classification label per time step we have an additional set of real values as targets (assume Gaussian distributed) per time step and a new BLANK1 symbol for masking real value targets.
# How this is used in OCR?
We use 6 real values to predict color of font and border color per character, we use space character as BLANK1 symbol.
# Acknowledgement
Original code is from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/LossCTC.cu
# Reference
```
[1] Graves, A., Fernández, S., Gomez, F. and Schmidhuber, J., 2006, June. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In Proceedings of the 23rd international conference on Machine learning (pp. 369-376).
[2] Wigington, C., Price, B. and Cohen, S., 2019, September. Multi-label connectionist temporal classification. In 2019 International Conference on Document Analysis and Recognition (ICDAR) (pp. 979-986). IEEE.
```
