nlpcaffe(https://github.com/Russell91/nlpcaffe) on windows
======


how to install
======
1. Follow the instructions here(https://initialneil.wordpress.com/2015/01/11/build-caffe-in-windows-with-visual-studio-2013-cuda-6-5-opencv-2-4-9/)
ps: use source code nlpcaffe instead of caffe, in this article.

2. After training MNIST dataset successfully, let's train language model.
	a. run scripts/GeneratePB.bat to obtain caffe_pb2.py
	b. run data/language_model/get_lm.bat to download dataset.
	c. generate architecture train_val.prototxt with: 
		python ./examples/language_model/create_lm.py --make_data
	d. run convert_language_data.cpp to obtain leveldb data
		(because it seems that there are still some issures with lmdb on windows)
	e. change backend of train_val.prototxt to LEVELDB
	f. train the network



