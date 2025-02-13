# English_Korean_translation_transformer
A  implementation based on transformer for translation from English to Korean

1: The implementation based on a fantastic tutorial, https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers.git

   If you are interested in pytorch and want to improve both in theory and experence, https://github.com/sgrvinod/Deep-Tutorials-for-PyTorch.git is recommended
   
   The data used in the project comes from https://github.com/Huffon/pytorch-transformer-kor-eng.git

   
2: after you clone the project, you can just enjoy it by 
   
   down load the data called "corpus.csv", you can follow the help in prepare_data.py,
   
   python prepare_data.py, then:
   
   python train.py
   
   when you get trained model, you can use:
   
   python translate.py after modifing the path simply

   
3: NOTE: You may Curious about why I set the train:val:test = 0.2, 0.3, 0.5. Cuz there may be a mistake in divied data sets. The corpus.csv
   contains two cols named "korean" and "english". They are one-to-one correspondence, so it is feel like not difficult to spilt the data to train, val, test
   data in each language. But after perpare_data, I find the total english contents is more than korean contents. Most of data after spilt can be
   seemed as en_kr pairs that can translate correctly except two rows consealed in the huge of data. 
   
   So, I make train and val sets as possible as small to avoid the mistake appear to these sets to protect the training process
   
   If you see "There are a different number of source or target sequences !!!!!STOP STOP STOP!!!!! please check the data's length and divide data again", 
   take it easy, redo "python prepare_data.py"
  
   the eval.py may not avaliable cuz the test set which cantains mistake when the training process is normal will be used in eval.
   
   After training, you can use translate.py to translate all you want, just make sense~



