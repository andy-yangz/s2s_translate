## Sequence to Sequence translator pytorch implementation

This is an implementation of sequence to sequence translator by pytorch.

![Image result for sequence to sequence translation](https://www.tensorflow.org/images/basic_seq2seq.png)

Currently using language is from English  to Germany.

Done.

1. An baseline RNN encoder decoder translator.
2. Use pretrained embedding.
3. Use Bidirection encoder get better context vector. Then pass forward direction hidden state to decoder.

#### Language Pairs Source from

http://www.manythings.org/anki/

#### Multilinguage Embedding:

[Polyglot Project 64d embedding](https://sites.google.com/site/rmyeid/projects/polyglot)

#### To-do 

 	1. Try to translate from English to Chinese.
 	2. Try autoencoder, then only take encoder as the encoder of translator.

About the language, at first try to reimplement system using German to English.
Then try to change it to Chinese to English.



