# ChatBot models

the models are bult with sequential stacked Dense layes the number of layers and nodes in each of them are present here (there is a 50% dropout beetwin every layer)

xs1 = 32–16–n of tag

xs2 = 64–32–n of tag

s1 = 128–64–n of tag

s2 = 128–64–32–n of tag

s3 = 128–64–64–n of tag

s4 = 128–64–32–16–n of tag

s5 = 128–64–64–32–n of tag

m1 = 256–128–n of tag

m2 = 256–128–64–n of tag

m3 = 256–128–64–32-n of tag

l1 = 512–256–128-n of tag  

l2 = 512–256–128–64-n of tag  

l3 = 512–256–128–64–32-n of tag  

l4 = 512–256–128–64–32–16-n of tag  

l5 = 512–256–128–128–64-n of tag

xl1 = 1024–512–256-n of tag  

xl2 = 1024–512–256–128-n of tag  

xl3 = 1024–512–256–128–64-n of tag  

xl4 = 1024–512–256–128–64–32-n of tag

# ImageClassificator and BinaryImageClassificator models

these models contain 2d convolutional, 2d max pooling, flatting, dropout and fully connected dense layers

s1 : Conv2D(16 node)–MaxPooling2D–Conv2D(32 node)–MaxPooling2D–Conv2D(16 node)–MaxPooling2D–Flatten–Dense(256 node)–Dense(classification)

s1a : RandomFlip–RandomRotation–RandomZoom–Random–brightness–Conv2D(16 node)–MaxPooling2D–Conv2D(32 node)–MaxPooling2D–Conv2D(16 node)–MaxPooling2D–Flatten–Dense(256 node)–Dense(classification)

s2 : Conv2D(16 node)–MaxPooling2D–Conv2D(32 node)–MaxPooling2D–Conv2D(32 node)–MaxPooling2D–Flatten–Dense(256 node)–Dense(classification)

s3 : Conv2D(32 node)–MaxPooling2D–Conv2D(32 node)–MaxPooling2D–Conv2D(64 node)–MaxPooling2D–Dropout(40%)–Flatten–Dense(256 node)–Dense(classification)

m1 : Conv2D(32 node)–Conv2D(32 node)–MaxPooling2D–Dropout(25%)–Conv2D(64 node)–Conv2D(64 node)–MaxPooling2D–Dropout(25%)–Flatten–Dense(512 node)–Dropout(50%)–Dense(classification)

m2 : Conv2D(16 node)–Conv2D(32 node)–Conv2D(64 node)–MaxPooling2D–Dropout(25%)–Conv2D(128 node (1,1 kernal))–Conv2D(128 node)–Conv2D(64 node)–MaxPooling2D–Conv2D(32 node)–Dropout(25%)–Flatten–Dense(512 node)–Dropout(50%)–Dense(classification)

l1 : a vgg-19 clone (Conv2d(64 node)*2-MaxPooling2D-Conv2d(128 node)*2-MaxPooling2D-Conv2d(256 node)*4-MaxPooling2D-Conv2d(512 node)*4-MaxPooling2D-Conv2d(512 node)*4-MaxPooling2D-Dropout(0.1)-Flatten-Dense(4096)-Dence(4096)-Dense(classification))

l1.1 : an improved vgg-19(Conv2d(64 node)*2-MaxPooling2D-Conv2d(128 node)*2-MaxPooling2D-Conv2d(256 node)*4-MaxPooling2D-Conv2d(512 node)*4-MaxPooling2D-Conv2d(512 node)*4-Batchnormalization-globalAvaragepooling-Dense(2048)-Dropout(0.1)-Dense(2048)-Dense(classification))

** exept for the last fully connected dense layers other layers activision is relu (dense or conv)
** in ImageClassificator the classification layer have the same number of nodes as the number of classes with a softmax activation
** in BinaryImageClassificator the classification layer have one node with sigmoid activation