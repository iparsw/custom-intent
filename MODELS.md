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

# BinaryImageClassificator models

these models contain 2d convolutional, 2d max pooling, flatting, dropout and fully connected dense layers

s1 : Conv2D(16 node)–MaxPooling2D–Conv2D(32 node)–MaxPooling2D–Conv2D(16 node)–MaxPooling2D–Flatten–Dense(256 node)–Dense(one node for classification)

s1a : RandomFlip–RandomRotation–RandomZoom–Random–brightness–Conv2D(16 node)–MaxPooling2D–Conv2D(32 node)–MaxPooling2D–Conv2D(16 node)–MaxPooling2D–Flatten–Dense(256 node)–Dense(one node for classification)

s2 : Conv2D(16 node)–MaxPooling2D–Conv2D(32 node)–MaxPooling2D–Conv2D(32 node)–MaxPooling2D–Flatten–Dense(256 node)–Dense(one node for classification)

s3 : Conv2D(32 node)–MaxPooling2D–Conv2D(32 node)–MaxPooling2D–Conv2D(64 node)–MaxPooling2D–Dropout(40%)–Flatten–Dense(256 node)–Dense(one node for classification)

m1 : Conv2D(32 node)–Conv2D(32 node)–MaxPooling2D–Dropout(25%)–Conv2D(64 node)–Conv2D(64 node)–MaxPooling2D–Dropout(25%)–Flatten–Dense(512 node)–Dropout(50%)–Dense(one node for classification)

** exept for the last fully connected dense layers other layers activision is relu (dense or conv)