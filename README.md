
### Vehicle detector and tracker

YOLOv3 + simple centroid tracker

![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/27694775/111885964-f2d7bc80-89db-11eb-954c-e72c6fb6a7c0.gif)

### Light detector
+ Convert image to YCrCb
+ Morphological operations on Cr channel
+ Get pair with bigger areas 

![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/27694775/111885671-5103a000-89da-11eb-80ad-e6875c2ac145.gif)

### Status analyser
+ Save mean of lightness in detected areas for every detected car
+ Compare current lightness with values from past (0.3 sec ago)

![ezgif com-gif-maker (2)](https://user-images.githubusercontent.com/27694775/111886169-43034e80-89dd-11eb-9a47-1411c00a485f.gif)

Video sources:
+ https://youtu.be/BqY9JZ3O7e8
+ https://youtu.be/vffgKEGG-8Y
