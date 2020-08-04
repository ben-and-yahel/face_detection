
# Welcome to our face detection project!

Hi! this was our biggest project yet and it include lodes of knowlage and hard work.
we made this github project to share with you the things we learn from it.
hope you will expand your knowloage about machine learning, AI and image filtering.
# what is it?
*![What is bounding box in image processing? - Quora](https://lh3.googleusercontent.com/Ug9cC9AEJxxnLRlwU9sKDQwPYxwkf40H1ksJ6EDt745XZfBd3SWuI88t_BMtMcF-JvM4IwaLquXmhGAG_QRcgoCW7e94P3v3SLhyw2bGwueDY5WebCqWX5gYi3Y5-1oHmO1ggzzL59Q)


Well basicly our code takes an image and detect where the face are in the image and who is it.
 - classifiction \bounding box- we uses svm, random forest and machine learnin for knowing where the face is.
 - clustering - the way messuring something and tell how similar it to another thing.
## Prerequisites
 - tensorflow 1.14
 - numpy
 - skimage
 - matplotlib
 - dlib
 - sklearn
 - firebase_admin
 

## introduction to AI
AI is the future, you can see it almost in everything: in your phone, TV, spotify and more...
in our project we develop a brain\mind who got a self thinking and can tell us wether an image contian face or not. 

# Stage 1.0 - Bounding Box Classifiar

we would like to make a brain\mind who can look at image and tell what is it just backgroud and what is it face.
## SVM
svm - Support Vector Machine 
its a method that using supervised learning.
It means that in this method we will be using learning in positive\negative additude.
we will give our machine 2 arrays,** one** is positive and contains faces we would like to allocate and the **second** contains negative images of grass, wall, skyes exd..
as we can see in pic1 we got 3 lines that the machine set about the data:
the **green** line is bad, the **blue** line is good but not accurate and the **red** line is the best line.

## Image filtering
Face is a complicate varient and we cannot just pass the pixels input of our image to the machine because it way to big and dosen't much.
to slove this problam we will use HOG or Histogram of Orriented Grandients.
basicly it messures the difference between point 9 pixels and set an arrow that point to biggest difference.

![](https://lh5.googleusercontent.com/FdJjher-NuALheOVJtohgLAq5dT5bMQcyC9kU4UlKzHatOy3lxF5QzYXCkDGGRw-HyHWpGtP0EiPfTFWVzb2ufgTijr1PmzMyJPWUStiTsLFJP97qryeju63c6np2F83KbV6EklCMAk)

## Data
svm require a lot of data to work poparly,
in our project we uses lfw for the faces data and skimage.data libary for the negative data
lfw - [https://scikit-image.org/docs/0.14.x/api/skimage.data.html#skimage.data.lfw_subset](https://scikit-image.org/docs/0.14.x/api/skimage.data.html#skimage.data.lfw_subset)

skimage.data - **[https://scikit-image.org/docs/dev/api/skimage.data.html](https://scikit-image.org/docs/dev/api/skimage.data.html)**

# Stage 1.5  - train
In this project we used python to power because it has so much support on AI and it very easy to use.
after seting up the code and training our svm moder we ended up with this:

**![](https://lh4.googleusercontent.com/K6S7j7wWKfxYtQoQ_1PHWgArS-DnFd-DrfTiynIqEAJKPm6UxftijtAx_i8QMkpubAyFHCSTQVgltLRyCgGZBbYnZa2HYCGClQuGyUG4nOIiXw-U2p-VxDCV1DYJgL7iFjJ2s8a4tfI)**

yay.

## The work behind AI
In our first time using Ai model we encountered in the hardes part of Ai which is **parameter tuning**
Now the creativity had to be shown and we have to work harder to get what we want, whitch is a fast and accurate machine.
so after trying to make our machine smarter we doubled the negative data and train it more times
and we ended up with this:
**image**

yay?
# Concloutin and a new way
our vision for the end of this task is the same like the privous one "upgrading the model" but it complicty diferent. our targets is very clear, we would to make our moder better on the staf below:
1. our model runs very slow, we would like to get to state  when clien would get a imidient response from the server.
2. The model is still not accurate as we want we have upgrade it but not enough.
3. the bounding boxes we get repersent lot of information and we need to reduced it to ome bounding box.
## stage 1.5.1 - NMS(Non Maximum Supperssion)
A quick way to take a lot of data and reduced it by avvarging all the variabels we need.
## stage 1.5.2 - Random Tree Forest
### choices trees:
To get a better understanding what is a Random Tree Forest we first need to understand what is a **Decision Tree Classifier**
choices trees is prrety old fashoin way but it worked very well for us in this project.
decition tree get a data and trys to classify it by sorting the data to questions.
ex:
- we got a large data of people on different age, location and money.
- i would like to build a model that when I enter age, location of someone i will know whether he above the The poverty line.
- The tree will build that way:
1. is man?
2. is live in the south?
3. is age above 50?
4. is live in israel?
- when i put a new man into the data the model will ask him all this questuins and i will get my result(like a real interview!).
- in the img we can see a dection tree that decids whether a man is fit or not.
- afcors that all this questions the model decideds by himself and when it became for pixels in a face it gets more complicate.

**![Decision Tree Classification - Towards Data Science](https://lh6.googleusercontent.com/uGT90mGTBNEy_z64sYR1VXEHhwvepYAbIuQaQodVcxVicHr7LFFAgo2SPBPZEOv67UPGov6ibqxdl74KSmsAqRaggYwkD5R-8X8JFIweGffnaDv4bd-AUKu4lzMccmYr9E8AJjfcwjI)**

**Random Tree Forest**
okay... we get what destion tree means but what "forest" means?
forest is a model that make use of several tree that have been trained on different parts of the data.
this model give us more accuracy because there is more outcomes from the trees and we can valid our output from differnet angel.
in the img below we can see the **blue** tree is regular tree and it has been trained on all the 4 parts of our data. On the other hand the **green** trees are a forest and they have been trained on a differnt data.

**![Understanding Random Forest - Towards Data Science](https://lh3.googleusercontent.com/0FWIRKkEdtC_r5EgmmiFx_YPxiatm2nIKfLKVsv63hqnYX_L_PlRRsp1TvBxQxAVUBRd8GwV0Pcdidtggr2_plVo6U6bfswh9-WvpV1WdnoSDP_hdV4vpfICOeXBAzLebhq-EDJ_u-_jvoGHog)**

In the image below we can a forest that got some dectition based one 6 trees that said 1 while 3 trees said 0.
which meens our answer in 1.

**![Understanding Random Forest - Towards Data Science](https://lh4.googleusercontent.com/IEBQQS8WE1FmhS8wtY3ao7nl1uIM_DIH26_uvc33Fd-aSv1h9oRSQERAbqrHfUsMQ-dcs8KchycZ_lNVy5yeTVEYX5Zfb70DFXgVeI7ROkLIWqXwNEcjC8ysKXMgD2JmGBFJCiR08fTplK26yQ)**

**coclution**
So why tree forest is doing such a good job for us? 
The reason for that is because it filters automaticly all the inputs. It means that until now the SVM model was checking every part in the image to undersatnd if its face or not, but tree works in a differnt way.
desition tree checks spesific parameters and only then coutinus to another check which means if we have part in image which is compelecy white or black the tree will imidiatly decides that its not face while SVM will think a lot more time.
as we can see in the image below SVM classify every image while Tree decides which dump imidiatly.

**![](https://lh6.googleusercontent.com/IIyONkam0tl5C1ekExgdShfSVqHXM4N1knXUbHiNVSJ0oUcm6TCPFXviSCgVSvc13wcHzRqroUL8DyNDfmjOAC__BQnwHTAI3REhwMs7w-d7Blc1ZkTiMFpoIaDamvm6asEILrVf_q7Jp28mZw)**

## stage 1.5.3 - retrain model
- To make our model more accurate we use in the method of **retrain model**
- Our model was traind on faces date that was 62 by 47 px, which means only faces in this size will absorb(we will fix that problam in the future) which means that if we run our - model on 200 by 200 image and all the image is face our shouldent find  anything.
- that wasent the case :(
- as you can see in the image below our model classify the eyes and mause as face, we can understand why because eyes and mause are very similar to a face although they don't.
- What wev'e done is taking 10,000 of this "mistakes" and train our model in addition to the data we already have.
- In the image below we can see the output  of **stage 1.5**, weve got one bounding box and its fast and accurate!

**image**

yay!!!


# ratrain and rethinking

**![people detection with haar cascade - Stack Overflow](https://lh3.googleusercontent.com/OQvhQnXnTFipcFtUnCunAsOt_xybR3S_dSeotkB3p8w6Dmg-GOm7FfDY1a105-ZOwEU5bYEyFbP65BEfzKRCTLXggYJog5gOKt2XTCEqojei_XYUp8IhmAjdz3cu6F-YMVUY8KkEQh-LMJMDbQ)**
### wait what?!
- I thout that we got everythink fixed, what happend?
- The reason for that is the image we tested on was to easy and our machine got the face right but in a more complicate image the machine also detect object that not face.
- The random forest was very fast but still he dosen't has the percition of the SVM, we diden't know what we should do from here, we have 2 models:
1. first is very fast but dosent accurate
2. second is slow and accurate
- From this state we were needed to think outside the box and think beyond **parameter tuning**
- After a lot of testing on image we realize that the tree model is reconize a lot of none-face objects as faces **but** it does reconize faces in addition to that.
- So what wev'e done is to take the tree model and let him filter the image and then run the SVM model on the images that were marked as face.
- It crated a very fast check(0.7 sec) and very Precise check.

# stage 2.0 - clustering
- Ok so more intersting stage then only "upgrading our model" is the reconize of other faces!
- to know which face is differnt from each other we 2 steps:
1. Extract feachers from the face, which means to know what makes our face unique.
2. To know which feachers are similar to other feachers.
## stage 2.1 - Face encodings
- There is a lot of methods of extracting feachers from a face and we choose the 128 messuraments method.
