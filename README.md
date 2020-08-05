
# Welcome to our face detection project!

Hi! this was our biggest project yet and it includes lodes of knowledge and hard work.
we made this GitHub project to share with you the things we learn from it.
hope you will expand your knowledge about machine learning, AI and image filtering.
# what is it?
*![What is bounding box in image processing? - Quora](https://lh3.googleusercontent.com/Ug9cC9AEJxxnLRlwU9sKDQwPYxwkf40H1ksJ6EDt745XZfBd3SWuI88t_BMtMcF-JvM4IwaLquXmhGAG_QRcgoCW7e94P3v3SLhyw2bGwueDY5WebCqWX5gYi3Y5-1oHmO1ggzzL59Q)


Well, basically our code takes an image and detects where the face area in the image and who is it.
 - classification \bounding box- we use SVM, random forest and machine learning for knowing where the face is.
 - clustering - the way measuring something and tell how similar it to another thing.
## Prerequisites
 - tensorflow 1.14
 - numpy
 - skimage
 - matplotlib
 - dlib
 - sklearn
 - firebase_admin
 

## Introduction to AI
AI is the future, you can see it almost in everything: on your phone, TV, Spotify and more...
in our project we develop a brain\mind who got a self-thinking and can tell us whether an image contains a face or not. 

# Stage 1.0 - Bounding Box Classifier

we would like to make a brain\mind who can look at an image and tell what is it just background and what is it face.
## SVM
SVM - Support Vector Machine 
its a method that using supervised learning.
It means that in this method we will be using learning in positive\negative attitude.
we will give our machine 2 arrays,** one** is positive and contains faces we would like to allocate and the **second** contains negative images of grass, wall, Sky exd...
as we can see in pic1 we got 3 lines that the machine set about the data:
the **green** line is bad, the **blue** line is good but not accurate and the **red** line is the best line.

## Image filtering
Our face is a complicated variant and we cannot just pass the input of the pixel of our image to the machine because it is way too big and doesn't much.
to solve this problem we will use HOG or Histogram of Oriented Gradients.
basically, it measures the difference between point 9 pixels and sets an arrow that points to the biggest difference.

![](https://lh5.googleusercontent.com/FdJjher-NuALheOVJtohgLAq5dT5bMQcyC9kU4UlKzHatOy3lxF5QzYXCkDGGRw-HyHWpGtP0EiPfTFWVzb2ufgTijr1PmzMyJPWUStiTsLFJP97qryeju63c6np2F83KbV6EklCMAk)

## Data
SVM requires a lot of data to work properly,
in our project we use lfw for the faces data and skimage.data library for the negative data
lfw - [https://scikit-image.org/docs/0.14.x/api/skimage.data.html#skimage.data.lfw_subset](https://scikit-image.org/docs/0.14.x/api/skimage.data.html#skimage.data.lfw_subset)

skimage.data - **[https://scikit-image.org/docs/dev/api/skimage.data.html](https://scikit-image.org/docs/dev/api/skimage.data.html)**

# Stage 1.5  - train
In this project we used python to power because it has so much support on AI and it very easy to use.
after setting up the code and training the SVM model we ended up with this:

**![](https://lh4.googleusercontent.com/K6S7j7wWKfxYtQoQ_1PHWgArS-DnFd-DrfTiynIqEAJKPm6UxftijtAx_i8QMkpubAyFHCSTQVgltLRyCgGZBbYnZa2HYCGClQuGyUG4nOIiXw-U2p-VxDCV1DYJgL7iFjJ2s8a4tfI)**

yay.

## The work behind AI
In our first time using Ai model we encountered in the hardest part of Ai which is **parameter tuning**
Now the creativity had to be shown and we have to work harder to get what we want, which is a fast and accurate machine.
so after trying to make our machine smarter we doubled the negative data and train it more times
and we ended up with this:
**image**

yay?
# Concloutin and a new way
our vision for the end of this task is the same as the previous one "upgrading the model" but it complicity different. our targets are very clear, we would make our model better on the staff below:
1. our model runs very slow, we would like to get to state when the client would get an immediate response from the server.
2. The model is still not accurate as we want we have to upgrade it but not enough.
3. the bounding boxes we get represent a lot of information and we need to reduce it to one bounding box.
## stage 1.5.1 - NMS(Non-Maximum-Suppression)
A quick way to take a lot of data and reduced it by averaging all the variables we need.
## stage 1.5.2 - Random Tree Forest
### choices trees:
To get a better understanding of what is a Random Tree Forest we first need to understand what is a **Decision Tree Classifier**
choices trees are pretty old fashion way but it worked very well for us in this project.
decision tree gets data and tries to classify it by sorting the data to questions.
ex:
- we got a large data of people of different age, location and money.
- I would like to build a model that when I enter age, location of someone I will know whether he above The poverty line.
- The tree will build that way:
1. is man?
2. is live in the south?
3. is age above 50?
4. is live in Israel?
- when I put a new man into the data the model will ask him all these questions and I will get my result(like a real interview!).
- in the img we can see a decision tree that decides whether a man is fit or not.
- of course that all these questions the model decides by himself and when it became for pixels in a face it gets more complicated.

**![Decision Tree Classification - Towards Data Science](https://lh6.googleusercontent.com/uGT90mGTBNEy_z64sYR1VXEHhwvepYAbIuQaQodVcxVicHr7LFFAgo2SPBPZEOv67UPGov6ibqxdl74KSmsAqRaggYwkD5R-8X8JFIweGffnaDv4bd-AUKu4lzMccmYr9E8AJjfcwjI)**

**Random Tree Forest**
okay... we get what decision tree means but what "forest" means?
the forest is a model that makes use of several trees that have been trained on different parts of the data.
this model gives us more accuracy because there are more outcomes from the trees and we can validate our output from a different angle.
in the img below we can see the **blue** tree is a regular tree and it has been trained on all the 4 parts of our data. On the other hand the **green** trees are a forest and they have been trained on different data.

**![Understanding Random Forest - Towards Data Science](https://lh3.googleusercontent.com/0FWIRKkEdtC_r5EgmmiFx_YPxiatm2nIKfLKVsv63hqnYX_L_PlRRsp1TvBxQxAVUBRd8GwV0Pcdidtggr2_plVo6U6bfswh9-WvpV1WdnoSDP_hdV4vpfICOeXBAzLebhq-EDJ_u-_jvoGHog)**

In the image below we can a forest that got some decision based one 6 trees that said 1 while 3 trees said 0.
which means our answer in 1.

**![Understanding Random Forest - Towards Data Science](https://lh4.googleusercontent.com/IEBQQS8WE1FmhS8wtY3ao7nl1uIM_DIH26_uvc33Fd-aSv1h9oRSQERAbqrHfUsMQ-dcs8KchycZ_lNVy5yeTVEYX5Zfb70DFXgVeI7ROkLIWqXwNEcjC8ysKXMgD2JmGBFJCiR08fTplK26yQ)**

**coclution**
So why tree forest is doing such a good job for us? 
The reason for that is because it filters automatically all the inputs. It means that until now the SVM model was checking every part in the image to understand if its face or not, but tree works differently.
desition tree checks specific parameters and only then continues to another check which means if we have a part in the image which is completely white or black the tree will immediately decide that it does not face while SVM will think a lot more time.
as we can see in the image below SVM classifies every image while Tree decides which dump immediately.

**![](https://lh6.googleusercontent.com/IIyONkam0tl5C1ekExgdShfSVqHXM4N1knXUbHiNVSJ0oUcm6TCPFXviSCgVSvc13wcHzRqroUL8DyNDfmjOAC__BQnwHTAI3REhwMs7w-d7Blc1ZkTiMFpoIaDamvm6asEILrVf_q7Jp28mZw)**

## stage 1.5.3 - retrain model
- To make our model more accurate we use in the method of **retrain model**
- Our model was trained on faces date that was 62 by 47 px, which means only faces in this size will absorb(we will fix that problem in the future) which means that if we run our - model on 200 by 200 image and all the image is facing we shouldn't find anything.
- that wasn't the case :(
- as you can see in the image below our model classify the eyes and mouse as a face, we can understand why because eyes and mouse are very similar to a face although they don't.
- What we've done is taking 10,000 of these "mistakes" and train our model in addition to the data we already have.
- In the image below we can see the output of **stage 1.5**, we've got one bounding box and it is fast and accurate!

**image**

yay!!!


# retrain and rethinking

**![people detection with haar cascade - Stack Overflow](https://lh3.googleusercontent.com/OQvhQnXnTFipcFtUnCunAsOt_xybR3S_dSeotkB3p8w6Dmg-GOm7FfDY1a105-ZOwEU5bYEyFbP65BEfzKRCTLXggYJog5gOKt2XTCEqojei_XYUp8IhmAjdz3cu6F-YMVUY8KkEQh-LMJMDbQ)**
### wait what?!
- I thought that we got everything fixed, what happened?
- The reason for that is the image we tested on was to easy and our machine got the face right but in a more complicated image the machine also detects objects that not face.
- The random forest was very fast but still he doesn't have the precision of the SVM, we didn't know what we should do from here, we have 2 models:
1. first is very fast but doesn't accurate
2. second is slow and accurate
- From this state we were needed to think outside the box and think beyond **parameter tuning**
- After a lot of testing on an image we realize that the tree model recognizes a lot of none-face objects as faces **but** it does recognize faces in addition to that.
- So what we've done is to take the tree model and let him filter the image and then run the SVM model on the images that were marked as a face.
- It created a very fast check(0.7 sec) and a very Precise check.

# stage 2.0 - clustering
- Ok, so more interesting stage then only "upgrading our model" is the recognition of other faces!
- to know which face is different from each other we 2 steps:
1. Extract features from the face, which means to know what makes our face unique.
2. To know which features are similar to other features.
## stage 2.1 - Face encodings
- There is a lot of methods of extracting features from a face and we choose the 128 measurement method.
- While us humans looks for simularity in faces by eye size, nose position... this method is using a complecte other way.
- We will train a computer to find himself the most 128 acurate measurement that unique the face, this model were trained on over 3 milion faces and took 48 hours to understand which is the best way to messure face.
- This method is called **deep learning** when we do not understand how the computer got his result.
- As you can see in the image below we have woman that her face is traslated to 128 different measurement.
![A Beginners guide to Building your own Face Recognition System to ...](https://lh6.googleusercontent.com/l9uCm4Ir3cLc_jA2FCJ6N9WJTrEqGsSqEE1lAJQv0ar72aXXkhciVxubq4zSrnoYujcllakvm2htBVQp4GG9pxdnESI6OYi9tGiq1NUFdSB90vBLxmRH9K9C6l-lkt_K2zoZoLoT-LPiaI1V3g)


## stage 2.5 - find the simalrity
- Ok so we extract the feachers from the face but now how can we know which is similar and which differnt.
- For that we will use clustering, by the DBSCAN method because it relative simple and fit for as.
- DBSCAN takes data and senter it into groups by thier simularity.
- as you can see in the image below we got 3 groups  and the algorithm can split them.![https://lh5.googleusercontent.com/cCHPy36O9qJTtxHeJ-hjciG1pfSCziA_EL-XSDUsxH-Q88Ah6UVLr6sjJsjB64yxBijLwviQl5x5yVkIHP8x54M74voHNwXWiT7pBljpDSz5af5ynu0DzNu6mPOWbjqiBwHm7lss](https://lh3.googleusercontent.com/xLHoHEu55iqDEdFweGGeBIEYBRwZSg_hEOuZDSmMGGigtCMCl_lWIPvM_hiPS5hrk4IiT-o3mFvN7HYBpV4FmR8phmZAiyUpHQYoYyviUjzc6-i4XxwJ-OeedYQEScznKdTJgwuDh-hLjdep3g)

## stage 2.7 - dynamic window
- Ok were so close to the end but we only one last thing for the model be perfect.[enter link description here](https://github.com/ben-and-yahel/face_detection/blob/master/README.md#bottom%20line)
- Our model were trained on 62*47 matrix and he can only reconize face in this size.
- what weve done to solve this is to give our model the image in differnt sizes and search for different size faces in the image. 

# stage 3.0 - GUI (Graphic User Interface)
Well the last and most fun part is the building of our GUI.
We build an interface that can comunicate using socket with python and comuunicate with firebase so we can build our users so they will be cloud base.

# bottom line
In this project we faced a ton of bags, problems and obstagels and still had fun from every moment and learn a lot on every thing we wanted and even we didn't get the best model or product we are very proud on what we have done and learn.
This prject made us learn a lot about our self and what can we preduce if we very want something.

### hope you enjoind :)
