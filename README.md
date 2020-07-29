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

# Stage one - Bounding Box Classifiar

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

skimage.data - [https://scikit-image.org/docs/dev/api/skimage.data.html](https://scikit-image.org/docs/dev/api/skimage.data.html)

# Stage 1.5  - train
In this project we used python to power because it has so much support on AI and it very easy to use.
after seting up the code and training our svm moder we ended up with this:


![](https://lh4.googleusercontent.com/K6S7j7wWKfxYtQoQ_1PHWgArS-DnFd-DrfTiynIqEAJKPm6UxftijtAx_i8QMkpubAyFHCSTQVgltLRyCgGZBbYnZa2HYCGClQuGyUG4nOIiXw-U2p-VxDCV1DYJgL7iFjJ2s8a4tfI)


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
