
**Problem Statement:**

To implement a deep learning model, that when given a photo, will classify it into one of two categories: sushi or sandwich. Keeping in mind that this will be used via a smartphone camera.

### Please check youtube video of the app

> **YOUTUBE VIDEO:** https://www.youtube.com/watch?v=3EE6ZpwKUQg

![alt text](https://github.com/arun-apad/cookpad_sandwushi/blob/master/problem.gif)

Below is a write up of the approach taken to solve this problem. I have categorized it into the following sections.

- **Section 1:** Notes on the choice of language, framework, model and architecture used to solve the problem.
- **Section 2:**  Notes on the general approach I chose to solve the problem. The datasets that are used and its characteristics.
- **Section 3:**  Notes on the tuning parameters used in the deep learning model created. Choice of values used for the tuning.
- **Section 4:**  Notes on training a model on Cookpad data and then testing it on the data from Food101. Validating the model and analysing the results. Validation metrics used.
- **Section 5:**  Notes on training a model on food101 data and then testing it on the Cookpad data. Validating the model and analysing the results.
- **Section 6:**  Notes on how the model can be used in production and what modifications would be necessary to deploy it into a useful product.


The code and instructions to run the project using a docker image directly or using a dockerfile or using git are available at https://github.com/arun-apad/cookpad_sandwushi/blob/master/ReadMe_For_installation.md


## **Section 1**

- The language used is  **python,** for its community support, simplicity and variety of libraries available.
- The framework used is  **Tensorflow** , since it is popular and can be easily used to optimize for mobiles using TOCO. Also, tensorflow gives intermediate protobuff files which can be used in parallel to optimize and use for mobile development and testing.
- The model uses an image-net pre-trained  **CNN**  model called  **mobilenet**. The  **mobilenet architecture** is light, fast and works well on embedded devices and mobiles but sacrifices a little on accuracy. This comes in various sub-archtechture based on speed to train and accuracy trade=offs. The sub architechure used is **mobilenet 1.0 which uses a 224X224 image size** to do the training. As the solution is intended to be used on a mobile, the trade-off seemed fair for this demonstration.

## **Section 2**

- The first thing I did was check for other available datasets I could use for the problem. I came across  **Food101****  data** from wiki of the book titled &quot;European Conference on Computer Vision&quot;.
- Food101 has 101 classes of food with 1000 images each. It has four categories of Sandwich, from which I randomly chose 250 images each to form 1000 images of Sandwich.
- Now there are 804 images from Cookpad(402 images per class) and 2000 images from Food101(1000 images per class), so I decided to build 2 models.
- The first will be a model trained on **Cookpad data** called **cookpad\_model** and validate that model using all the images from food101.
- The second model will be a model trained on **Food101 images** , called the **food101\_model** and validate this model using Cookpad images.
- Then analyse and compare the results and evaluation metrics of the 2 models.
- Finally, create a model called **all\_food model** and an android app to classify all 101 dishes of the **Food101 data.**

## **Section 3**

- The following are the tuning parameter values I chose to train All 3 models.
- **--how\_many\_training\_steps**  (epochs) 2000
- **--learning\_rate**  = 0.001 as it yields optimal results for given epochs.
- **--testing\_percentage**  = 10, % of unseen data to test after training and validation.
- **--validation\_percentage**  = 30, % of data used during learning to validate and improve accuracy.
- **--eval\_step\_interval**  = 10, how often to evaluate, keeping it low as the training would be faster.
- **--train\_batch\_size**  = 100, no. of training images to learn at every iteration. Increase this will use more computing resources.
- **--validation\_batch\_size** = 100, no. of images to validate and learn at every iteration. Increase this will use more computing resources.
- **--test\_batch\_size** == -1, use all images, as this will be done only one at the end of all iterations and only 10% of the data.
- **--flip\_left\_right, --random\_crop, --random\_brightness, --random\_scale ** are all set to 0 as using this will increase accuracy but at the cost of training time. But, can be used if accuracy is a priority over training time.

## **Section 4**

> Executing the retrain.py with following parameters will create cookpad\_model trained on cookpad data.  Parameters indicate model name, location to save model and model architecture to be used.
> 
> `python retrain.py --bottleneck\_dir=tf\_files/bottlenecks  --model\_dir=tf\_files/models/ --summaries\_dir=tf\_files/training\_summaries/cooking\_cookpad\_model  --output\_graph=tf\_files/cookpad\_model.pb --output\_labels=tf\_files/retrained\_labels.txt --architecture=&quot;mobilenet\_1.0\_224&quot; --image\_dir=tf\_files/cookpad`

#### This will output 3 metrics:

![alt text](https://github.com/arun-apad/cookpad_sandwushi/blob/master/cookpad_model.JPG)

![alt text](https://github.com/arun-apad/cookpad_sandwushi/blob/master/cookpad_model_val.JPG)

> **Validation Accuracy:**  **84%** approximately.
> 
> **Cross Entropy:**  **0.58**  approximately, lower the better. Interpreted as loss metric to measure the optimization.
> 
> **Test\_Accuracy:**   **85%** approximately. The 10% of images the model has not seen before.

Next, the model is used to classify the 2000 food101 images using the command.

> `python validating\_food101\_images\_with\_cookpadModel.py`

This will output the following metrics:

> **Precision(sushi)**: **0.7561436672967864****  **(% of images classified as sushi was actually sushi)
>
> Formula = No. images the model classified as sushi correctly / No. images the model classified as sushi instead of sandwich

> **Precision(sandwich):**  **0.7876857749469215** (% of images classified as sandwich was actually sandwich)
>
> Formula = No. images the model classified as sandwich correctly / No. images the model classified as sandwich instead of sushi

> **Recall(sushi):**  **0.8** (Accuracy %)
>
> Formula = No. images the model classified as sushi correctly / total no of sushi images

> **Recall(sandwich):** **0.742** (Accuracy %)
> 
> Formula = No. images the model classified as sandwich correctly / total no of sandwich images

> ##  **Interpreting the results:** 
> Using **food101 data for testing** , the **cookpad\_model** has an overall accuracy greater for sushi over sandwich ( **recall =.8 vs .74** )but, it has a slightly better precision for sandwich over sushi ( **precision =.75 vs .78** ). This means that there is a tendency for the model over-learn sushi better than sandwich or that there are anomalies/bad labels in the testing or training data. Removing anomalies and training the data or decreasing the learning-rate and increasing the iterations will show better precision for both. Overall the model does much better when tested on unseen test data from the same cookpad dataset **accuracy=85%.**

## **Section 5**

> Executing the retrain.py with following parameters will create food101\_model trained on food101 data.  Parameters indicate model name, location to save model and model architecture to be used.
> 
> `python retrain.py --bottleneck\_dir=tf\_files/bottlenecks  --model\_dir=tf\_files/models/ --summaries\_dir=tf\_files/training\_summaries/cooking\_food101\_model  --output\_graph=tf\_files/food101\_model.pb --output\_labels=tf\_files/retrained\_labels.txt --architecture=&quot;mobilenet\_1.0\_224&quot; --image\_dir=tf\_files/food101`

#### This will output 3 metrics:

![alt text](https://github.com/arun-apad/cookpad_sandwushi/blob/master/food101.JPG)

![alt text](https://github.com/arun-apad/cookpad_sandwushi/blob/master/food101_val.JPG)

> **Validation Accuracy:** 94% approximately.
> 
> **cross entropy:** 0.2 approximately, lower the better. Interpreted as loss metric to measure the optimization.
> 
> **Test\_Accuracy:**   94% approximately. The 10% of images the model has not seen before.

Next, the model is used to classify the 804 Cookpad images using the command.

> `python validating\_cookpad\_images\_with\_food101Model.py`

This will output the following metrics:

>**Precision(sushi):**  **0.8228571428571428** (% of images classified as sushi was actually sushi)
> 
> Formula = No. images the model classified as sushi correctly / No. images the model classified as sushi instead of sandwich

> **Precision(sandwich):**  **0.748898678414097** (% of images classified as sandwich was actually sandwich)
> 
> Formula = No. images the model classified as sandwich correctly / No. images the model classified as sandwich instead of sushi

>**Recall(sushi):**  **0.7164179104477612** (Accuracy %)
>
> Formula = No. images the model classified as sushi correctly / total no of sushi images

>**Recall(sandwich):**  **0.845771144278607** (Accuracy %)
> 
> Formula = No. images the model classified as sandwich correctly / total no of sandwich images


> ## **Interpreting the results:** 
> Using **cookpad data for testing** , the **food101\_model** has an overall accuracy greater for sandwich over sushi ( **recall =.84 vs .71** )but, it has a better precision for sushi over sandwich ( **precision =.74 vs .82** ). This means that there is a tendency for the model over-learn sandwich better than sushi or there exists some noise in the training or testing data. Removing noise in the data or increasing the no. of iterations and decreasing the learning-rate during training can improve the precision for both. Overall the model does much better when tested on unseen test data from the same food101 dataset **accuracy=94%.**

### cookpad_model vs food101_model vs allFood101_model 

![alt text](https://github.com/arun-apad/cookpad_sandwushi/blob/master/all.JPG)

![alt text](https://github.com/arun-apad/cookpad_sandwushi/blob/master/all_val.JPG)


## **Section 6**

Now that we have a decent enough tensorflow model, the next step is to see how it can be used via a smart phone. There are mainly 3 approaches that are popular.

> 1) Host the model on a Django server(any popular) server and do the classification via transferring the image and getting the result through a REST api. Very common approach that is fast becoming irrelevant when it comes to mobile and embedded platforms.

> 2) Host the model on Tesnsorflow serving, a flexible, high-performance serving system for machine learning models, designed for production environments. TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. TensorFlow Serving provides out-of-the-box integration with TensorFlow models but can be easily extended to serve other types of models and data. Good approach if there can be latency trade-offs in the app.

> 3) Store the model inside the app by converting it to a tflite(for Android) or coreML(for ios) format. This is great for using the phone&#39;s camera and stream the incoming frames or just using an image from the phone&#39;s gallery without having to hit an external API everytime. This is tensorflow&#39;s lightweight format to be used within embedded devices and phones. It needs no internet for classification and hence is great for apps which use the camera directly and IOT devices.

I used option 3 and created an app to classify 101 dishes as part the demonstration for this assignment. The app is built on Android and uses tflite format of the model that was trained on all the foods of Food101 dataset.

### Please check youtube video of the app

> **YOUTUBE VIDEO:** https://www.youtube.com/watch?v=3EE6ZpwKUQg

 I used the protobuff or .pb graph/model file that was trained on all the food and used a tool called TOCO( [TensorFlow Lite Optimizing Converter](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/README.md)) to convert the .pd to a .lite format. Basically, the tool compresses the features learned of the model, so that it is more optimized for mobiles and devices. It is analogical to doing a PCA to reduce the components of the array holding the weights learned by the model. Once the .lite format is created it can be stored within the assets used by the mobile app.

**Please note:**

> All resources used to build the project and app are for demonstrating this assignment exclusively and have no intention of using/publishing it anywhere. The model trained on this, allfood_model has an accuracy of only at about 60%. Did not train further for accuracy.

> I got data from Food101 data from [https://www.vision.ee.ethz.ch/datasets\_extra/food-101/](https://www.vision.ee.ethz.ch/datasets_extra/food-101/). 

> I used images and links from [https://cookpad.com/uk/](https://cookpad.com/uk/%20) for the app development.

> I re-used a lot of code and resources for the app development from my own app that I published on April 2018 [https://play.google.com/store/apps/details?id=com.aeye.flowers.app](https://play.google.com/store/apps/details?id=com.aeye.flowers.app)
