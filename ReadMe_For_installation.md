# **Installation instructions**

### **The following are instructions to run the project followed by descriptions of the project folders and files**

There are 3 ways to run this project:

> 1) Using the docker image directly from Docker Hub repositories and running the scripts.
>
> 2) Using github to clone and run the scripts.
>
> 3) Using the docker file to get image and run the scripts.


### 1) These are the steps to run the scripts using docker image. The image has all we need to rum the project.

> * Install docker, skip this if you have docker already.                    
> ` sudo curl -fsSL https://get.docker.com/ | sh `

> * Get the docker image.                      
> ` sudo docker pull docker.io/arunpadmanabhan/cooking-sandwushi ` 

> * Run the docker image.                               
> ` sudo docker run -it arunpadmanabhan/cooking-sandwushi ` 

> * Activate the anaconda envoirnment.                         
> ` source activate cookpad `

> * Move in-to the project folder.                           
> ` cd /home/cookpad_sandwushi/ ` 

> * Run script to create cookpad_model using sandwich and sushi images from cookpad. Prints accuracy metrics on console after training and saves a model called cookpad_model.pb to the **tf_files/** folder in the project.
>
> ` python retrain.py --bottleneck_dir=tf_files/bottlenecks  --model_dir=tf_files/models/ --summaries_dir=tf_files/training_summaries/cooking_cookpad_model  --output_graph=tf_files/cookpad_model.pb --output_labels=tf_files/retrained_labels.txt --architecture="mobilenet_1.0_224" --image_dir=tf_files/cookpad `

> * Run script to create food101_model using sandwich and sushi images from food101. Prints accuracy metrics on console after training and saves a model called food101_model.pb to the **tf_files/** folder in the project.
>
> ` python retrain.py --bottleneck_dir=tf_files/bottlenecks  --model_dir=tf_files/models/ --summaries_dir=tf_files/training_summaries/cooking_food101_model  --output_graph=tf_files/food101_model.pb --output_labels=tf_files/retrained_labels.txt --architecture="mobilenet_1.0_224" --image_dir=tf_files/food101 ` 

> * Run script to validate cookpad_model on sandwich and sushi images from food101. Prints pricision and recall of cookpad_model against food101 data.
>
> ` python validating_food101_images_with_cookpadModel.py `  

> * Run script to validate food101_model on sandwich and sushi images from cookpad. Prints pricision and recall of cookpad_model against cookpad data.
>
> ` python validating_cookpad_images_with_food101Model.py ` 



### 2) These are the steps to run the scripts using git-clone. The git project has all we need to rum the project, except a few packages.

> * Create a conda environment with python version 3.  
> ` conda create --name cookpad python=3 ` 

> * Activate the anaconda envoirnment.                            
> ` source activate cookpad `

> * Install packages. 
>
> `   pip install tensorflow `
>
> `   pip install opencv-python ` 
>
> `   apt-get install libgtk2.0-dev -y `
>
> `   pip install Pillow ` 

> * Clone the git project and move in-to project folder.   
> ` git clone https://github.com/arun-apad/cookpad_sandwushi.git ` 
> 
> ` cd cookpad_sandwushi/ ` 


> * Run script to create cookpad_model using sandwich and sushi images from cookpad. Prints accuracy metrics on console after training and saves a model called cookpad_model.pb to the **tf_files/** folder in the project.
>
> ` python retrain.py --bottleneck_dir=tf_files/bottlenecks  --model_dir=tf_files/models/ --summaries_dir=tf_files/training_summaries/cooking_cookpad_model  --output_graph=tf_files/cookpad_model.pb --output_labels=tf_files/retrained_labels.txt --architecture="mobilenet_1.0_224" --image_dir=tf_files/cookpad `

> * Run script to create food101_model using sandwich and sushi images from food101. Prints accuracy metrics on console after training and saves a model called food101_model.pb to the **tf_files/** folder in the project.
>
> ` python retrain.py --bottleneck_dir=tf_files/bottlenecks  --model_dir=tf_files/models/ --summaries_dir=tf_files/training_summaries/cooking_food101_model  --output_graph=tf_files/food101_model.pb --output_labels=tf_files/retrained_labels.txt --architecture="mobilenet_1.0_224" --image_dir=tf_files/food101 ` 

> * Run script to validate cookpad_model on sandwich and sushi images from food101. Prints pricision and recall of cookpad_model against food101 data.
>
> ` python validating_food101_images_with_cookpadModel.py `  

> * Run script to validate food101_model on sandwich and sushi images from cookpad. Prints pricision and recall of cookpad_model against cookpad data.
>
> ` python validating_cookpad_images_with_food101Model.py ` 


### 3) Finally the Dockefile can be used to get the docker image and dependecies for the project.

> * The Dockerfile gets the image, activates conda envoirnemet and runs the scripts. Below are the contents of Dockerfile

> ` sudo docker pull docker.io/arunpadmanabhan/cooking-sandwushi`  
>
> ` sudo docker run -it arunpadmanabhan/cooking-sandwushi/bin/bash -c “source activate cookpad && RUN cd /home/cookpad_sandwushi/  && exec python retrain.py --bottleneck_dir=tf_files/bottlenecks  --model_dir=tf_files/models/ --summaries_dir=tf_files/training_summaries/cooking_cookpad_model  --output_graph=tf_files/cookpad_model.pb --output_labels=tf_files/retrained_labels.txt --architecture="mobilenet_1.0_224" --image_dir=tf_files/cookpad  && exec python retrain.py --bottleneck_dir=tf_files/bottlenecks  --model_dir=tf_files/models/ --summaries_dir=tf_files/training_summaries/cooking_food101_model  --output_graph=tf_files/food101_model.pb --output_labels=tf_files/retrained_labels.txt --architecture="mobilenet_1.0_224" --image_dir=tf_files/food101 && exec python validating_food101_images_with_cookpadModel.py && exec python validating_cookpad_images_with_food101Model.py ” `  


## These are the files and folders that are of importance to the project.

> * Apart from the python and markdown files in the project folder, there is a  ` app-debug.apk  ` which is the Android apk created for this demonstration. 

>  * The ` /tf_lite ` has the data and models for both cookpad and food101 in respective folders. There is only the model for allfood_model which classifies all 101 food types under ` /allFoodModel ` directory. The data can be accessed from https://www.vision.ee.ethz.ch/datasets_extra/food-101/. 













