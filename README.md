This is an implementation of the Transformer algorithm on time series data in pytorch. In this case the modelling of the sigmoid function is used as a toy problem

Usage:  
First all the necessary imports as well as matplotlib for visualisation.  
![](https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/doc/imports%2Cpng.PNG)  
Next we need to define some hyperparameters which will vary depending on the task.  
![](https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/doc/hyperparams.png)  
We initilisise the Network and an optimizier, in this case Adam, as well as an empty list to track losses for visualisation.  
![](https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/doc/init.png)  
Using matplotlib in jupyter notebook we can graph losses in real time, first lets initialise a figure.  
![](https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/doc/buildfig.png)  
We can now being training  
![](https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/doc/training.png)  
You should see a live plot that looks similar to this tracking the ouput error  
![](https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/doc/lossgraph.png)  
Now that the network is trained, lets give it the first few values of the sigmoid function and see how it approximates the rest.  
We create another figure to visualise this.  
![](https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/doc/compare.png)  
If all went well, the output should look something like this :  
![](https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/doc/comparegraph.png)  
Note that the network uses past values instead of the x axis for its predictions , so it makes sense that the output is offset.
However it did succesfully capture the shape.    

Resources:  
* Attention is all you need : https://arxiv.org/abs/1706.03762
* Deep Transformer Models for Time Series Forecasting : https://arxiv.org/abs/2001.08317


