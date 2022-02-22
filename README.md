# mmWave-dataset
60 GHz Millimeter-Wave FMCW Radar-based Open Dataset for Hand Gesture Recognition.

mmWave dataset has collected using BGT60TR13C 60GHz mmWave radar chip and MATLAB (2D FFT).

mmWave dataset consists of four hand gestures, each with 1,000 images (total 4,000 images).

# Prerequisites
python 3.x

Tensorflow > 2

matplotlib

argparse

opencv


# Usage
* For training, 

<pre>
<code>
python train.py --model model --epoch 0 [model has [vgg, resnet, efficientnet] and train epoch]
</code>
</pre>
   
* For testing, 

<pre>
<code>
python test.py 
</code>
</pre>
   
* For visualize mmWave radar dataset, 

<pre>
<code>
python plot_dataset.py 
</code>
</pre>   

* For visualize train results, 

<pre>
<code>
python plot_result.py
</code>
</pre>     

# Author
Minjae Yoo
