This is a project about the transfer attack on ASV spoofing systems.
It is based on the code by https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts, 
https://github.com/asvspoof-challenge, https://github.com/ghuawhu/end-to-end-synthetic-speech-detection, and https://github.com/Harry24k/adversarial-attacks-pytorch. 


## useage: 

### Training

To train the model, use

```python main.py --config ./config/your_model.conf```

to evaluate add `--eval` to evaluate the model itself.


### Attacks

To do adversarial attacks and save the adversarial examples, use

```python main.py --config ./config/you_attack.conf```

Note that you should specify the blackbox and whitebox models, and the attack method in the config file

Add `--eval` to evaluate the adversarial examples, add `--attack_eval` to evaluate during attack without saving the examples.


