In ddpm_turb2d:

/Compressed_Sensing/ddpm_turb2d/setup_turb2d.yaml: Defines the appropriate paths for the the repository, the location of the data, the output directory, etc. Also defines if you are using cpu or gpu.

/Compressed_Sensing/ddpm_turb2d/ddpm_turb2d_config.yml: Defines the runtime argumetns and hyperparameters for training. Includes the linear noise scheduler, the beginning/end beta values, the number of epochs, the timesteps in the forward/reverse processes, etc. Descriptions given at the bottom of the yaml

/Compressed_Sensing/ddpm_turb2d/ddpm_turb2d_train_cond.py: Training the models based off the sparse, the fno(sparse), and the truth outputs. It will load the configurations in the configuration yaml files.