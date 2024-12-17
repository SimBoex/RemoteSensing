# RemoteSensing
Here I include some projects in python for remote sensing!


1. For the semantic segmentation:
   I used wandb for the first time and applied a procedure learnt during the university to fine tune the hyperparameters without using a ton of GPU.
   Some code and ideas is taken from [here](https://www.youtube.com/@DigitalSreeni)


  ![img1](https://github.com/SimBoex/RemoteSensing/blob/6e393f39eda12bb88a09781afa3442b7f800336c/Wandb_images/W%26B%20Chart%2025_06_2024%2C%2015_45_00.png)

  ![img1](https://github.com/SimBoex/RemoteSensing/blob/6e393f39eda12bb88a09781afa3442b7f800336c/Wandb_images/W%26B%20Chart%2025_06_2024%2C%2015_44_46.png)


     - Observation: it is possible to improve the performances improving the convergence of the models (Train-Val losses):
        1. Performing a more in depth Grid-search;
        2. Expand the Grid-search using a CV to check if the data quality is an issue;
        3. add a LR-scheduler to improve the convergence;


