import os

class WandbLogger:
    """
    Log using `Weights and Biases`.
    """
    def __init__(self, opt):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )
        
        self._wandb = wandb

        # Initialize a W&B run
        if self._wandb.run is None:
            self._wandb.init(
                project=opt['wandb']['project'],
                config=opt,
                dir='./experiments'
            )

        self.config = self._wandb.config

        if self.config['log_eval']:
            self.eval_table = self._wandb.Table(columns=['fake_image', 
                                                         'sr_image', 
                                                         'hr_image',
                                                         'psnr',
                                                         'ssim'])

    def log_metrics(self, metrics, commit=True): 
        """
        Log train/validation metrics onto W&B.

        metrics: dictionary of metrics to be logged
        """
        self._wandb.log(metrics, commit=commit)

    def log_image(self, key_name, image_array):
        """
        Log image array onto W&B.

        key_name: name of the key 
        image_array: numpy array of image.
        """
        self._wandb.log({key_name: self._wandb.Image(image_array)})

    def log_checkpoint(self, current_epoch, current_step):
        """
        Log the model checkpoint as W&B artifacts

        current_epoch: the current epoch 
        current_step: the current batch step
        """
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        gen_path = os.path.join(
            self.config.path['checkpoint'], 'I{}_E{}_gen.pth'.format(current_step, current_epoch))
        opt_path = os.path.join(
            self.config.path['checkpoint'], 'I{}_E{}_opt.pth'.format(current_step, current_epoch))

        model_artifact.add_file(gen_path)
        model_artifact.add_file(opt_path)
        self._wandb.log_artifact(model_artifact, aliases=["latest"])

    def log_eval_data(self, fake_img, sr_img, hr_img, psnr, ssim):
        """
        Add data row-wise to the initialized table.
        """
        self.eval_table.add_data(
            self._wandb.Image(fake_img),
            self._wandb.Image(sr_img),
            self._wandb.Image(hr_img),
            psnr,
            ssim
        )

    def log_eval_table(self, commit=False):
        """
        Log the table
        """
        self._wandb.log({'eval_data': self.eval_table}, commit=commit)
