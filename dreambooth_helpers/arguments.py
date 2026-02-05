import argparse

from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1


def parse_arguments() -> JoePennaDreamboothConfigSchemaV1:
    def _get_parser(**parser_kwargs):
        def str2bool(v):
            if isinstance(v, bool):
                return v
            if v.lower() in ("yes", "true", "t", "y", "1"):
                return True
            elif v.lower() in ("no", "false", "f", "n", "0"):
                return False
            else:
                raise argparse.ArgumentTypeError("Boolean value expected.")

        parser = argparse.ArgumentParser(**parser_kwargs)

        parser.add_argument(
            "--config_file_path",
            type=str,
            required=False,
            default=None,
            help="A config file containing all of your variables"
        )
        parser.add_argument(
            "--project_name",
            type=str,
            required=True,
            default=None,
            help="Name of the project"
        )
        parser.add_argument(
            "--debug",
            action='store_true'
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=1337,
            help="seed for seed_everything",
        )
        parser.add_argument(
            "--token",
            type=str,
            required=True,
            help="Unique token you want to represent your trained model. Ex: firstNameLastName."
        )
        parser.add_argument(
            "--token_only",
            action='store_true',
            help="Train only using the token and no class."
        )
        parser.add_argument(
            "--training_model",
            type=str,
            required=False
        )

        parser.add_argument(
            "--train_images",
            type=str,
            required=True,
            help="Path to training images directory"
        )
        parser.add_argument(
            "--reg_images",
            type=str,
            required=False
        )
        parser.add_argument(
            "--class_word",
            type=str,
            required=False
        )
        parser.add_argument(
            "--flip_p",
            type=float,
            required=False,
            default=0.0
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            required=False,
            default=1.0e-6,
            help="Set the learning rate. Defaults to 1.0e-06 (0.000001).  Accepts scientific notation."
        )
        parser.add_argument(
            "--save_every_x_steps",
            type=int,
            required=False,
            default=1500,
            help="Saves a checkpoint every x steps"
        )
        parser.add_argument(
            "--gpu",
            type=int,
            default=0,
            required=False,
            help="Specify a GPU other than 0 to use for training.  Multi-GPU support is not currently implemented."
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            required=False,
            default=1,
            help="image batch size and number of epochs to perform for iterable datasets"
        )
        parser.add_argument(
            "--epochs",
            type=int,
            required=False,
            default=100
        )
        parser.add_argument(
            "--val_iters",
            type=int,
            required=False,
            default=10,
            help="number of validating iterations to perform per image duirng validation split"
        )
        parser.add_argument(
            "--reg_iters",
            type=int,
            required=False,
            default=10,
            help="number of regularizing iterations to perform per image during training phase split"
        )
        parser.add_argument(
            "--resolution",
            type=int,
            required=False,
            default=512
        )
        parser.add_argument(
            "--resampler",
            type=str,
            required=False,
            choices=["bilinear", "bicubic", "area"],
            default="area"
        )
        parser.add_argument(
            "--center_crop",
            action="store_true",
            help="make ANY polygon your new favorite rhomboid!!!11!1"
        )
        parser.add_argument(
            "--accum_grads",
            type=int,
            required=False,
            default=1,
            help="Number of forward pass iteration gradients to process as a single iteration: 1 global training step = (1 x accum_grad) iterations"
        )
        return parser

    
    parser = _get_parser()
    opt, unknown = parser.parse_known_args()

    config = JoePennaDreamboothConfigSchemaV1()

    if opt.config_file_path is not None:
        config.saturate_from_file(config_file_path=opt.config_file_path)
    else:
        config.saturate(
            project_name=opt.project_name,
            seed=opt.seed,
            debug=opt.debug,
            gpu=opt.gpu,
            save_every_x_steps=opt.save_every_x_steps,
            training_images_folder_path=opt.train_images,
            regularization_images_folder_path=opt.reg_images,
            token=opt.token,
            token_only=opt.token_only,
            class_word=opt.class_word,
            flip_percent=opt.flip_p,
            learning_rate=opt.learning_rate,
            model_repo_id='',
            model_path=opt.training_model,
            epochs=opt.epochs,
            batch_size=opt.batch_size,
            regularization_iterations=opt.reg_iters,
            validation_iterations=opt.val_iters,
            resampler=opt.resampler,
            resolution=opt.resolution,
            center_crop=opt.center_crop,
            accumulated_gradients=opt.accum_grads,
        )

    return config
