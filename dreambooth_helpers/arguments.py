import os, sys, shutil, json, argparse
from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1


def parse_arguments() -> JoePennaDreamboothConfigSchemaV1:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "project_name",
        type=str
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        required=False,
        help="A config file containing all of your variables"
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
        default=0.5
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
        default=500,
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
        "--epochs", '--repeats',
        type=int,
        meta='--repears',
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
        choices=["linear", "cubic", "area", 'lanczos'],
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
    config_file_path=opt.config_file_path)
    if not os.path.exists(config_file_path):
        print(f"{config_file_path} not found.", file=sys.stderr)
        return None
    else:
        config_file = open(config_file_path)
        config_parsed = json.load(config_file)
        
        if config_parsed['schema'] == 1:
            return config(
                project_name=config_parsed['project_name'],
                max_training_steps=config_parsed['max_training_steps'],
                save_every_x_steps=config_parsed['save_every_x_steps'],
                training_images_folder_path=config_parsed['training_images_folder_path'],
                regularization_images_folder_path=config_parsed['regularization_images_folder_path'],
                token=config_parsed['token'],
                class_word=config_parsed['class_word'],
                flip_percent=config_parsed['flip_percent'],
                learning_rate=config_parsed['learning_rate'],
                model_path=config_parsed['model_path'],
                config_date_time=config_parsed['config_date_time'],
                seed=config_parsed['seed'],
                debug=config_parsed['debug'],
                gpu=config_parsed['gpu'],
                model_repo_id=config_parsed['model_repo_id'],
                token_only=config_parsed['token_only'],
                resolution=config_parsed['resolution'],
                batch_size=config_parsed['batch_size'],
                accumulated_gradients=config_parsed['accum_grads'],
                resampler=config_parsed['resampler'],
                epochs=config_parsed['repeats'],
                center_crop=config_parsed['center_crop']
        )
        else:
            print(f"Unrecognized schema: {config_parsed['schema']}", file=sys.stderr)
else:
    return config(
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
        accumulated_gradients=opt.accum_grads
    )

