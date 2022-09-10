import argparse,os
from train import Train
# from train_fusion import Train


def main(arg_list):
    training_pipe = Train(arg_list)
    training_pipe.run()
    return

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Get command-line arguments.")
    main_path = os.path.dirname(os.path.abspath(__file__))
    arg_parser.add_argument('--checkpoint_folder', type=str,
                            default=os.path.join(main_path, 'checkpoint','denoised'))

    # ------------------------------------------------
    # Training arguments
    # ------------------------------------------------

    # Log folder where Tensorboard logs are saved
    arg_parser.add_argument('--log_name', type=str,
                            default='classifier')
    arg_parser.add_argument('--model_type', type=str,
    default='effnetv2s')
    arg_parser.add_argument('--epochs', type=int,
                            default=200)
    arg_parser.add_argument('--batch_size', type=int,
                            default=16)
    arg_parser.add_argument('--LR', type=int,
                            default=5e-3)

    # Folders for training and validation datasets.
    arg_parser.add_argument('--train_input', type=str, default=os.path.join('/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/MCCAI-2022/DRAC/','ds', 'grading/denoised/denoise_mid/train'))
    arg_parser.add_argument('--valid_input', type=str, default=os.path.join('/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/MCCAI-2022/DRAC/','ds','grading/denoised/denoise_mid/val'))

    arg_list = arg_parser.parse_args()
    main(arg_list)