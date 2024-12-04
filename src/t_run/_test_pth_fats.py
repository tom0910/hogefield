
from core.CheckPoinManager import CheckpointManager
file_path = "/project/hyperparam/checkLIF_099_01_20241202_221419/pth/epoch_9.pth"
# dir_path = "/project/hyperparam/"
chp_manager = CheckpointManager.load_checkpoint_with_defaults_v1_1(file_path=file_path)
chp_manager.print_contents()
# print(dir_path)



# from core.CheckPoinManager import CheckpointManager
# file_path = "/project/hyperparam/standard_n22_8khz/pth/epoch_2.pth"
# dir_path = "/project/hyperparam/standard_n22_8khz/pth/"
# chp_manager = CheckpointManager.load_checkpoint_with_defaults_v1_1(file_path=file_path)
# chp_manager.print_contents()
# print(dir_path)

                                                                   