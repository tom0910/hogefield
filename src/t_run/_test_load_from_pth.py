import utils.training_functional as FT
import utils.loading_functional as FL
import utils.training_utils as TU
import core.Model as SNNModel

pth_file_path="/project/hyperparam/trial/pth/epoch_1.pth"
FT.load_checkpoint(pth_file_path=pth_file_path, model=None, optimizer=None )