from run.side_maintrain import main_train_v1_1


pth_saved_dir   = "/project/hypertrain/nband_nmel16_popl"
pth_save_path   = "/project/hypertrain/nband_nmel16_popl/snn_model_hogefield.pth"

main_train_v1_1(pth_saved_dir, pth_save_path)