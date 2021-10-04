"""""""""""""""""""""""""""""""""" Config file """""""""""""""""""""""""""""""""""

########## Path to test images ####################
ROC_id = 'cm_cm_msk_MOCO_1.0_hnm_2'
TEST_HEAT_MAP_DIR = '/home/srinidhi/Research/Code/Lymph_Node_Detection/Results/CAM/Heat_Maps/MSK/MoCo/1.0_hnm_2/prob_binary_map/'      # test prob maps
TEST_WSI_PATH ='/home/srinidhi/Research/Code/Lymph_Node_Detection/Results/Annotations/MSK/WSIs/'                                # test WSIs
TEST_CSV_GT = '/home/srinidhi/Research/Code/Lymph_Node_Detection/Results/Annotations/MSK/test_GT.csv'                           # test GT
Level_used_heatmaps_gen = 2    # (6 - Camelyon, 2 - MSK)

# Save test features
TEST_HEATMAP_FEATURE_CSV = '/home/srinidhi/Research/Code/Lymph_Node_Detection/Results/Save_Results/Cam/cm_cm_cm_MOCO_1.0_hnm_2.csv'    # 129 test images
ROC_Value_Store = '/home/srinidhi/Research/Code/Lymph_Node_Detection/Results/Save_Results/Cam/'                                 # save test ROCs

# Statistical significance of two ROC's
TEST2_HEATMAP_FEATURE_CSV = '/home/srinidhi/Research/Code/Lymph_Node_Detection/Results/Save_Results/Cam/cm_cm_cm_MOCO_1.0_hnm.csv'

# ########## Path to train images ####################
TRAIN_HEAT_MAP_DIR = '/home/srinidhi/Research/Code/Lymph_Node_Detection/Results/CAM/Heat_Maps/Cam/MoCo/train_val/1.0_hnm_2_tr_val/train/'   # train prob maps
VAL_HEAT_MAP_DIR = '/home/srinidhi/Research/Code/Lymph_Node_Detection/Results/CAM/Heat_Maps/Cam/MoCo/train_val/1.0_hnm_2_tr_val/val/'       # val prob maps

TRAIN_WSI_PATH ='/media/srinidhi/CSrinidhi_Lab/Public_Datasets/CAMELYON2016/train/'       # train WSIs
VAL_WSI_PATH ='/media/srinidhi/CSrinidhi_Lab/Public_Datasets/CAMELYON2016/val/'           # val WSIs

TRAIN_CSV_GT = '/home/srinidhi/Research/Code/Lymph_Node_Detection/Results/Annotations/Cam/train/train_GT.csv'                          # train GT
VAL_CSV_GT = '/home/srinidhi/Research/Code/Lymph_Node_Detection/Results/Annotations/Cam/train/val_GT.csv'                              # val GT

# # Save train/val features
TRAIN_HEATMAP_FEATURE_CSV = '/home/srinidhi/Research/Code/Lymph_Node_Detection/Results/Save_Results/Cam/TRAIN_cm_cm_cm_MOCO_1.0_hnm_2.csv'
VAL_HEATMAP_FEATURE_CSV = '/home/srinidhi/Research/Code/Lymph_Node_Detection/Results/Save_Results/Cam/VAL_cm_cm_cm_MOCO_1.0_hnm_2.csv'


######### List of features ##########
heatmap_feature_names = ['region_count', 'ratio_tumor_tissue', 'largest_tumor_area', 'longest_axis_largest_tumor at t0.5',
                         'longest_axis_largest_tumor at t0.9', 'pixels_gt_90', 'avg_prediction', 'max_area', 'mean_area',
                         'area_variance', 'area_skew', 'area_kurt', 'max_perimeter', 'mean_perimeter', 'perimeter_variance', 'perimeter_skew',
                         'perimeter_kurt', 'max_eccentricity', 'mean_eccentricity', 'eccentricity_variance',
                         'eccentricity_skew', 'eccentricity_kurt', 'max_extent', 'mean_extent', 'extent_variance',
                         'extent_skew', 'extent_kurt', 'max_solidity', 'mean_solidity', 'solidity_variance',
                         'solidity_skew', 'solidity_kurt', 'label']
#############################


############ Some basic functions ##################
def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename


def format_2f(number):
    return float("{0:.2f}".format(number))


def step_range(start, end, step):
    while start <= end:
        yield start
        start += step
