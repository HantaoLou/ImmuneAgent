import pandas as pd
import numpy as np

def process_and_split_data(original_train_file, header_str,label_header, test_num, fold_num, saved_test_file):
    # Step 1: Read the table
    df_origin = read_table(original_train_file)
    
    # Step 2: Sort the table by the specified column
    df_origin = df_origin.sort_values(by=header_str)
    
    # Step 3: Extract the rows of end from the table to df_test
    df_test = df_origin.tail(test_num)
    df_origin = df_origin.iloc[:-test_num]
    
    # Step 4: Split the remaining table into positive and negative classes
    df_pos = df_origin[df_origin[label_header] == 1]
    df_neg = df_origin[df_origin[label_header] == 0]
    
    # Step 5: Split each class into a list of folds
    df_pos_list = np.array_split(df_pos, fold_num)
    df_neg_list = np.array_split(df_neg, fold_num)
    
    # Step 6: Save the positive and negative folds separately to maintain the ratio
    base_name = original_train_file.split('.')[0]
    for fold_id in range(fold_num):
        saved_pos_file = f"{base_name}_absplit_pos_fold{fold_id}.csv"
        saved_neg_file = f"{base_name}_absplit_neg_fold{fold_id}.csv"
        df_pos_list[fold_id].to_csv(saved_pos_file, index=False)
        df_neg_list[fold_id].to_csv(saved_neg_file, index=False)
        
    # Step 6: Save the table from df_test into the specified directory
    df_test.to_csv(saved_test_file, index=False)

def read_table(file):
    try:
        data = pd.read_csv(file)
    except:
        data = pd.read_excel(file)
    return data

if __name__ == '__main__':
    

    # Example usage

    original_train_file = 'Data/FLU_finetune_250312/0310-all_flu_binding_results.csv'
    label_header = 'bind_value'
    saved_test_file = 'Data/FLU_finetune_250312/0310-all_flu_binding_absplit_tst.csv'

    # original_train_file = 'Data/RSV_v5/0223_RSV_bind.csv'
    # label_header = 'bind_value'
    # saved_test_file = 'Data/RSV_v5/0223_RSV_bind_absplit_tst.csv'

    # original_train_file = 'Data/RSV_v4/0222_RSV_bind.csv'
    # label_header = 'bind_value'
    # saved_test_file = 'Data/RSV_v4/0222_RSV_bind_absplit_tst.csv'

    # original_train_file = 'Data/RSV_v3/0221_RSV_bind.csv'
    # label_header = 'bind_value'
    # saved_test_file = 'Data/RSV_v3/0221_RSV_bind_absplit_tst.csv'

    # original_train_file = 'Data/RSV_v3/0220_RSV_bind.csv'
    # label_header = 'bind_value'
    # saved_test_file = 'Data/RSV_v3/0220_RSV_bind_absplit_tst.csv'

    # original_train_file = 'Data/RSV_v2/0215_RSV_bind.csv'
    # label_header = 'bind_value'
    # saved_test_file = 'Data/RSV_v2/0215_RSV_bind_absplit_tst.csv'

    # original_train_file = 'Data/RSV/0212_RSV_bind.csv'
    # label_header = 'bind_value'
    # saved_test_file = 'Data/RSV/0212_RSV_bind_absplit_tst.csv'

    # original_train_file = 'Data/RSV_v3/0221_RSV_neu.csv'
    # label_header = 'neu_value'
    # saved_test_file = 'Data/RSV_v3/0221_RSV_neu_absplit_tst.csv'

    # original_train_file = 'Data/RSV_v2/0215_RSV_neu.csv'
    # label_header = 'neu_value'
    # saved_test_file = 'Data/RSV_v2/0215_RSV_neu_absplit_tst.csv'
    
    # original_train_file = 'Data/RSV/0212_RSV_neu.csv'
    # label_header = 'neu_value'
    # saved_test_file = 'Data/RSV/0212_RSV_neu_absplit_tst.csv'
    
    header_str = 'Heavy'
    # test_num = 100
    test_num = 50
    fold_num = 5


    process_and_split_data(original_train_file, header_str,label_header, test_num, fold_num, saved_test_file)
