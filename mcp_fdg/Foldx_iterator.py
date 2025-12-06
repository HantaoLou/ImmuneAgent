import time
from FDG_function import *
from Bio.PDB import PDBParser, PDBIO

def write_down(filename,time_keeper):
    with open(filename, 'w', encoding='utf-8') as file:
        # 遍历嵌套列表中的每个子列表
        for sublist in time_keeper:
            # 将子列表中的元素转换为字符串，并用逗号分隔，然后写入文件，每个子列表后跟一个换行符
            file.write(sublist + '\n')

def foldx_analyse(pdb_name):
    #pdb_name="origin_1"
    os.system(f"cd temp/foldx && ./foldx  --command=AnalyseComplex --pdb={pdb_name}.pdb  --analyseComplexChains=HL,A")
    inter_data=pd.read_csv(f"temp/foldx/Interaction_{pdb_name}_AC.fxout",sep="	", skiprows=8, engine='python')
    inter_data=inter_data[["Pdb","Interaction Energy","Interface Residues"]]
    inter_residue=list(pd.read_csv(f"temp/foldx/Interface_Residues_{pdb_name}_AC.fxout",sep="\t", skiprows=9, engine='python').index[0])
    epitope=[i for i in inter_residue if i[1]=="A"]
    paratope=[i for i in inter_residue if i[1]=="H"]
    inter_data["epitope"],inter_data["paratope"]=",".join(epitope),",".join(paratope)
    return inter_data

def Foldx_mutations(WT_pdb,mutations): #代表原始pdb结构和突变数
    foldx_Data = foldx_helper(WT_pdb, mutations, is_repair=False).reset_index(drop=True)  # 已经修复就不用管了
    data_1 = foldx_Data.rename(columns={"foldx_energy": "foldx_energy(buildModel)"})
    origin_foldx_data = foldx_analyse(f"origin")  # 记录原始的energy
    for i in data_1.index:
        wt_path = f"temp/foldx/WT_origin_{i + 1}.pdb"
        mut_path = f"temp/foldx/origin_{i + 1}.pdb"
        # ddg
        ddg_pred = ddg_predict(wt_path, mut_path)
        # gearbind
        gearbind_pred = gearbind_predict(wt_path, mut_path)
        data_1.loc[i, "ddg_pred"] = ddg_pred
        data_1.loc[i, [f"gearbindp_{j + 1}" for j in range(5)]] = list(gearbind_pred.reshape(-1))
        # foldx
        mutant_foldx_data = foldx_analyse(f"origin_{i + 1}")
        data_1.loc[i, "foldx_Inter_ddg"] = round(
            mutant_foldx_data.loc[0, "Interaction Energy"] - origin_foldx_data.loc[0, "Interaction Energy"], 3)
        data_1.loc[i, ["epitope", "paratope"]] = mutant_foldx_data.loc[0, ["epitope", "paratope"]],
        # break
    data_1["gearbindp_mean"] = data_1[[f"gearbindp_{i + 1}" for i in range(5)]].mean(axis=1)
    return data_1

#这两个函数可以用来改链名
def change_name(input_path,output_path,old_name,new_name):
    parser = PDBParser()
    structure = parser.get_structure('protein', input_path)
    for model in structure:
        for chain in model:
            if chain.get_id()==old_name:
                chain.id = new_name
                break
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)

def swap_name(input_path,output_path,nameA,nameB):
    change_name(input_path,output_path,nameA,"T")
    change_name(output_path,output_path,nameB,nameA)
    change_name(output_path,output_path,"T",nameB)

def foldx_ana_complex(ana_pdb):
    temp_path="temp/chain_inter_helper/"
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)  # 删除整个文件夹
        print(f"文件夹 '{temp_path}' 已删除。")
    os.makedirs(temp_path)
    print(f"文件夹 '{temp_path}' 已重新创建。")
    #移入foldx
    shutil.copy(f"./foldx_bin/foldx", f"{temp_path}/foldx")
    shutil.copy(f"./foldx_bin/rotabase.txt", f"{temp_path}/rotabase.txt")
    #移入抗体
    shutil.copy(ana_pdb,f"{temp_path}/origin.pdb")
    #
    parser = PDBParser()
    structure = parser.get_structure('protein', f"{temp_path}/origin.pdb")
    for model in structure:
        chains=[chain.get_id() for chain in model]
        print("chains:",chains)
    for chain in ["A","H","L"]:
        if chain not in chains:continue
    os.system(f"cd {temp_path} && ./foldx  --command=AnalyseComplex --pdb=origin.pdb  --analyseComplexChains=HL,A")
    inter_residue=list(pd.read_csv(f"{temp_path}/Interface_Residues_origin_AC.fxout",sep="\t", skiprows=9, engine='python').index[0])
    return chains,inter_residue
    #break

def FDG_main(Ab_name,WT_pdb,score_path,output_path,shm_num=200,turns=5,limitations=100,limit_value=-0.1):
    # 输入是一个pdb
    # Ab_name = "inf-28"
    # WT_pdb = f"input/0315_sevo_repair/{Ab_name}_Repair.pdb"
    # score_path = f"input/0315_sevo_results/{Ab_name}_scores.csv"
    # output_path = f"output/{Ab_name}_test/"

    # 判断文件夹是否存在
    if os.path.exists(output_path):
        # 如果文件夹存在，则删除它
        shutil.rmtree(output_path)

    # 创建新的文件夹
    os.makedirs(output_path)

    ##正式内容
    # shm_num=200 #代表每轮选择200+个
    # shm_num = 200  # 代表每轮选择top200+个
    # turns = 5  # 代表总共进行5轮
    # limitations = 100  # 当某一轮包含的突变超过这个数字时，使用random随机筛选limitations个进入下一轮迭代
    # limit_value = -0.1
    # 这个值调低就可以直接测试了

    random.seed(408)
    # 计算备用score_data库
    score_data = pd.read_csv(score_path)
    score_data = score_data.loc[score_data["seqid"] != "wt",]
    score_data.columns = ["mutation", "sevo_ll", "sevo_llt"]
    score_data["mutation"] = score_data["mutation"].apply(lambda x: f"{x[0]}H{x[1:]}")  # 本轮变化
    score_data["clone_id"] = Ab_name
    score_data = score_data.sort_values("sevo_llt", ascending=False).reset_index(drop=True)  #
    score_data["location"] = score_data["mutation"].apply(lambda x: int(x[2:-1]))
    scoreA = score_data.head(shm_num)
    scoreB = score_data
    score_data = pd.concat([scoreA, scoreB.drop_duplicates("location")]).drop_duplicates()
    # score_data=scoreA
    index_dict = dict(zip(score_data["mutation"], score_data["location"]))  # 只能选location更大的位置
    if "location" in score_data: del score_data["location"]
    print(f"considerd mutations:{len(score_data)}")  # 全部的备选shms
    # 展开多轮优化
    time_keeper = [Ab_name]
    mutations = [""]
    for cur_turn in range(1, turns + 1):
        start_time = time.time()
        print(f"cur turn:{cur_turn}")
        next_mutations = []
        for i in mutations:
            if i == "":
                for j in list(score_data["mutation"]):
                    next_mutations.append(f"{j}")
            else:
                now_mut = i.split(",")[-1]  # 最后一项
                for j in list(score_data["mutation"]):
                    if index_dict[j] > index_dict[now_mut]:
                        next_mutations.append(f"{i},{j}")
                    # 要求j一定要在i的最后一位的后面后面
        mutations = next_mutations
        if len(mutations) > limitations:
            mutations = random.sample(mutations, limitations)
        elif len(mutations) == 0:
            time_keeper.append("no mutations can satisfied the condition!stop!!!")
            break

        time_keeper.append(f"cur_turn:{cur_turn}-->mutations num:{len(mutations)}")
        # 为了避免数量太多，当这个值超过limitations时，仅保留limitations个
        # 进行突变并计算ddg
        # 主体部分直接写在一个函数里面
        data_1 = Foldx_mutations(WT_pdb, mutations)  # 在这里
        data_1["turn"] = cur_turn
        data_1["gearbindp_mean"] = data_1[[f"gearbindp_{i + 1}" for i in range(5)]].mean(axis=1)
        for j in range(5):
            del data_1[f"gearbindp_{j + 1}"]
        # 代表判断条件，为了方便改写在函数外面了
        data_1["check_1"] = data_1["foldx_Inter_ddg"] < limit_value
        data_1["check_2"] = data_1.apply(lambda x: x["ddg_pred"] < limit_value or x["gearbindp_mean"] < limit_value, axis=1)
        data_1["final_check"] = data_1.apply(lambda x: x["check_1"] and x["check_2"], axis=1)
        data_1.to_csv(f"{output_path}/{Ab_name}_turn={cur_turn}_summary.csv")
        # 记录本轮mutations
        data_1["final_check"] = True
        mutations = list(data_1.loc[data_1["final_check"], "mutant"])  # 下一轮只会在这一轮通过的抗体上修改
        end_time = time.time()
        time_keeper.append(f"cur_turn:{cur_turn}-->time used:{round((end_time - start_time) / 60, 2)}min")
        time_keeper.append("====")
        # 使用with语句打开文件，确保文件会被正确关闭
        filename = f"{output_path}/log.txt"
        write_down(filename, time_keeper)
        # break
    time_keeper.append("========")
    write_down(filename, time_keeper)


def main():
    """
    主函数，执行程序的主要逻辑。
    """
    wt_path = "./input/example/WT_INF-26_Repair_1.pdb"
    mutations = ["YH75D", "YH75Q,QH1E"]
    # 包含了生成突变后的结构等一系列的操作，最后返回一个列表
    results = Foldx_mutations(wt_path,mutations)
    # 代表一个WT_path和一组突变
    print("[shape]results:",results.shape)
    print("results:", results)
    #这个函数可以用来分析有哪些链以及接触位点
    chains, inter_residues = foldx_repair("./input/example/","temp/test")
    print("chains:",chains)
    print("inter_residues:",inter_residues)

if __name__ == "__main__":
    main()