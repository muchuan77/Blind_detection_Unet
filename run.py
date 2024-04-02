import subprocess


def train_model():
    train_mode = input("请选择训练模式（1：使用config.json中的backbone, 2：遍历backbones列表, 3: 选择backbone）: ")
    if train_mode == "1":
        subprocess.call(['python', 'train.py', '--mode', 'config'])
    elif train_mode == "2":
        subprocess.call(['python', 'train.py', '--mode', 'list'])
    elif train_mode == "3":
        backbone = input("请输入backbone名称（vgg或resnet）: ")
        subprocess.call(['python', 'train.py', '--mode', 'select', '--backbone', backbone])
    else:
        print("未知的模式，请输入1、2或3。")


def test_model():
    backbone = input("请选择模型（vgg 或 resnet）: ")
    data_type = input("请选择数据集类型（train 或 val）: ")
    subprocess.call(['python', 'test.py', '--backbone', backbone, '--data_type', data_type])


def main_menu():
    while True:
        decision = input("请选择操作：1-训练模型, 2-测试模型, 3-退出: ")
        if decision == "1":
            train_model()
        elif decision == "2":
            test_model()
        elif decision == "3":
            break
        else:
            print("无效输入，请输入1、2或3。")


if __name__ == "__main__":
    main_menu()
