# import matplotlib
# matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
# import sys
# sys.stdout.reconfigure(encoding='utf-8')
# from util.utils import *
# from sklearn.preprocessing import normalize
# import os
# import shutil
# import numpy as np
# import torch
# from torchvision import transforms
# from IPython import embed
# import models
# from scipy.spatial.distance import euclidean
# from util.utils import read_image, img_to_tensor  # 确保这些函数已正确导入
# from util.FeatureExtractor import FeatureExtractor
#
# # 固定需要查找的图像 img_path1
# img_path1 = './yolov8-deepsort-tracking/output/crops/c1s1/0001_c1s1_000003_01.jpg'
#
# # 遍历 query 文件夹中的所有图片
# query_folder = './yolov8-deepsort-tracking/output/crops/c1s1'
#
# # 设置输出文件夹路径
# output_folder = "./output_similar_images/0929"  # 替换为您希望保存的文件夹路径
# os.makedirs(output_folder, exist_ok=True)
#
# def pool2d(tensor, type= 'max'):
#     sz = tensor.size()
#     if type == 'max':
#         kernel_size = (int(sz[2] // 8), int(sz[3]))
#         x = torch.nn.functional.max_pool2d(tensor, kernel_size=kernel_size)
#     if type == 'mean':
#         kernel_size = (int(sz[2] // 8), int(sz[3]))
#         x = torch.nn.functional.mean_pool2d(tensor, kernel_size=kernel_size)
#     x = x[0].cpu().data.numpy()
#     x = np.transpose(x,(2,1,0))[0]
#     return x
#
# def main():
#     os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#     use_gpu = torch.cuda.is_available()
#
#     # 初始化模型
#     model = models.init_model(name='resnet50', num_classes=751, loss={'softmax', 'metric'}, use_gpu=use_gpu, aligned=True)
#     checkpoint = torch.load("./log/market1501/alignedreid/checkpoint_ep300.pth.tar", map_location="cpu", weights_only=False)
#     model.load_state_dict(checkpoint['state_dict'])
#
#     # 图像预处理
#     img_transform = transforms.Compose([
#         transforms.Resize((256, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     # # 固定 img_path1
#     # img_path1 = './data/market1501/query/0001_c1s1_001051_00.jpg'
#     exact_list = ['7']
#     myexactor = FeatureExtractor(model, exact_list)
#
#     # 处理 img_path1 的特征（仅需一次）
#     img1 = read_image(img_path1)
#     img1_tensor = img_to_tensor(img1, img_transform)
#     if use_gpu:
#         model = model.cuda()
#         img1_tensor = img1_tensor.cuda()
#     model.eval()
#     f1 = myexactor(img1_tensor)
#     a1 = normalize(pool2d(f1[0], type='max'))  # 提取并归一化特征
#
#     # # 遍历 query 文件夹中的所有图片
#     # query_folder = './data/market1501/query'
#     for img_name in os.listdir(query_folder):
#         img_path2 = os.path.join(query_folder, img_name)
#
#
#         # 跳过 img_path1 本身（可选）
#         if img_path2 == img_path1:
#             continue
#
#         # 处理 img_path2 的特征
#         img2 = read_image(img_path2)
#         img2_tensor = img_to_tensor(img2, img_transform)
#         if use_gpu:
#             img2_tensor = img2_tensor.cuda()
#         f2 = myexactor(img2_tensor)
#         a2 = normalize(pool2d(f2[0], type='max'))
#
#         # 计算对齐距离（取 8x8 距离矩阵的平均值）
#         dist = np.zeros((8, 8))
#         for i in range(8):
#             temp_feat1 = a1[i]
#             for j in range(8):
#                 temp_feat2 = a2[j]
#                 dist[i][j] = euclidean(temp_feat1, temp_feat2)
#         aligned_distance = np.mean(dist)  # 使用平均距离作为判断依据
#
#         print(f"img_path2 {img_path2}（距离: {aligned_distance:.4f}）")
#         # 判断并复制符合条件的图片
#         if aligned_distance < 0.8:
#             output_path = os.path.join(output_folder, img_name)
#             shutil.copy2(img_path2, output_path)
#             print(f"已复制 {img_name}（距离: {aligned_distance:.4f}）")
#
#
#
# if __name__ == '__main__':
#     main()

def run_find_most(query_path, crops_dir, output_dir):
    import os
    import shutil
    import numpy as np
    import torch
    from torchvision import transforms
    from sklearn.preprocessing import normalize
    from scipy.spatial.distance import euclidean

    import models
    from util.utils import read_image, img_to_tensor
    from util.FeatureExtractor import FeatureExtractor

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    use_gpu = torch.cuda.is_available()

    # 初始化模型
    model = models.init_model(name='resnet50', num_classes=751, loss={'softmax', 'metric'}, use_gpu=use_gpu, aligned=True)
    checkpoint = torch.load("./log/market1501/alignedreid/checkpoint_ep300.pth.tar",
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    img_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    exact_list = ['7']
    myexactor = FeatureExtractor(model, exact_list)

    # 处理 query 图像
    img1 = read_image(query_path)
    img1_tensor = img_to_tensor(img1, img_transform)
    if use_gpu:
        model = model.cuda()
        img1_tensor = img1_tensor.cuda()
    model.eval()

    def pool2d(tensor, type='max'):
        sz = tensor.size()
        if type == 'max':
            kernel_size = (int(sz[2] // 8), int(sz[3]))
            x = torch.nn.functional.max_pool2d(tensor, kernel_size=kernel_size)
        if type == 'mean':
            kernel_size = (int(sz[2] // 8), int(sz[3]))
            x = torch.nn.functional.avg_pool2d(tensor, kernel_size=kernel_size)
        x = x[0].cpu().data.numpy()
        x = np.transpose(x, (2, 1, 0))[0]
        return x

    f1 = myexactor(img1_tensor)
    a1 = normalize(pool2d(f1[0], type='max'))
    cnt = 0

    # 遍历 crops_dir
    for img_name in os.listdir(crops_dir):
        img_path2 = os.path.join(crops_dir, img_name)
        if img_path2 == query_path:
            continue
        img2 = read_image(img_path2)
        img2_tensor = img_to_tensor(img2, img_transform)
        if use_gpu:
            img2_tensor = img2_tensor.cuda()
        f2 = myexactor(img2_tensor)
        a2 = normalize(pool2d(f2[0], type='max'))

        dist = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                dist[i][j] = euclidean(a1[i], a2[j])
        aligned_distance = np.mean(dist)

        print(f"img_path2 {img_path2}（距离: {aligned_distance:.4f}）")
        if aligned_distance < 0.8:
            output_path = os.path.join(output_dir, img_name)
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(img_path2, output_path)
            print(f"已复制 {img_name}（距离: {aligned_distance:.4f}）")
            cnt+=1
    print(f"成功！！！ 已复制{cnt}张图片到{output_dir}")

if __name__ == '__main__':
    # 默认参数，兼容之前直接运行脚本
    query_path = './yolov8-deepsort-tracking/output/crops/c1s1/0001_c1s1_000003_01.jpg'
    crops_dir = './yolov8-deepsort-tracking/output/crops/c1s1'
    output_dir = './output_similar_images/0929'
    run_find_most(query_path, crops_dir, output_dir)
