#从哪里读取图片，输入最顶层的文件夹即可
img_path = "./imgs/"

#处理之后的图片放在哪里，文件夹必须存在
res_path = "./res/"

#出错的图片保存在哪里，文件夹必须存在
err_path = "./error/"

#是否保存识别的结果
save_res = True

#是否将原图进行裁剪，因为上面和下面有一些不需要的信息，可以通过这个配置来裁掉
crop_img = True
#上方裁掉多少
crop_top = 0.22
#下方裁掉多少
crop_bot = 0.2

#高级选项， 根据需要修改一般电脑有几个核心就写几，一般规律是越大越快，但是电脑性能不好可能作用会相反
bs = 8