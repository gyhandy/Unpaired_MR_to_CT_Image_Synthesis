import numpy as np
import md
import os
from PIL import Image
import matplotlib
import scipy.misc
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from PIL import Image
from torchvision.transforms import functional as F
import shutil

# Data_root_3d = '/data0/geyunhao/MR2CT/ZS10307488/'
# Data_root_3d = '/data0/geyunhao/MR2CT_SAMPLING/ZS18111863/'
# Data_root_3d = '/data0/geyunhao/MR2CT_SAMPLING/ZS18158187/'
Data_root_3d = '/data0/geyunhao/MR2CT_X/' # whole testing data 46 patients
# Data_sam_root = '/data0/geyunhao/MR2CT_SAMPLING/'
# Data_root_3d = '/data0/geyunhao/MR2CT/'
Data_root_2d = '/home/geyunhao/Mapping/Mapping/pytorch-CycleGAN-and-pix2pix-master/datasets/test_pipline/'
DATA_NAME = 'test' # train, test, val, ex
IMAGE_TYPE = '.png'


'''
    Turn the 3D slice to 2D slice and satisfy the structure of cycleGAN
'''
# save to .png

target_path_mr = Data_root_2d + DATA_NAME + 'A'
target_path_nct = Data_root_2d + DATA_NAME + 'B'
if not os.path.exists(target_path_mr):
    os.makedirs(target_path_mr)
if not os.path.exists(target_path_nct):
    os.makedirs(target_path_nct)


'''
load model
'''
opt = TestOptions().parse()
model = create_model(opt)
model.setup(opt)

for roots, dirs, files in os.walk(Data_root_3d):
    for file in files:
        file_path = os.path.join(roots, file)
        if 'IP' in file:

            '''
            remove the last type
            '''
            shutil.rmtree(target_path_mr)
            if not os.path.exists(target_path_mr):
                os.makedirs(target_path_mr)
            mr_3d = md.read_image(file_path)  # (x,y,z)
            mr_3d_np = mr_3d.to_numpy() # (z,y,x)
            for i in range(mr_3d_np.shape[0]): # z

                '''
                get part of data
                '''
                # the approximately leg part in mri is range (0-60)
                # the approximately pelvicum part in mri is range (70-230)
                # the approximately lib part in mri is range (200-360)
                # if i >= 70 and i <= 230:
                '''
                whole body
                '''
                slice0 = np.expand_dims(mr_3d_np[i, :, :], 0)  # get slice data without decay (1,y,x)
                '''
                resize the mr
                '''
                # resize[1, 384, 549] to [1, 384, 548] for net_G
                slice0 = slice0[:, :, :-1]
                # resize[1, 384, 548] to [1, 512, 512]
                slice0 = slice0[:, :, 18:530]
                slice0 = np.pad(slice0, ((0, 0), (64, 64), (0, 0)), 'constant', constant_values=0)
                '''
                normalize 0-255
                '''
                mrIP_intensity_min = np.float32(0.0)
                mrIP_intensity_max = np.float32(400.0)

                #  cut off the image
                slice0[slice0 > mrIP_intensity_max] = mrIP_intensity_max
                slice0[slice0 < mrIP_intensity_min] = mrIP_intensity_min
                slice0 = (slice0 - mrIP_intensity_min) / (mrIP_intensity_max - mrIP_intensity_min) * 255

                # nct_ref.from_numpy(slice0)  # put slice data into ref

                # im = Image.fromarray(slice0)
                target_path = target_path_mr
                target_filename = file_path.split('/')[-2]  + '_'+ str(i) + IMAGE_TYPE
                if not os.path.exists(target_path + '/' + target_filename):
                    scipy.misc.imsave(target_path + '/' + target_filename, slice0[0])
                    # im.save(target_path + '/' + target_filename)

            '''
            test
            '''
            opt = TestOptions().parse()
            opt.nThreads = 1  # test code only supports nThreads = 1
            opt.batchSize = 1  # test code only supports batchSize = 1
            opt.serial_batches = True  # no shuffle
            opt.no_flip = True  # no flip
            opt.display_id = -1  # no visdom display
            data_loader = CreateDataLoader(opt)
            dataset = data_loader.load_data()
            # # create website
            # web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
            # webpage = html.HTML(web_dir,
            #                     'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

            nct_3d_samp = mr_3d  # 384 * 549
            for i, data in enumerate(dataset):
                # if i >= opt.how_many:
                if i >= dataset.__len__():
                    break
                model.set_input(data)
                model.test()
                visuals = model.get_current_visuals()
                img_path = model.get_image_paths()
                if i % 5 == 0:
                    print('processing (%04d)-th image... %s' % (i, img_path))
                # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

                '''
                save to .mhd
                '''
                out_slice_np = visuals['fake_B'].squeeze(0).cpu().numpy()
                out_slice_PIL = Image.fromarray(out_slice_np[0])
                out_slice_np = np.array(F.resize(out_slice_PIL, (512, 512), 3))[np.newaxis, :]
                # out_slice_np = np.array(F.resize(out_slice_PIL, (384, 549), 3))[np.newaxis, :]
                out_slice_np = np.pad(out_slice_np[:, 64: 448, :], ((0, 0), (0, 0), (18, 19)), 'constant',
                                      constant_values=-1)  # recover to(1*384*548)
                if i == 0:
                    test_output_array = out_slice_np
                else:
                    test_output_array = np.concatenate((test_output_array, out_slice_np), axis=0)

            nct_3d_samp.from_numpy(test_output_array)
            # md.write_image(mr_3d_samp, webpage.web_dir + '/fake_nfct_' + img_path[0].split('_')[-2] + '.mhd')
            # md.write_image(nct_3d_samp, webpage.web_dir + '/' + opt.name + '.mhd')
            md.write_image(nct_3d_samp, './results/pure_cyclegan/' +  file_path.split('/')[-2] + '.mhd')

            print(file_path.split('/')[-2] +" tested!")

            # webpage.save()








#
# '''
# test and save image
# '''
# opt = TestOptions().parse()
# opt.nThreads = 1   # test code only supports nThreads = 1
# opt.batchSize = 1  # test code only supports batchSize = 1
# opt.serial_batches = True  # no shuffle
# opt.no_flip = True  # no flip
# opt.display_id = -1  # no visdom display
# data_loader = CreateDataLoader(opt)
# dataset = data_loader.load_data()
# model = create_model(opt)
# model.setup(opt)
# # create website
# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
#
# # test
# Data_root_3d = '/data0/geyunhao/MR2CT_SAMPLING/'
# # nct_3d_samp = md.read_image(Data_root_3d + 'ZS10307488/nfct.nii.gz')
# # nct_3d_samp = md.read_image(Data_root_3d + 'ZS18111863/t1_wfi_wb_IP_Tra_9.nii') #384 * 549
# nct_3d_samp = md.read_image(Data_root_3d + 'ZS18158187/T1_WFI_WB_IP_Tra_8.nii')  # 384 * 549
#
# # for j in range(4): # crop 4 times for the randomcrop in training
# #     opt.test_crop_mode == str(j)
#
# for i, data in enumerate(dataset):
#     # if i >= opt.how_many:
#     if i >= dataset.__len__():
#         break
#     model.set_input(data)
#     model.test()
#     visuals = model.get_current_visuals()
#     img_path = model.get_image_paths()
#     if i % 5 == 0:
#         print('processing (%04d)-th image... %s' % (i, img_path))
#     # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
#
#     '''
#     save to .mhd
#     '''
#     out_slice_np = visuals['fake_B'].squeeze(0).cpu().numpy()
#     out_slice_PIL = Image.fromarray(out_slice_np[0])
#     out_slice_np = np.array(F.resize(out_slice_PIL, (512, 512), 3))[np.newaxis, :]
#     # out_slice_np = np.array(F.resize(out_slice_PIL, (384, 549), 3))[np.newaxis, :]
#     out_slice_np = np.pad(out_slice_np[:, 64: 448, :], ((0, 0), (0, 0), (18, 19)), 'constant',
#                           constant_values=-1)  # recover to(1*384*548)
#     if i == 0:
#         test_output_array = out_slice_np
#     else:
#         test_output_array = np.concatenate((test_output_array, out_slice_np), axis=0)
#
# nct_3d_samp.from_numpy(test_output_array)
# # md.write_image(mr_3d_samp, webpage.web_dir + '/fake_nfct_' + img_path[0].split('_')[-2] + '.mhd')
# md.write_image(nct_3d_samp, webpage.web_dir + '/' + opt.name + '.mhd')
#
# webpage.save()
#
#
