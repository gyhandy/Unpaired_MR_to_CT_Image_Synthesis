import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
from PIL import Image
import md

from torchvision.transforms import functional as F


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # test
    Data_root_3d = '/data0/geyunhao/MR2CT_SAMPLING/'
    # nct_3d_samp = md.read_image(Data_root_3d + 'ZS10307488/nfct.nii.gz')
    # nct_3d_samp = md.read_image(Data_root_3d + 'ZS18111863/t1_wfi_wb_IP_Tra_9.nii') #384 * 549
    nct_3d_samp = md.read_image(Data_root_3d + 'ZS18158187/T1_WFI_WB_IP_Tra_8.nii')  # 384 * 549


    # for j in range(4): # crop 4 times for the randomcrop in training
    #     opt.test_crop_mode == str(j)

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
        out_slice_np = np.array( F.resize(out_slice_PIL, (512, 512), 3) )[np.newaxis, :]
        # out_slice_np = np.array(F.resize(out_slice_PIL, (384, 549), 3))[np.newaxis, :]
        out_slice_np = np.pad(out_slice_np[:, 64: 448, :], ((0, 0), (0, 0), (18, 19)), 'constant', constant_values=-1) # recover to(1*384*548)
        if i == 0:
            test_output_array = out_slice_np
        else:
            test_output_array = np.concatenate((test_output_array, out_slice_np), axis=0)



    nct_3d_samp.from_numpy(test_output_array)
    # md.write_image(mr_3d_samp, webpage.web_dir + '/fake_nfct_' + img_path[0].split('_')[-2] + '.mhd')
    md.write_image(nct_3d_samp, webpage.web_dir + '/' + opt.name + '.mhd')

    webpage.save()
